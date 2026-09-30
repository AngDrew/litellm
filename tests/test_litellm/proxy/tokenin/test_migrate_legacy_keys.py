from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Final

import pytest

_SPEC: Final = importlib.util.spec_from_file_location(
    "tokenin_migrate_legacy_keys",
    Path(__file__).resolve().parents[4] / "scripts/tokenin_migrate_legacy_keys.py",
)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = migration  # dataclasses look the module up while building
_SPEC.loader.exec_module(migration)

NOW: Final = datetime(2026, 9, 27, 12, 0, 0)
ACCOUNT: Final = "account-1"
GRANT_ID: Final = f"legacy-carry:{ACCOUNT}"


TEAM_MODELS: Final = ["model-b", "model-a"]


class FakeDB:
    """Prisma-shaped stand-in honouring the team filter, the grant key and one policy per user."""

    def __init__(
        self,
        keys: list[dict[str, object]],
        grants: dict[str, int] | None = None,
        team_models: list[str] | None = TEAM_MODELS,
        policies: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.keys = keys
        self.grants = dict(grants or {})
        self.team_models = team_models
        self.policies = dict(policies or {})  # user_id -> policy
        self.accounts: set[str] = set()
        self.cleared_tokens: list[object] = []
        self.executed: list[str] = []

    async def query_raw(self, sql: str, *values: object) -> list[dict[str, object]]:
        if '"LiteLLM_VerificationToken"' in sql:
            return [dict(row) for row in self.keys if row["team_id"] == values[0]]
        if '"LiteLLM_TeamTable"' in sql:
            assert values[0] == migration.FLAT_TEAM_ID
            return [] if self.team_models is None else [{"models": list(self.team_models)}]
        if '"LiteLLM_TokeninGrant"' in sql:
            return [{"idempotency_key": key, "amount_nano": amount} for key, amount in self.grants.items()]
        raise AssertionError(sql)

    async def execute_raw(self, sql: str, *values: object) -> int:
        self.executed.append(sql)
        if 'INSERT INTO "LiteLLM_TokeninAccount"' in sql:
            self.accounts.add(str(values[0]))
        elif 'INSERT INTO "LiteLLM_TokeninPolicy"' in sql:
            key, user_id, _hash, plan_id, models, rpm, parallel, effective = values
            self.policies.setdefault(  # WHERE NOT EXISTS (any policy for the user)
                str(user_id),
                {"idempotency_key": key, "plan_id": plan_id, "models": models, "rpm_limit": rpm,
                 "max_parallel_requests": parallel, "effective_at": effective},
            )
        elif 'INSERT INTO "LiteLLM_TokeninGrant"' in sql:
            grant_id, _user_id, _hash, _plan, _kind, amount = values
            self.grants.setdefault(str(grant_id), int(amount))  # ON CONFLICT DO NOTHING
        elif 'UPDATE "LiteLLM_VerificationToken"' in sql:
            self.cleared_tokens.extend(values[0])  # type: ignore[arg-type]
        return 1


def _key(
    token: str,
    budget: float | None,
    spend: float,
    expires: datetime | None = None,
    user_id: str = ACCOUNT,
    team_id: str | None = None,
) -> dict[str, object]:
    return {
        "token": token,
        "user_id": user_id,
        "team_id": migration.FLAT_TEAM_ID if team_id is None else team_id,
        "max_budget": budget,
        "spend": spend,
        "expires": expires,
    }


@pytest.mark.asyncio
async def test_expired_unbudgeted_and_overspent_keys_are_excluded() -> None:
    keys: Final = [
        _key("sk-live", 10.0, 4.0),
        _key("sk-overdrawn", 5.0, 7.5),
        _key("sk-expired", 50.0, 0.0, expires=NOW - timedelta(days=1)),
        _key("sk-uncapped", None, 3.0),
        _key("sk-zero", 2.0, 2.0),
    ]
    carry, counted, skipped = migration.carry_over(rows=keys, now=NOW)
    assert carry == Decimal("6")
    assert counted == ["sk-live"]
    assert skipped == [
        "sk-overdrawn:no-remaining-value",
        "sk-expired:expired",
        "sk-uncapped:no-key-budget",
        "sk-zero:no-remaining-value",
    ]
    plans: Final = migration.plan_accounts(rows=keys, existing={}, now=NOW)
    assert len(plans) == 1
    assert plans[0].amount_nano == 6_000_000_000
    assert plans[0].action == "would_create"


@pytest.mark.asyncio
async def test_dry_run_reports_and_writes_nothing() -> None:
    db: Final = FakeDB([_key("sk-live", 10.0, 4.0)])
    lines: Final[list[str]] = []
    summary: Final = await migration.migrate(
        client=db, apply=False, clear_legacy_key_budgets=False, now=NOW, log=lines.append
    )
    assert db.executed == []
    assert summary["created"] == 1
    assert summary["carry_usd"] == "6"
    assert summary["applied"] is False
    assert any(line.startswith("AUDIT ") for line in lines)
    assert json.loads(lines[-1][len("AUDIT ") :])["accounts"] == 1


@pytest.mark.asyncio
async def test_apply_is_idempotent_across_reruns() -> None:
    db: Final = FakeDB([_key("sk-live", 10.0, 4.0), _key("sk-live-2", 1.5, 0.5)])
    first: Final = await migration.migrate(
        client=db, apply=True, clear_legacy_key_budgets=False, now=NOW, log=lambda _line: None
    )
    assert first["created"] == 1
    assert db.grants == {GRANT_ID: 7_000_000_000}
    assert db.accounts == {ACCOUNT}
    rerun: Final = await migration.migrate(
        client=db, apply=True, clear_legacy_key_budgets=False, now=NOW, log=lambda _line: None
    )
    assert rerun["created"] == 0
    assert rerun["existing"] == 1
    assert rerun["conflicts"] == []
    assert db.grants == {GRANT_ID: 7_000_000_000}
    assert sum(sql.startswith('INSERT INTO "LiteLLM_TokeninGrant"') for sql in db.executed) == 2


@pytest.mark.asyncio
async def test_rerun_with_a_different_amount_reports_instead_of_writing() -> None:
    db: Final = FakeDB([_key("sk-live", 10.0, 4.0)], grants={GRANT_ID: 5_000_000_000})
    summary: Final = await migration.migrate(
        client=db, apply=True, clear_legacy_key_budgets=True, now=NOW, log=lambda _line: None
    )
    assert summary["conflicts"] == [{"user_id": ACCOUNT, "detail": "recorded 5000000000 != recomputed 6000000000"}]
    assert db.grants == {GRANT_ID: 5_000_000_000}
    assert db.accounts == set()
    assert db.cleared_tokens == []
    assert 'INSERT INTO "LiteLLM_TokeninGrant"' not in db.executed


@pytest.mark.asyncio
async def test_clearing_legacy_key_budgets_is_opt_in_and_covers_every_active_key() -> None:
    db: Final = FakeDB([_key("sk-live", 10.0, 4.0), _key("sk-overdrawn", 1.0, 9.0)])
    await migration.migrate(client=db, apply=True, clear_legacy_key_budgets=False, now=NOW, log=lambda _line: None)
    assert db.cleared_tokens == []
    await migration.migrate(client=db, apply=True, clear_legacy_key_budgets=True, now=NOW, log=lambda _line: None)
    assert sorted(db.cleared_tokens) == ["sk-live", "sk-overdrawn"]


@pytest.mark.asyncio
async def test_zero_carry_account_is_still_enrolled_so_its_key_caps_cannot_double_constrain() -> None:
    """A spent-out key plus a fresh account purchase must not be denied by the old cap."""
    db: Final = FakeDB([_key("sk-zero", 3.0, 3.0)])
    skipped: Final = await migration.migrate(
        client=db, apply=True, clear_legacy_key_budgets=False, now=NOW, log=lambda _line: None
    )
    assert skipped["skipped"] == 1
    assert skipped["created"] == 0
    assert db.grants == {}
    assert db.accounts == {ACCOUNT}
    assert db.cleared_tokens == []
    await migration.migrate(client=db, apply=True, clear_legacy_key_budgets=True, now=NOW, log=lambda _line: None)
    assert db.cleared_tokens == ["sk-zero"]
    assert db.grants == {}


@pytest.mark.asyncio
async def test_uncapped_only_account_is_left_untouched_and_listed_for_a_decision() -> None:
    db: Final = FakeDB([_key("sk-unlimited", None, 120.0)])
    summary: Final = await migration.migrate(
        client=db, apply=True, clear_legacy_key_budgets=True, now=NOW, log=lambda _line: None
    )
    assert db.executed == []
    assert db.accounts == set()
    assert db.grants == {}
    assert db.cleared_tokens == []
    assert summary["uncapped_accounts"] == [ACCOUNT]
    assert summary["accounts_detail"][0]["detail"] == "uncapped, needs operator decision"


@pytest.mark.asyncio
async def test_mixed_uncapped_and_capped_account_enrols_and_clears_only_the_capped_keys() -> None:
    db: Final = FakeDB([_key("sk-capped", 10.0, 4.0), _key("sk-unlimited", None, 50.0)])
    summary: Final = await migration.migrate(
        client=db, apply=True, clear_legacy_key_budgets=True, now=NOW, log=lambda _line: None
    )
    assert db.accounts == {ACCOUNT}
    assert db.grants == {GRANT_ID: 6_000_000_000}
    assert summary["carry_usd"] == "6"
    assert db.cleared_tokens == ["sk-capped"]
    assert summary["uncapped_accounts"] == []


@pytest.mark.asyncio
async def test_carrying_account_keeps_exactly_its_carried_amount_and_only_credits_live_keys() -> None:
    db: Final = FakeDB(
        [
            _key("sk-live", 10.0, 4.0),
            _key("sk-expired", 25.0, 0.0, expires=NOW - timedelta(hours=1)),
            _key("sk-overdrawn", 5.0, 9.0),
        ]
    )
    summary: Final = await migration.migrate(
        client=db, apply=True, clear_legacy_key_budgets=True, now=NOW, log=lambda _line: None
    )
    assert db.grants == {GRANT_ID: 6_000_000_000}
    assert summary["carry_usd"] == "6"
    assert summary["accounts_detail"][0]["keys"] == ["sk-live"]
    assert sorted(db.cleared_tokens) == ["sk-live", "sk-overdrawn"]


@pytest.mark.asyncio
async def test_internal_keys_outside_the_flat_team_are_never_carried_enrolled_or_cleared() -> None:
    db: Final = FakeDB(
        [
            _key("sk-customer", 10.0, 4.0),
            _key("sk-dashboard", 50.0, 0.0, user_id="admin", team_id="litellm-dashboard"),
        ]
    )
    summary: Final = await migration.migrate(
        client=db, apply=True, clear_legacy_key_budgets=True, now=NOW, log=lambda _line: None
    )
    assert db.accounts == {ACCOUNT}
    assert db.grants == {GRANT_ID: 6_000_000_000}
    assert db.cleared_tokens == ["sk-customer"]
    assert set(db.policies) == {ACCOUNT}
    assert summary["carry_usd"] == "6"


@pytest.mark.asyncio
async def test_every_enrolled_account_gets_a_payg_policy_with_the_flat_team_models_once() -> None:
    db: Final = FakeDB([_key("sk-live", 10.0, 4.0), _key("sk-zero", 3.0, 3.0, user_id="account-2")])
    for _ in range(2):
        await migration.migrate(client=db, apply=True, clear_legacy_key_budgets=False, now=NOW, log=lambda _line: None)
    assert set(db.policies) == {ACCOUNT, "account-2"}
    policy: Final = db.policies[ACCOUNT]
    assert policy["idempotency_key"] == f"legacy-carry-policy:{ACCOUNT}"
    assert policy["plan_id"] == "payg"
    assert policy["models"] == ["model-a", "model-b"]
    assert (policy["rpm_limit"], policy["max_parallel_requests"]) == (120, 5)
    assert policy["effective_at"] == NOW.isoformat()


@pytest.mark.asyncio
async def test_an_existing_account_policy_is_kept() -> None:
    admin_policy: Final = {"idempotency_key": "admin-set", "models": ["model-a"]}
    db: Final = FakeDB([_key("sk-live", 10.0, 4.0)], policies={ACCOUNT: admin_policy})
    await migration.migrate(client=db, apply=True, clear_legacy_key_budgets=False, now=NOW, log=lambda _line: None)
    assert db.policies == {ACCOUNT: admin_policy}


@pytest.mark.asyncio
async def test_apply_refuses_to_enrol_when_the_flat_team_has_no_models() -> None:
    db: Final = FakeDB([_key("sk-live", 10.0, 4.0)], team_models=[])
    dry: Final = await migration.migrate(
        client=db, apply=False, clear_legacy_key_budgets=False, now=NOW, log=lambda _line: None
    )
    assert dry["policy_models"] == []
    with pytest.raises(RuntimeError, match="no models"):
        await migration.migrate(client=db, apply=True, clear_legacy_key_budgets=True, now=NOW, log=lambda _line: None)
    assert db.executed == []


def test_non_local_target_needs_force_prod_and_an_explicit_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote: Final = "postgresql://user:pass@db.example.com:5432/litellm"
    assert migration.refuse_unsafe_target("postgresql://user:pass@127.0.0.1:5432/litellm", False) is None
    assert migration.refuse_unsafe_target("postgresql://user:pass@localhost:5432/litellm", True) is None
    assert migration.refuse_unsafe_target(remote, False) is not None
    monkeypatch.delenv("TOKENIN_MIGRATION_ALLOW_PROD", raising=False)
    assert migration.refuse_unsafe_target(remote, True) is not None
    monkeypatch.setenv("TOKENIN_MIGRATION_ALLOW_PROD", "true")
    assert migration.refuse_unsafe_target(remote, True) is None
