"""
Regression tests for the tokenin account wallet.

One wallet per account (the user row's `max_budget`), credited by
`/tokenin/key/generate` and `/tokenin/key/update` and enforced for the account's
flat-team keys through `general_settings.apply_user_budget_to_team_keys`. No key
carries a budget, a budget window, or an expiry of its own.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final, NamedTuple, TypedDict

import pytest
from fastapi import HTTPException

from litellm.models.team import LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.budget_reservation import _BudgetCounter, _get_budget_counters
from litellm.proxy.tokenin import plans as tokenin_plans
from litellm.proxy.tokenin.plans import TokeninPlan
from litellm.proxy.tokenin.router import (
    TokeninGenerateRequest,
    TokeninUpdateRequest,
    tokenin_generate_key,
    tokenin_update_key,
)

ACCOUNT_ID: Final = "acct-1"
FLAT_TEAM_ID: Final = "ac0b4e54-71a7-4e1f-bfaf-32fad13c09e9"

FIXED_PLAN: Final = TokeninPlan(
    id="medium",
    name="Medium",
    kind="fixed",
    rpm_limit=120,
    max_parallel_requests=2,
    max_budget=10,
    budget_duration="30d",
)
PAYG_PLAN: Final = TokeninPlan(
    id="payg",
    name="Pay As You Go",
    kind="payg",
    rpm_limit=120,
    max_parallel_requests=5,
    max_budget=0,
    budget_duration="1000d",
)


class WalletUserRow(TypedDict):
    user_id: str
    max_budget: float
    models: list[str]


class WalletIncrement(NamedTuple):
    user_id: str
    amount: float


class FakeUserTable:
    """User table stand-in that honors the two write shapes the wallet uses."""

    def __init__(self) -> None:
        self.wallets: dict[str, dict[str, float | None]] = {}
        self.increments: list[WalletIncrement] = []

    async def update_many(self, *, where: dict[str, str | None], data: dict[str, float]) -> int:
        row = self.wallets.get(where["user_id"] or "")
        if row is None or row["max_budget"] is not None:
            return 0
        row["max_budget"] = data["max_budget"]
        return 1

    async def update(self, *, where: dict[str, str], data: dict[str, dict[str, float]]) -> SimpleNamespace:
        row = self.wallets[where["user_id"]]
        increment: Final = data["max_budget"]["increment"]
        self.increments.append(WalletIncrement(user_id=where["user_id"], amount=increment))
        row["max_budget"] = (row["max_budget"] or 0.0) + increment
        return SimpleNamespace(max_budget=row["max_budget"])


class FakePrisma:
    """Only the tables the wallet touches. Any key-table write raises AttributeError."""

    def __init__(self) -> None:
        self.user_table: Final = FakeUserTable()
        self.db: Final = SimpleNamespace(litellm_usertable=self.user_table)
        self.user_row_creations: list[WalletUserRow] = []

    async def insert_data(self, data: WalletUserRow, table_name: str) -> None:
        assert table_name == "user"
        self.user_row_creations.append(data)
        self.user_table.wallets.setdefault(
            data["user_id"], {"max_budget": data["max_budget"], "spend": 0.0}
        )


class FakeUserApiKeyCache:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def async_delete_cache(self, key: str) -> None:
        self.deleted.append(key)

    async def async_get_cache(self, key: str, **kwargs: object) -> None:
        return None


RecordedKeyPayload = dict[str, str | float | list[str]]


def _install(
    monkeypatch: pytest.MonkeyPatch,
    plans: tuple[TokeninPlan, ...] = (FIXED_PLAN, PAYG_PLAN),
    key_user_id: str | None = ACCOUNT_ID,
) -> tuple[FakePrisma, FakeUserApiKeyCache, list[RecordedKeyPayload]]:
    prisma: Final = FakePrisma()
    cache: Final = FakeUserApiKeyCache()
    generated: Final[list[RecordedKeyPayload]] = []

    monkeypatch.setattr(tokenin_plans, "load_plans", lambda: list(plans))
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", cache)

    async def fake_existing_key(token: str, prisma_client: object) -> SimpleNamespace:
        return SimpleNamespace(user_id=key_user_id)

    async def fake_generate_key_helper_fn(**kwargs: str | float | list[str]) -> dict[str, object]:
        generated.append(kwargs)
        return {
            "token": "sk-new",
            "token_id": "tok-1",
            "key_alias": kwargs["key_alias"],
            "key_name": "sk-new",
            "expires": None,
            "created_at": datetime(2026, 9, 13, tzinfo=timezone.utc),
        }

    monkeypatch.setattr(
        "litellm.proxy.tokenin.router._get_and_validate_existing_key", fake_existing_key
    )
    monkeypatch.setattr(
        "litellm.proxy.tokenin.router.generate_key_helper_fn", fake_generate_key_helper_fn
    )
    return prisma, cache, generated


async def test_generate_credits_the_wallet_and_creates_a_budgetless_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prisma, cache, generated = _install(monkeypatch)

    response: Final = await tokenin_generate_key(
        data=TokeninGenerateRequest(user_id=ACCOUNT_ID, plan_id="medium", alias="acct-1-key"),
    )

    assert prisma.user_table.wallets[ACCOUNT_ID]["max_budget"] == 10.0
    assert prisma.user_table.increments == [WalletIncrement(user_id=ACCOUNT_ID, amount=10.0)]
    assert cache.deleted == [ACCOUNT_ID]

    assert [creation["max_budget"] for creation in prisma.user_row_creations] == [0.0]
    for creation in prisma.user_row_creations:
        assert set(creation) <= {"user_id", "max_budget", "models"}

    key_payload: Final = generated[0]
    assert key_payload["user_id"] == ACCOUNT_ID
    assert key_payload["team_id"] == FLAT_TEAM_ID
    assert key_payload["table_name"] == "key"
    assert key_payload["rpm_limit"] == 120
    assert key_payload["max_parallel_requests"] == 2
    for absent in ("key_max_budget", "key_budget_duration", "budget_limits", "duration"):
        assert absent not in key_payload

    assert response.expires is None


async def test_fixed_plan_credit_ignores_a_caller_supplied_amount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prisma, _, _ = _install(monkeypatch)

    await tokenin_generate_key(
        data=TokeninGenerateRequest(
            user_id=ACCOUNT_ID,
            plan_id="medium",
            alias="acct-1-key",
            payg_initial_balance=999.0,
        ),
    )

    assert prisma.user_table.wallets[ACCOUNT_ID]["max_budget"] == 10.0


async def test_every_key_of_an_account_shares_the_one_wallet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prisma, _, generated = _install(monkeypatch)

    await tokenin_generate_key(
        data=TokeninGenerateRequest(user_id=ACCOUNT_ID, plan_id="medium", alias="key-1"),
    )
    await tokenin_generate_key(
        data=TokeninGenerateRequest(user_id=ACCOUNT_ID, plan_id="medium", alias="key-2"),
    )
    topup: Final = await tokenin_update_key(
        data=TokeninUpdateRequest(key="sk-second-key", plan_id="medium", action="extend"),
    )

    assert len(prisma.user_table.wallets) == 1
    assert prisma.user_table.wallets[ACCOUNT_ID]["max_budget"] == 30.0
    assert [payload["user_id"] for payload in generated] == [ACCOUNT_ID, ACCOUNT_ID]
    assert prisma.user_table.increments == [
        WalletIncrement(user_id=ACCOUNT_ID, amount=10.0),
        WalletIncrement(user_id=ACCOUNT_ID, amount=10.0),
        WalletIncrement(user_id=ACCOUNT_ID, amount=10.0),
    ]
    assert topup.credited == 10.0
    assert topup.max_budget == 30.0


async def test_credit_seeds_a_null_ceiling_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prisma, _, _ = _install(monkeypatch)
    prisma.user_table.wallets[ACCOUNT_ID] = {"max_budget": None, "spend": 0.0}

    await tokenin_update_key(
        data=TokeninUpdateRequest(key="sk-key", plan_id="medium", action="extend"),
    )
    await tokenin_update_key(
        data=TokeninUpdateRequest(key="sk-key", plan_id="medium", action="extend"),
    )

    assert prisma.user_table.wallets[ACCOUNT_ID]["max_budget"] == 20.0
    assert prisma.user_table.increments == [WalletIncrement(user_id=ACCOUNT_ID, amount=10.0)]


async def test_payg_topup_credits_the_paid_amount_and_rejects_other_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prisma, _, _ = _install(monkeypatch)

    paid: Final = await tokenin_update_key(
        data=TokeninUpdateRequest(key="sk-payg", plan_id="payg", action="payg_topup", topup_usd=3.5),
    )
    assert paid.credited == 3.5
    assert prisma.user_table.wallets[ACCOUNT_ID]["max_budget"] == 3.5

    with pytest.raises(HTTPException):
        await tokenin_update_key(
            data=TokeninUpdateRequest(key="sk-key", plan_id="medium", action="payg_topup", topup_usd=3.5),
        )
    with pytest.raises(HTTPException):
        await tokenin_update_key(
            data=TokeninUpdateRequest(key="sk-key", plan_id="medium", action="payg_topup"),
        )
    with pytest.raises(HTTPException):
        await tokenin_update_key(
            data=TokeninUpdateRequest(key="sk-payg", plan_id="payg", action="extend"),
        )
    assert prisma.user_table.wallets[ACCOUNT_ID]["max_budget"] == 3.5


async def test_an_unlinked_key_cannot_move_credit(monkeypatch: pytest.MonkeyPatch) -> None:
    prisma, _, _ = _install(monkeypatch, key_user_id=None)

    with pytest.raises(HTTPException):
        await tokenin_update_key(
            data=TokeninUpdateRequest(key="sk-orphan", plan_id="medium", action="extend"),
        )
    assert prisma.user_table.increments == []


async def test_team_key_is_constrained_by_the_wallet_only_when_enabled() -> None:
    cache: Final = FakeUserApiKeyCache()
    team_key: Final = UserAPIKeyAuth(
        api_key="sk-team-key",
        user_id=ACCOUNT_ID,
        team_id=FLAT_TEAM_ID,
    )

    async def counters_when(enabled: bool) -> list[_BudgetCounter]:
        return await _get_budget_counters(
            request_body={},
            valid_token=team_key,
            team_object=LiteLLM_TeamTable(team_id=FLAT_TEAM_ID),
            user_object=LiteLLM_UserTable(user_id=ACCOUNT_ID, max_budget=10.0, spend=2.5),
            prisma_client=None,
            user_api_key_cache=cache,  # pyright: ignore[reportArgumentType]  # test double, checked by behavior below
            proxy_logging_obj=None,  # pyright: ignore[reportArgumentType]  # unused for the counters under test
            apply_user_budget_to_team_keys=enabled,
        )

    enabled_counters: Final = [
        counter for counter in await counters_when(True) if counter.entity_type == "User"
    ]
    disabled_counters: Final = [
        counter for counter in await counters_when(False) if counter.entity_type == "User"
    ]

    assert [counter.counter_key for counter in enabled_counters] == [f"spend:user:{ACCOUNT_ID}"]
    assert enabled_counters[0].max_budget == 10.0
    assert disabled_counters == []
