from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy.tokenin.ledger import (
    GrantBalance,
    allocate_fifo,
    cancel_account_request,
    eligible_grants,
    mark_account_request_uncertain,
    settle_account_request,
    to_nano_ceil,
)
from litellm.proxy.tokenin.ledger import (
    reserve_account_request as reserve_with_key,
)

ACCOUNT: Final = "paid-account"
NANO: Final = 1_000_000_000
JAN31: Final = datetime(2026, 1, 31)
FEB28: Final = datetime(2026, 2, 28)


def _stored(value: object) -> datetime | None:
    """Postgres casts a text bind into its timestamp column; mirror that for the fake store."""
    if value is None:
        return None
    moment: Final = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(moment, datetime):
        raise AssertionError(f"not a timestamp bind: {value!r}")
    return moment.astimezone(timezone.utc).replace(tzinfo=None) if moment.tzinfo is not None else moment


MAR31: Final = datetime(2026, 3, 31)
APR30: Final = datetime(2026, 4, 30)


def grant(
    name: str,
    amount: int,
    start: datetime | None = None,
    end: datetime | None = None,
    index: int | None = None,
    kind: str = "fixed",
) -> GrantBalance:
    return GrantBalance(
        grant_id=name,
        kind=kind,
        plan_id="medium" if kind == "fixed" else "payg",
        amount_nano=amount,
        allocated_nano=0,
        subscription_id="subscription" if kind == "fixed" else None,
        period_index=index,
        period_start=start,
        period_end=end,
    )


async def reserve_account_request(
    db: LedgerDB, user_id: str, request_id: str, model: str, estimated_cost: float, key_hash: str | None = None
) -> tuple[tuple[str, int], ...]:
    return await reserve_with_key(db, user_id, request_id, model, estimated_cost, key_hash=key_hash or request_id)


def test_one_extra_paid_period_fifo_and_payg_independence() -> None:
    first: Final = grant("g0", 5, JAN31, FEB28, 0)
    second: Final = grant("g1", 5, FEB28, MAR31, 1)
    third: Final = grant("g2", 5, MAR31, APR30, 2)
    payg: Final = grant("payg", 10, kind="payg")
    before: Final = eligible_grants((first, second, third, payg), FEB28 - timedelta(microseconds=1))
    assert tuple(g.grant_id for g in before) == ("g0", "payg")
    renewal: Final = eligible_grants((first, second, third, payg), FEB28)
    assert allocate_fifo(renewal, 14) == (("g0", 5), ("g1", 5), ("payg", 4))
    assert tuple(g.grant_id for g in eligible_grants((first, second, third, payg), MAR31)) == ("g1", "g2", "payg")
    assert tuple(g.grant_id for g in eligible_grants((first, second, payg), MAR31)) == ("payg",)
    assert tuple(g.grant_id for g in eligible_grants((first, payg), FEB28)) == ("payg",)
    assert tuple(g.grant_id for g in eligible_grants((first, payg), APR30)) == ("payg",)


def test_round_up_cost_and_reject_unknown_unpriced_or_overflow() -> None:
    assert to_nano_ceil(0.0000000001) == 1
    assert to_nano_ceil(0.0000000011) == 2
    assert to_nano_ceil(0, allow_zero=True) == 0
    for cost in (float("nan"), float("inf"), -1, 0, 1e100):
        with pytest.raises(HTTPException) as denied:
            to_nano_ceil(cost)
        assert denied.value.status_code == 503


class LedgerDB:
    def __init__(self, now: datetime = FEB28) -> None:
        self.now = now
        self.lock = asyncio.Lock()
        self.debt = 0
        self.grants: dict[str, GrantBalance] = {}
        self.holds: dict[str, dict[str, object]] = {}
        self.allocations: dict[tuple[str, str], int] = {}
        self.policies: list[dict[str, object]] = [
            {
                "plan_id": "medium",
                "models": ["allowed"],
                "rpm_limit": 10,
                "max_parallel_requests": 2,
                "effective_at": FEB28,
            },
            {
                "plan_id": "payg",
                "models": ["allowed"],
                "rpm_limit": 10,
                "max_parallel_requests": 2,
                "effective_at": JAN31,
            },
        ]
        self.db = self
        self.snapshot: tuple[int, dict[str, dict[str, object]], dict[tuple[str, str], int]] | None = None

    def tx(self) -> LedgerDB:
        return self

    async def __aenter__(self) -> LedgerDB:
        await self.lock.acquire()
        self.snapshot = (self.debt, deepcopy(self.holds), dict(self.allocations))
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is not None and self.snapshot is not None:
            self.debt, self.holds, self.allocations = self.snapshot
        self.lock.release()

    async def query_raw(self, sql: str, *values: object) -> list[dict[str, object]]:
        if '"LiteLLM_TokeninAccount"' in sql:
            return [{"user_id": ACCOUNT, "debt_nano": self.debt}] if values[0] == ACCOUNT else []
        if "clock_timestamp()" in sql:
            return [{"now": self.now}]
        if '"LiteLLM_TokeninHold" h' in sql:
            return [
                {
                    "request_id": hold["request_id"],
                    "state": hold["state"],
                    "model": hold["model"],
                    "estimated_nano": hold["estimated_nano"],
                    "charged_nano": hold["charged_nano"],
                    "admitted_at": hold["admitted_at"],
                    "settled_at": None,
                    "allocated_nano": sum(
                        amount
                        for (request_id, _), amount in self.allocations.items()
                        if request_id == hold["request_id"]
                    ),
                }
                for hold in self.holds.values()
                if hold["user_id"] == values[0]
            ]
        if '"LiteLLM_TokeninHold" WHERE "request_id"' in sql:
            hold = self.holds.get(str(values[0]))
            return [hold] if hold is not None else []
        if '"LiteLLM_TokeninGrant" g LEFT JOIN' in sql:
            return [
                {
                    "idempotency_key": grant_obj.grant_id,
                    "kind": grant_obj.kind,
                    "plan_id": grant_obj.plan_id,
                    "amount_nano": grant_obj.amount_nano,
                    "subscription_id": grant_obj.subscription_id,
                    "period_index": grant_obj.period_index,
                    "period_start": grant_obj.period_start,
                    "period_end": grant_obj.period_end,
                    "allocated_nano": sum(
                        amount for (_, grant_id), amount in self.allocations.items() if grant_id == grant_obj.grant_id
                    ),
                }
                for grant_obj in self.grants.values()
            ]
        if '"LiteLLM_TokeninPolicy"' in sql:
            return [policy for policy in self.policies if policy["effective_at"] <= _stored(values[1])]
        if "COUNT(*) FILTER" in sql:
            return [
                {
                    "rpm": sum(hold["admitted_at"] > self.now - timedelta(seconds=60) for hold in self.holds.values()),
                    "concurrent": sum(hold["state"] in ("held", "uncertain") for hold in self.holds.values()),
                }
            ]
        if '"LiteLLM_TokeninAllocation"' in sql and "JOIN" in sql:
            rows = [
                {"grant_id": grant_id, "amount_nano": amount}
                for (request_id, grant_id), amount in self.allocations.items()
                if request_id == values[0]
            ]
            return sorted(
                rows, key=lambda row: (self.grants[row["grant_id"]].period_start or datetime.max, row["grant_id"])
            )
        if '"LiteLLM_TokeninAllocation"' in sql:
            return [
                {"grant_id": grant_id, "amount_nano": amount}
                for (request_id, grant_id), amount in self.allocations.items()
                if request_id == values[0]
            ]
        raise AssertionError(sql)

    async def execute_raw(self, sql: str, *values: object) -> int:
        if 'INSERT INTO "LiteLLM_TokeninHold"' in sql:
            if len(values) == 5:
                request_id, user_id, amount, timestamp, _ = values
                key_hash, model, state, charged = "tokenin-debt", "tokenin-debt", "settled", amount
            else:
                request_id, user_id, key_hash, model, amount, timestamp = values
                state, charged = "held", None
            self.holds[str(request_id)] = {
                "request_id": request_id,
                "user_id": user_id,
                "key_hash": key_hash,
                "model": model,
                "state": state,
                "estimated_nano": amount,
                "charged_nano": charged,
                "admitted_at": _stored(timestamp),
            }
        elif 'UPDATE "LiteLLM_TokeninAllocation"' in sql and len(values) == 1:
            self.allocations = {key: amount for key, amount in self.allocations.items() if key[0] != str(values[0])}
        elif 'INSERT INTO "LiteLLM_TokeninAllocation"' in sql or 'UPDATE "LiteLLM_TokeninAllocation"' in sql:
            request_id, grant_id, amount = values
            key = (str(request_id), str(grant_id))
            self.allocations[key] = self.allocations.get(key, 0) + int(amount) if "ON CONFLICT" in sql else int(amount)
        elif 'UPDATE "LiteLLM_TokeninAccount"' in sql:
            self.debt = 0 if '"debt_nano" = 0' in sql else self.debt + int(values[1])
        elif 'UPDATE "LiteLLM_TokeninHold"' in sql:
            hold = self.holds[str(values[0])]
            if "'cancelled'" in sql:
                hold["state"], hold["charged_nano"] = "cancelled", 0
            elif "'uncertain'" in sql:
                hold["state"] = "uncertain"
            else:
                hold["state"], hold["charged_nano"] = "settled", values[1]
        else:
            raise AssertionError(sql)
        return 1


@pytest.mark.asyncio
async def test_shared_two_key_reservations_atomic_and_refund_after_settlement() -> None:
    db: Final = LedgerDB()
    db.grants = {"old": grant("old", 5 * NANO, JAN31, FEB28, 0), "current": grant("current", 5 * NANO, FEB28, MAR31, 1)}
    first, second = await asyncio.gather(
        reserve_account_request(db, ACCOUNT, "key-1", "allowed", 4.0),
        reserve_account_request(db, ACCOUNT, "key-2", "allowed", 4.0),
    )
    assert first == (("old", 4 * NANO),)
    assert second == (("old", NANO), ("current", 3 * NANO))
    with pytest.raises(HTTPException) as third:
        await reserve_account_request(db, ACCOUNT, "key-3", "allowed", 1.0)
    assert third.value.status_code == 429
    assert await settle_account_request(db, "key-1", 2.0) == 2 * NANO
    assert await settle_account_request(db, "key-2", 4.0) == 4 * NANO
    assert await reserve_account_request(db, ACCOUNT, "key-3", "allowed", 1.0) == (("old", NANO),)
    assert db.allocations[("key-1", "old")] == 2 * NANO
    assert db.allocations[("key-2", "old")] == NANO


@pytest.mark.asyncio
async def test_policy_model_and_rpm_reject_across_keys_and_future_policy_is_ignored() -> None:
    db: Final = LedgerDB()
    db.grants = {"paid": grant("paid", 5 * NANO, FEB28, MAR31, 1)}
    db.policies[0]["rpm_limit"] = 1
    db.policies.append(
        {
            "plan_id": "medium",
            "models": ["future-only"],
            "rpm_limit": 100,
            "max_parallel_requests": 100,
            "effective_at": MAR31,
        }
    )
    with pytest.raises(HTTPException) as model:
        await reserve_account_request(db, ACCOUNT, "wrong-model", "future-only", 1.0)
    assert model.value.status_code == 403
    await reserve_account_request(db, ACCOUNT, "first-key", "allowed", 1.0)
    await settle_account_request(db, "first-key", 1.0)
    with pytest.raises(HTTPException) as rpm:
        await reserve_account_request(db, ACCOUNT, "second-key", "allowed", 1.0)
    assert rpm.value.status_code == 429


@pytest.mark.asyncio
async def test_expired_fixed_stops_at_boundary_without_renewal_but_payg_spends() -> None:
    db: Final = LedgerDB(now=MAR31)
    db.grants = {"paid": grant("paid", 5 * NANO, FEB28, MAR31, 1), "prepaid": grant("prepaid", 4 * NANO, kind="payg")}
    assert await reserve_account_request(db, ACCOUNT, "prepaid-key", "allowed", 3.0) == (("prepaid", 3 * NANO),)
    db.policies = [policy for policy in db.policies if policy["plan_id"] == "medium"]
    with pytest.raises(HTTPException) as missing_payg_policy:
        await reserve_account_request(db, ACCOUNT, "missing-policy", "allowed", 1.0)
    assert missing_payg_policy.value.status_code == 503


@pytest.mark.asyncio
async def test_operator_reconciliation_lists_and_resolves_stuck_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.tokenin import accounts

    db: Final = LedgerDB()
    db.grants = {"paid": grant("paid", 2 * NANO, FEB28, MAR31, 1)}
    monkeypatch.setattr(accounts, "_db", lambda: SimpleNamespace(db=db))
    monkeypatch.setenv("TOKENIN_ACCOUNT_V2_ENABLED", "true")
    monkeypatch.setenv("TOKENIN_ACCOUNT_SPEND_ENABLED", "true")
    await reserve_account_request(db, ACCOUNT, "stuck", "allowed", 2.0)
    listed: Final = await accounts.account_holds(user_id=ACCOUNT, states="held,uncertain", older_than_minutes=0)
    assert [row["request_id"] for row in listed["holds"]] == ["stuck"]
    assert listed["holds"][0]["state"] == "held"
    assert listed["holds"][0]["reserved_usd"] == 2.0
    assert listed["reserved_usd"] == 2.0
    assert listed["enforcement_active"] is True
    assert (await accounts.account_holds(user_id=ACCOUNT, states="settled", older_than_minutes=0))["holds"] == []
    with pytest.raises(HTTPException) as bad_state:
        await accounts.account_holds(user_id=ACCOUNT, states="held,nonsense", older_than_minutes=0)
    assert bad_state.value.status_code == 400

    cancelled: Final = await accounts.resolve_account_hold(
        request_id="stuck", data=accounts.HoldResolution(action="cancel", note="provider never dispatched")
    )
    assert cancelled == {
        "request_id": "stuck",
        "state": "cancelled",
        "charged_usd": 0.0,
        "note": "provider never dispatched",
    }
    assert await reserve_account_request(db, ACCOUNT, "after", "allowed", 2.0) == (("paid", 2 * NANO),)
    repeated: Final = await accounts.resolve_account_hold(
        request_id="stuck", data=accounts.HoldResolution(action="cancel", note="retry")
    )
    assert repeated["state"] == "cancelled"
    with pytest.raises(HTTPException) as missing_cost:
        await accounts.resolve_account_hold(
            request_id="after", data=accounts.HoldResolution(action="settle", note="no cost given")
        )
    assert missing_cost.value.status_code == 400
    settled: Final = await accounts.resolve_account_hold(
        request_id="after",
        data=accounts.HoldResolution(action="settle", cost_usd=Decimal("0.5"), note="provider invoice"),
    )
    assert settled == {"request_id": "after", "state": "settled", "charged_usd": 0.5, "note": "provider invoice"}
    assert db.allocations[("after", "paid")] == 500_000_000
    with pytest.raises(HTTPException) as finalized:
        await accounts.resolve_account_hold(
            request_id="after", data=accounts.HoldResolution(action="cancel", note="too late to refund")
        )
    assert finalized.value.status_code == 409
    assert (await accounts.account_holds(user_id=ACCOUNT, states="uncertain", older_than_minutes=0))["holds"] == []


@pytest.mark.asyncio
async def test_admission_replay_is_bound_to_account_key_model_and_estimate() -> None:
    db: Final = LedgerDB()
    db.grants = {"paid": grant("paid", 5 * NANO, FEB28, MAR31, 1)}
    first: Final = await reserve_account_request(db, ACCOUNT, "request-id", "allowed", 1.0, key_hash="key-a")
    assert await reserve_account_request(db, ACCOUNT, "request-id", "allowed", 1.0, key_hash="key-a") == first
    for key_hash, model, estimate in (("key-b", "allowed", 1.0), ("key-a", "other", 1.0), ("key-a", "allowed", 2.0)):
        with pytest.raises(HTTPException) as mismatch:
            await reserve_account_request(db, ACCOUNT, "request-id", model, estimate, key_hash=key_hash)
        assert mismatch.value.status_code == 409
    assert len(db.allocations) == 1


@pytest.mark.asyncio
async def test_new_paid_period_requires_initial_policy_and_admin_revision_takes_effect() -> None:
    db: Final = LedgerDB()
    db.grants = {"paid": grant("paid", 5 * NANO, FEB28, MAR31, 1)}
    db.policies[0]["effective_at"] = JAN31
    with pytest.raises(HTTPException) as missing_initial:
        await reserve_account_request(db, ACCOUNT, "old-policy", "allowed", 1.0)
    assert missing_initial.value.status_code == 503
    db.policies[0]["effective_at"] = FEB28
    db.now = FEB28 + timedelta(hours=2)
    db.policies.append(
        {
            "plan_id": "medium",
            "models": ["other"],
            "rpm_limit": 10,
            "max_parallel_requests": 2,
            "effective_at": FEB28 + timedelta(hours=1),
        }
    )
    with pytest.raises(HTTPException) as changed:
        await reserve_account_request(db, ACCOUNT, "old-model", "allowed", 1.0)
    assert changed.value.status_code == 403
    assert await reserve_account_request(db, ACCOUNT, "new-model", "other", 1.0) == (("paid", NANO),)


@pytest.mark.asyncio
async def test_overrun_uses_other_available_credit_before_recording_debt() -> None:
    db: Final = LedgerDB()
    db.grants = {"paid": grant("paid", 5 * NANO, FEB28, MAR31, 1)}
    await reserve_account_request(db, ACCOUNT, "overrun", "allowed", 1.0)
    assert await settle_account_request(db, "overrun", 1.5) == 1_500_000_000
    assert db.allocations[("overrun", "paid")] == 1_500_000_000
    assert db.debt == 0
    assert await reserve_account_request(db, ACCOUNT, "more", "allowed", 3.5) == (("paid", 3_500_000_000),)


@pytest.mark.asyncio
async def test_overrun_debt_repaid_by_new_payg_without_consuming_rpm_slot() -> None:
    db: Final = LedgerDB()
    db.grants = {"paid": grant("paid", NANO, FEB28, MAR31, 1)}
    db.policies[0]["rpm_limit"] = 2
    await reserve_account_request(db, ACCOUNT, "first", "allowed", 1.0)
    await settle_account_request(db, "first", 1.5)
    assert db.debt == 500_000_000
    with pytest.raises(HTTPException) as denied:
        await reserve_account_request(db, ACCOUNT, "before-topup", "allowed", 0.25)
    assert denied.value.status_code == 402
    db.grants["topup"] = grant("topup", NANO, kind="payg")
    assert await reserve_account_request(db, ACCOUNT, "after-topup", "allowed", 0.25) == (("topup", 250_000_000),)
    assert db.debt == 0
    debt_holds: Final = [row for row in db.holds.values() if row["model"] == "tokenin-debt"]
    assert len(debt_holds) == 1
    assert debt_holds[0]["admitted_at"] == datetime(1970, 1, 1)
    assert sum(amount for (_, grant_id), amount in db.allocations.items() if grant_id == "topup") == 750_000_000


@pytest.mark.asyncio
async def test_cancel_releases_credit_but_never_refunds_settled_holds() -> None:
    db: Final = LedgerDB()
    db.grants = {"paid": grant("paid", 4 * NANO, FEB28, MAR31, 1)}
    await reserve_account_request(db, ACCOUNT, "cancelled", "allowed", 3.0)
    assert await cancel_account_request(db, "cancelled") is True
    assert await cancel_account_request(db, "cancelled") is True
    assert db.holds["cancelled"]["state"] == "cancelled"
    assert await reserve_account_request(db, ACCOUNT, "reuse", "allowed", 4.0) == (("paid", 4 * NANO),)
    assert await cancel_account_request(db, "unknown-request") is False
    assert await settle_account_request(db, "reuse", 4.0) == 4 * NANO
    assert await cancel_account_request(db, "reuse") is False
    assert db.allocations[("reuse", "paid")] == 4 * NANO


@pytest.mark.asyncio
async def test_uncertain_hold_keeps_reservation_and_never_releases_speculatively() -> None:
    db: Final = LedgerDB()
    db.grants = {"paid": grant("paid", 2 * NANO, FEB28, MAR31, 1)}
    await reserve_account_request(db, ACCOUNT, "lost", "allowed", 2.0)
    assert await mark_account_request_uncertain(db, "lost") is True
    assert await mark_account_request_uncertain(db, "lost") is True
    assert await cancel_account_request(db, "lost") is False
    with pytest.raises(HTTPException) as blocked:
        await settle_account_request(db, "lost", 1.0)
    assert blocked.value.status_code == 503
    with pytest.raises(HTTPException) as exhausted:
        await reserve_account_request(db, ACCOUNT, "next", "allowed", 0.5)
    assert exhausted.value.status_code == 402
    assert db.allocations[("lost", "paid")] == 2 * NANO


@pytest.mark.asyncio
async def test_settlement_replay_cost_conflict_and_unexpected_overrun_debt() -> None:
    db: Final = LedgerDB()
    db.grants = {"paid": grant("paid", NANO, FEB28, MAR31, 1)}
    await reserve_account_request(db, ACCOUNT, "request-id", "allowed", 1.0)
    assert await settle_account_request(db, "request-id", 1.5) == 1_500_000_000
    assert db.debt == 500_000_000
    assert await settle_account_request(db, "request-id", 1.5) == 1_500_000_000
    with pytest.raises(HTTPException) as changed:
        await settle_account_request(db, "request-id", 1.4)
    assert changed.value.status_code == 409
    with pytest.raises(HTTPException) as blocked:
        await reserve_account_request(db, ACCOUNT, "next", "allowed", 0.01)
    assert blocked.value.status_code == 402
