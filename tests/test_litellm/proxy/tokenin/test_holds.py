from __future__ import annotations

from datetime import datetime, timedelta
from importlib import reload
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy import litellm_pre_call_utils
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.tokenin import holds as reaper
from litellm.proxy.tokenin.ledger import reserve_account_request
from tests.test_litellm.proxy.tokenin.test_ledger import ACCOUNT, FEB28, LedgerDB, grant

NANO: Final = 1_000_000_000
NOW: Final = datetime(2026, 9, 30, 20, 0, 0)


def _reserve_db(max_parallel_requests: int = 2, payg_max_parallel_requests: int = 5) -> LedgerDB:
    db: Final = LedgerDB(now=NOW)
    db.grants = {"paid": grant("paid", 5 * NANO, NOW - timedelta(days=1), NOW + timedelta(days=29), 1)}
    db.policies = [
        {
            "plan_id": "medium",
            "models": ["allowed"],
            "rpm_limit": 60,
            "max_parallel_requests": max_parallel_requests,
            "effective_at": db.grants["paid"].period_start,
        },
        {
            "plan_id": "payg",
            "models": ["allowed"],
            "rpm_limit": 120,
            "max_parallel_requests": payg_max_parallel_requests,
            "effective_at": FEB28 - timedelta(days=1),
        },
    ]
    return db


@pytest.mark.asyncio
async def test_stuck_uncertain_still_no_concurrency_with_ttl() -> None:
    db: Final = _reserve_db(max_parallel_requests=1, payg_max_parallel_requests=1)
    await reserve_account_request(db, ACCOUNT, "stale", "allowed", 1.0, key_hash="k1")
    db.holds["stale"]["state"] = "uncertain"
    await reserve_account_request(db, ACCOUNT, "new", "allowed", 1.0, key_hash="k2")
    assert db.holds["stale"]["charged_nano"] is None


@pytest.mark.asyncio
async def test_stale_held_not_counted_and_fresh_held_counted() -> None:
    db: Final = _reserve_db(max_parallel_requests=1, payg_max_parallel_requests=2)
    await reserve_account_request(db, ACCOUNT, "stale-held", "allowed", 1.0, key_hash="k1")
    db.holds["stale-held"]["admitted_at"] = NOW - timedelta(seconds=901)
    await reserve_account_request(db, ACCOUNT, "fresh", "allowed", 1.0, key_hash="k2")
    await reserve_account_request(db, ACCOUNT, "fresh-2", "allowed", 1.0, key_hash="k3")
    with pytest.raises(HTTPException) as exceeded:
        await reserve_account_request(db, ACCOUNT, "third", "allowed", 1.0, key_hash="k4")
    assert exceeded.value.status_code == 429


@pytest.mark.asyncio
async def test_paid_plan_gets_payg_higher_limit() -> None:
    db: Final = _reserve_db(max_parallel_requests=2, payg_max_parallel_requests=5)
    for rid in ("a", "b", "c", "d"):
        await reserve_account_request(db, ACCOUNT, rid, "allowed", 0.2, key_hash=f"k{rid}")
    await reserve_account_request(db, ACCOUNT, "e", "allowed", 0.2, key_hash="ke")
    assert len(db.holds) == 5


@pytest.mark.asyncio
async def test_reaper_chooses_and_settles_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    spends: Final = AsyncMock(side_effect=[0.35, None, 0.01])
    candidates: Final = [
            SimpleNamespace(
                request_id="stale-held", state="held", estimated_nano=1 * NANO,
                admitted_at=NOW - timedelta(minutes=30),
            ),
            SimpleNamespace(
                request_id="uncertain-spend", state="uncertain", estimated_nano=2 * NANO,
                admitted_at=NOW - timedelta(hours=25),
            ),
            SimpleNamespace(
                request_id="uncertain-no-spend", state="uncertain", estimated_nano=3 * NANO,
                admitted_at=NOW - timedelta(hours=26),
            ),
            SimpleNamespace(
                request_id="uncertain-fresh-log", state="uncertain", estimated_nano=4 * NANO,
                admitted_at=NOW - timedelta(minutes=5),
            ),
    ]

    async def choose_rows(db: object, now: datetime, offset: int = 0) -> list[SimpleNamespace]:
        del db
        assert now == NOW
        assert offset == 0
        return [candidates.pop(0)] if candidates else []

    uncertain_tx: Final = AsyncMock(return_value=True)
    settle_tx: Final = AsyncMock(return_value=1)
    prisma: Final = SimpleNamespace(db=SimpleNamespace(tx=lambda: _NoTx()))
    monkeypatch.setattr(reaper, "_spend_cost", spends)
    monkeypatch.setattr(reaper, "_lock_contenders", choose_rows)
    monkeypatch.setattr(reaper, "mark_account_request_uncertain_tx", uncertain_tx)
    monkeypatch.setattr(reaper, "settle_account_request_tx", settle_tx)
    warnings: Final = MagicMock()
    monkeypatch.setattr(reaper.verbose_proxy_logger, "warning", warnings)
    summary: Final = await reaper.reconcile_account_holds(prisma, now_override=NOW)
    assert summary == {"promoted": 1, "settled": 3, "unreconciled": 0}
    uncertain_tx.assert_awaited_once()
    assert uncertain_tx.call_args.kwargs == {} or uncertain_tx.call_args.kwargs.get("request_id") == "stale-held"
    assert settle_tx.call_args_list[0].args[2] == 0.35
    assert settle_tx.call_args_list[1].args[2] == 3.0
    assert settle_tx.call_args_list[2].args[2] == 0.01
    assert any("stale state=" in str(call.args[0]) for call in warnings.call_args_list)


class _NoTx:
    async def __aenter__(self) -> _NoTx:
        return self

    async def __aexit__(self, *exception: object) -> None:
        pass

    async def query_raw(self, sql: str, *values: object) -> list[dict[str, object]]:
        assert "pg_try_advisory_xact_lock" in sql
        return [{"locked": True}]


@pytest.mark.asyncio
async def test_paid_plan_without_payg_policy_keeps_its_own_limit() -> None:
    db: Final = _reserve_db(max_parallel_requests=2, payg_max_parallel_requests=5)
    db.policies = [db.policies[0]]
    await reserve_account_request(db, ACCOUNT, "one", "allowed", 0.2, key_hash="k1")
    await reserve_account_request(db, ACCOUNT, "two", "allowed", 0.2, key_hash="k2")
    with pytest.raises(HTTPException) as exceeded:
        await reserve_account_request(db, ACCOUNT, "three", "allowed", 0.2, key_hash="k3")
    assert exceeded.value.status_code == 429


@pytest.mark.asyncio
async def test_repeated_concurrency_refusals_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.tokenin import ledger

    db: Final = _reserve_db(max_parallel_requests=1, payg_max_parallel_requests=1)
    ledger._DENIAL_CACHE.pop(ACCOUNT, None)  # pyright: ignore[reportPrivateUsage]
    await reserve_account_request(db, ACCOUNT, "running", "allowed", 0.2, key_hash="k0")
    warning: Final = MagicMock()
    monkeypatch.setattr(ledger.verbose_proxy_logger, "warning", warning)
    for index in range(3):
        with pytest.raises(HTTPException):
            await reserve_account_request(db, ACCOUNT, f"denied-{index}", "allowed", 0.2, key_hash=f"k{index}")
    assert warning.call_count >= 1
    final: Final = warning.call_args_list[-1]
    assert final.args[1:] == (ACCOUNT, 3)


@pytest.mark.asyncio
async def test_young_uncertain_warns_without_settling(monkeypatch: pytest.MonkeyPatch) -> None:
    pending: Final = SimpleNamespace(
        request_id="watch", state="uncertain", estimated_nano=NANO, admitted_at=NOW - timedelta(minutes=90)
    )
    settle: Final = AsyncMock()
    spend: Final = AsyncMock(return_value=None)

    async def choose_pending(db: object, now: datetime, offset: int = 0) -> list[SimpleNamespace]:
        del db
        assert now == NOW
        return [pending] if offset == 0 else []

    warning: Final = MagicMock()
    monkeypatch.setattr(reaper, "_lock_contenders", choose_pending)
    monkeypatch.setattr(reaper, "_spend_cost", spend)
    monkeypatch.setattr(reaper, "settle_account_request_tx", settle)
    monkeypatch.setattr(reaper.verbose_proxy_logger, "warning", warning)
    summary: Final = await reaper.reconcile_account_holds(
        SimpleNamespace(db=SimpleNamespace(tx=lambda: _NoTx())), now_override=NOW
    )
    assert summary == {"promoted": 0, "settled": 0, "unreconciled": 0}
    settle.assert_not_called()
    spend.assert_awaited_once()
    assert any("stale state=uncertain" in str(call.args[0]) % call.args[1:] for call in warning.call_args_list)


@pytest.mark.asyncio
async def test_reaper_sql_and_locking_patterns() -> None:
    source: Final = open("litellm/proxy/tokenin/holds.py").read()
    assert "pg_try_advisory_xact_lock(hashtextextended" in source
    assert "'{spend_logs_metadata,tokenin_account_hold_id}'" in source
    assert "'{user_api_key_tokenin_account_hold_id}'" in source
    assert "ORDER BY" in source and "LIMIT 1 OFFSET $2" in source
    assert 'OR \\"state\\" = \'uncertain\'' in source
    server: Final = open("litellm/proxy/proxy_server.py").read()
    assert "tokenin_account_hold_reaper_job" in server
    assert "HOLD_REAPER_INTERVAL_SECONDS" in server


@pytest.mark.asyncio
async def test_reaper_failure_is_logged_and_left_for_next_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate: Final = SimpleNamespace(
        request_id="bad", state="uncertain", estimated_nano=NANO, admitted_at=NOW - timedelta(hours=26)
    )
    async def choose_bad(db: object, now: datetime, offset: int = 0) -> list[SimpleNamespace]:
        del db
        assert now == NOW
        return [candidate] if offset == 0 else []

    monkeypatch.setattr(reaper, "_lock_contenders", choose_bad)
    monkeypatch.setattr(reaper, "_spend_cost", AsyncMock(return_value=None))
    monkeypatch.setattr(reaper, "settle_account_request_tx", AsyncMock(side_effect=RuntimeError("wipe")))
    warning: Final = MagicMock()
    monkeypatch.setattr(reaper.verbose_proxy_logger, "exception", warning)
    summary: Final = await reaper.reconcile_account_holds(
        SimpleNamespace(db=SimpleNamespace(tx=lambda: _NoTx())), now_override=NOW
    )
    assert summary["unreconciled"] == 1
    warning.assert_called_once()


def test_spend_metadata_contains_hold_id() -> None:
    auth: Final = UserAPIKeyAuth(api_key="hashed", tokenin_account_hold_id="tok-hold-1")
    data: Final = {"model": "model-a", "messages": [{"role": "user", "content": "hi"}], "metadata": {}}
    litellm_pre_call_utils.LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        data, auth, "metadata"
    )
    nested = data["metadata"]["spend_logs_metadata"]
    assert isinstance(nested, dict)
    assert nested["tokenin_account_hold_id"] == "tok-hold-1"

    from litellm.proxy.spend_tracking.spend_tracking_utils import _get_spend_logs_metadata

    converted: Final = _get_spend_logs_metadata(metadata={"spend_logs_metadata": nested})
    spend_logs: Final = converted["spend_logs_metadata"]
    assert isinstance(spend_logs, dict) and spend_logs["tokenin_account_hold_id"] == "tok-hold-1"


def test_reaper_defaults_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    assert reaper.HOLD_INFLIGHT_TTL_SECONDS == 900
    assert reaper.UNCERTAIN_SETTLE_AFTER_SECONDS == 86400
    assert reaper.HOLD_REAPER_INTERVAL_SECONDS == 300
    monkeypatch.setenv("TOKENIN_ACCOUNT_HOLD_INFLIGHT_TTL_SECONDS", "600")
    from litellm.proxy.tokenin import ledger

    reload(ledger)
    reload(reaper)
    assert reaper.HOLD_INFLIGHT_TTL_SECONDS == 600
    monkeypatch.undo()
    reload(ledger)
    reload(reaper)
