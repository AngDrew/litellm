from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import _authorize_authenticated_request
from litellm.proxy.tokenin import accounts
from litellm.proxy.tokenin.plans import FLAT_TEAM_ID, KEY_MODELS, TokeninPlan

ACCOUNT: Final = "customer-123"
PERIOD_START: Final = datetime(2026, 1, 31, tzinfo=timezone.utc)
PERIOD_END: Final = datetime(2026, 2, 28, tzinfo=timezone.utc)


def _stored(value: object) -> datetime | None:
    """Postgres casts a text bind into its timestamp column; mirror that for the fake store."""
    if value is None:
        return None
    moment: Final = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(moment, datetime):
        raise AssertionError(f"not a timestamp bind: {value!r}")
    return moment.astimezone(timezone.utc).replace(tzinfo=None) if moment.tzinfo is not None else moment


class FakeTx:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.accounts: set[str] = set()
        self.grants: dict[str, dict[str, object]] = {}
        self.policies: dict[str, dict[str, object]] = {}
        self.allocations: dict[tuple[str, str], int] = {}
        self.debt: dict[str, int] = {}
        self.now: datetime = datetime(2026, 2, 1)

    def tx(self) -> FakeTx:
        return self

    async def __aenter__(self) -> FakeTx:
        await self.lock.acquire()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.lock.release()

    async def execute_raw(self, sql: str, *values: object) -> int:
        if 'INSERT INTO "LiteLLM_TokeninAccount"' in sql:
            self.accounts.add(str(values[0]))
            return 1
        if 'INSERT INTO "LiteLLM_TokeninGrant"' in sql:
            key, user_id, payload_hash, plan_id, kind, amount, sub, index, start, end = values
            if str(key) in self.grants or any(
                row["user_id"] == user_id and row["subscription_id"] == sub and row["period_index"] == index
                for row in self.grants.values()
                if sub is not None
            ):
                return 0
            self.grants[str(key)] = {
                "idempotency_key": key,
                "user_id": user_id,
                "payload_hash": payload_hash,
                "plan_id": plan_id,
                "kind": kind,
                "amount_nano": amount,
                "subscription_id": sub,
                "period_index": index,
                "period_start": _stored(start),
                "period_end": _stored(end),
            }
            return 1
        if 'INSERT INTO "LiteLLM_TokeninPolicy"' in sql:
            key, user_id, payload_hash, plan_id, models, rpm, concurrent, effective = values
            self.policies[str(key)] = {
                "idempotency_key": key,
                "user_id": user_id,
                "payload_hash": payload_hash,
                "plan_id": plan_id,
                "models": models,
                "rpm_limit": rpm,
                "max_parallel_requests": concurrent,
                "effective_at": _stored(effective),
                "created_at": datetime(2026, 1, 1) + timedelta(seconds=len(self.policies)),
            }
            return 1
        raise AssertionError(sql)

    async def query_raw(self, sql: str, *values: object) -> list[dict[str, object]]:
        if '"LiteLLM_TokeninAccount"' in sql:
            return (
                [{"user_id": values[0], "debt_nano": self.debt.get(str(values[0]), 0)}]
                if values[0] in self.accounts
                else []
            )
        if "clock_timestamp()" in sql:
            return [{"now": self.now.isoformat() + "+00:00"}]
        if '"LiteLLM_TokeninGrant" g LEFT JOIN "LiteLLM_TokeninAllocation"' in sql:
            return [
                {
                    **row,
                    "allocated_nano": sum(
                        amount for (_, grant_id), amount in self.allocations.items() if grant_id == key
                    ),
                }
                for key, row in self.grants.items()
                if row["user_id"] == values[0]
            ]
        if '"LiteLLM_TokeninGrant"' in sql:
            if '"period_index" IN' in sql:
                user_id, sub, kind, before, after = values
                return [
                    row
                    for row in self.grants.values()
                    if row["user_id"] == user_id
                    and row["subscription_id"] == sub
                    and row["kind"] == kind
                    and row["period_index"] in (before, after)
                ]
            if '"idempotency_key" = $1 AND "user_id" = $2' in sql:
                row = self.grants.get(str(values[0]))
                return [row] if row is not None and row["user_id"] == values[1] else []
            if '"idempotency_key" = $1' in sql:
                row = self.grants.get(str(values[0]))
                return [row] if row is not None else []
            return [row for row in self.grants.values() if row["user_id"] == values[0]]
        if '"LiteLLM_TokeninPolicy"' in sql:
            if '"idempotency_key" = $1' in sql:
                row = self.policies.get(str(values[0]))
                return [row] if row is not None else []
            newest_first: Final = sorted(
                (row for row in self.policies.values() if row["user_id"] == values[0]),
                key=lambda row: (row["created_at"], row["idempotency_key"]),
                reverse=True,
            )
            selected: Final = newest_first[:1] if "LIMIT 1" in sql else newest_first
            projection: Final = sql.split("FROM", 1)[0]
            if "*" in projection:
                return selected
            return [{key: value for key, value in row.items() if f'"{key}"' in projection} for row in selected]
        raise AssertionError(sql)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeTx:
    fake: Final = FakeTx()
    monkeypatch.setenv("TOKENIN_ACCOUNT_V2_ENABLED", "true")
    monkeypatch.setenv("TOKENIN_ACCOUNT_SERVICE_TOKEN", "very-long-test-secret-32-characters-minimum")
    monkeypatch.setattr(accounts, "_db", lambda: SimpleNamespace(db=fake))
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", SimpleNamespace(get_model_names=lambda: ["model-a", "model-b"]))
    monkeypatch.setattr(
        accounts,
        "load_plans",
        lambda: [
            TokeninPlan(
                id="medium",
                name="Medium",
                kind="fixed",
                rpm_limit=20,
                max_parallel_requests=1,
                max_budget=2.5,
                monthly_value=10.0,
            ),
            TokeninPlan(id="payg", name="PAYG", kind="payg", rpm_limit=20, max_parallel_requests=1, max_budget=0.0),
        ],
    )
    return fake


@pytest.fixture
def managed_router(store: FakeTx, monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm
    from litellm.proxy import proxy_server

    router: Final = litellm.Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake"}}
            for name in ("model-a", "model-b")
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)


def _fixed(
    key: str, index: int = 0, start: datetime = PERIOD_START, end: datetime = PERIOD_END
) -> accounts.GrantRequest:
    return accounts.GrantRequest(
        user_id=ACCOUNT,
        idempotency_key=key,
        plan_id="medium",
        kind="fixed",
        subscription_id="sub-123",
        period_index=index,
        period_start=start,
        period_end=end,
    )


@pytest.mark.asyncio
async def test_fixed_paid_grant_is_monthly_and_atomic_on_retry(store: FakeTx) -> None:
    first, second = await asyncio.gather(accounts.grant_account(_fixed("p-0")), accounts.grant_account(_fixed("p-0")))
    assert first["credited"] == second["credited"] == Decimal("10")
    assert {first["duplicate"], second["duplicate"]} == {False, True}
    assert first["available_usd"] is None
    assert len(store.grants) == 1
    assert (await accounts.get_grant("p-0", ACCOUNT))["credited"] == Decimal("10")
    with pytest.raises(HTTPException) as wrong_user:
        await accounts.get_grant("p-0", "someone-else")
    assert wrong_user.value.status_code == 404


@pytest.mark.asyncio
async def test_legacy_deployed_catalog_uses_monthly_max_budget_when_no_monthly_value(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        accounts,
        "load_plans",
        lambda: [
            TokeninPlan(
                id="medium",
                name="Medium",
                kind="fixed",
                rpm_limit=20,
                max_parallel_requests=1,
                max_budget=10.0,
                budget_duration="30d",
            )
        ],
    )
    assert (await accounts.grant_account(_fixed("live-catalog-0")))["credited"] == Decimal("10")


@pytest.mark.asyncio
async def test_reused_event_conflict_and_duplicate_paid_period_conflict(store: FakeTx) -> None:
    await accounts.grant_account(_fixed("purchase-1:0"))
    with pytest.raises(HTTPException) as changed:
        await accounts.grant_account(_fixed("purchase-1:0", index=1))
    assert changed.value.status_code == 409
    with pytest.raises(HTTPException) as duplicate_period:
        await accounts.grant_account(_fixed("purchase-2:0"))
    assert duplicate_period.value.status_code == 409
    assert len(store.grants) == 1


@pytest.mark.asyncio
async def test_paid_period_neighbors_keep_database_millisecond_precision(store: FakeTx) -> None:
    first_start: Final = PERIOD_START.replace(microsecond=123456)
    first_end: Final = PERIOD_END.replace(microsecond=987654)
    await accounts.grant_account(_fixed("first-ms", start=first_start, end=first_end))
    await accounts.grant_account(
        _fixed(
            "next-ms",
            index=1,
            start=first_end,
            end=datetime(2026, 3, 28, tzinfo=timezone.utc).replace(microsecond=987654),
        )
    )
    assert store.grants["first-ms"]["period_end"] == PERIOD_END.replace(tzinfo=None, microsecond=987000)
    assert len(store.grants) == 2


@pytest.mark.asyncio
async def test_paid_period_neighbors_must_be_contiguous(store: FakeTx) -> None:
    await accounts.grant_account(_fixed("first"))
    with pytest.raises(HTTPException) as gap:
        await accounts.grant_account(
            _fixed(
                "gap",
                index=1,
                start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 1, tzinfo=timezone.utc),
            )
        )
    assert gap.value.status_code == 409
    await accounts.grant_account(
        _fixed("next", index=1, start=PERIOD_END, end=datetime(2026, 3, 28, tzinfo=timezone.utc))
    )
    assert len(store.grants) == 2


@pytest.mark.asyncio
async def test_payg_exact_credit_and_policy_revision_and_summary(store: FakeTx) -> None:
    payg: Final = accounts.GrantRequest(
        user_id=ACCOUNT, idempotency_key="payment-123", plan_id="payg", kind="payg", topup_usd=Decimal("8.25")
    )
    assert (await accounts.grant_account(payg))["credited"] == Decimal("8.25")
    policy: Final = accounts.PolicyRequest(
        user_id=ACCOUNT,
        idempotency_key="policy-1",
        plan_id="payg",
        models=["model-a"],
        rpm_limit=20,
        max_parallel_requests=2,
        effective_at=PERIOD_END,
    )
    assert (await accounts.set_account_policy(policy))["duplicate"] is False
    assert (await accounts.set_account_policy(policy))["duplicate"] is True
    summary: Final = await accounts.account_summary(ACCOUNT)
    assert summary["available_usd"] is None
    assert summary["enforcement_active"] is False
    assert summary["grants"][0]["amount_usd"] == Decimal("8.25")
    assert summary["policies"][0]["models"] == ["model-a"]
    assert summary["policies"][0]["user_id"] == ACCOUNT
    assert summary["policies"][0]["idempotency_key"] == "policy-1"
    assert store.policies["policy-1"]["effective_at"] == PERIOD_END.replace(tzinfo=None)


@pytest.mark.asyncio
async def test_policy_cas_binds_latest_created_revision_and_writes_nothing_when_stale(store: FakeTx) -> None:
    first: Final = accounts.PolicyRequest(
        user_id=ACCOUNT,
        idempotency_key="policy-1",
        plan_id="payg",
        models=["model-a"],
        rpm_limit=10,
        max_parallel_requests=1,
        effective_at=PERIOD_START,
    )
    assert (await accounts.set_account_policy(first))["policy_id"] == "policy-1"
    summary: Final = await accounts.account_summary(ACCOUNT)
    assert summary["latest_policy_id"] == "policy-1"
    assert summary["policies"][0]["policy_id"] == "policy-1"
    assert summary["policies"][0]["created_at"] is not None
    applied: Final = accounts.PolicyRequest(
        user_id=ACCOUNT,
        idempotency_key="policy-2",
        plan_id="payg",
        models=["model-b"],
        rpm_limit=10,
        max_parallel_requests=1,
        effective_at=PERIOD_END,
        expected_policy_id="policy-1",
    )
    assert (await accounts.set_account_policy(applied))["duplicate"] is False
    assert (await accounts.set_account_policy(applied))["duplicate"] is True
    assert (await accounts.account_summary(ACCOUNT))["latest_policy_id"] == "policy-2"
    stale: Final = accounts.PolicyRequest(
        user_id=ACCOUNT,
        idempotency_key="policy-3",
        plan_id="payg",
        models=["model-a"],
        rpm_limit=10,
        max_parallel_requests=1,
        effective_at=PERIOD_END,
        expected_policy_id="policy-1",
    )
    with pytest.raises(HTTPException) as conflict:
        await accounts.set_account_policy(stale)
    assert conflict.value.status_code == 409
    assert conflict.value.detail == {"error": "stale_policy", "current_policy_id": "policy-2"}
    assert "policy-3" not in store.policies
    with pytest.raises(HTTPException) as unknown_alias:
        await accounts.set_account_policy(stale.model_copy(update={"models": ["model-z"], "idempotency_key": "p4"}))
    assert unknown_alias.value.status_code == 400


@pytest.mark.asyncio
async def test_managed_admission_scopes_routes_and_requires_the_spend_switch(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.accounts.add(ACCOUNT)
    client: Final = SimpleNamespace(db=store)
    with pytest.raises(HTTPException) as denied:
        await accounts.require_request_admission(client, ACCOUNT, FLAT_TEAM_ID, "/chat/completions")
    assert denied.value.status_code == 503
    monkeypatch.setenv("TOKENIN_ACCOUNT_SPEND_ENABLED", "true")
    assert await accounts.require_request_admission(client, ACCOUNT, FLAT_TEAM_ID, "/chat/completions") is None
    assert await accounts.require_request_admission(client, ACCOUNT, FLAT_TEAM_ID, "/v1/chat/completions") is None
    # A live global user-budget guard would deny every managed key, so arming spend on
    # top of it is refused rather than charging an account nobody can use.
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"apply_user_budget_to_team_keys": True})
    with pytest.raises(HTTPException) as conflicting:
        await accounts.require_request_admission(client, ACCOUNT, FLAT_TEAM_ID, "/chat/completions")
    assert conflicting.value.status_code == 503
    assert "apply_user_budget_to_team_keys" in conflicting.value.detail
    monkeypatch.setattr(proxy_server, "general_settings", {})
    assert await accounts.require_request_admission(client, ACCOUNT, FLAT_TEAM_ID, "/v1/embeddings", "POST") is None
    for read_route in ("/models", "/v1/models", "/models/model-a", "/v1/models/model-a"):
        assert await accounts.require_request_admission(client, ACCOUNT, FLAT_TEAM_ID, read_route, "GET") is None
    with pytest.raises(HTTPException) as write_read:
        await accounts.require_request_admission(client, ACCOUNT, FLAT_TEAM_ID, "/v1/models", "POST")
    assert write_read.value.status_code == 503
    with pytest.raises(HTTPException) as model_info:
        await accounts.require_request_admission(client, ACCOUNT, FLAT_TEAM_ID, "/model/info", "GET")
    assert model_info.value.status_code == 503
    with pytest.raises(HTTPException) as audio:
        await accounts.require_request_admission(client, ACCOUNT, FLAT_TEAM_ID, "/v1/audio/transcriptions", "POST")
    assert audio.value.status_code == 503
    # A pass-through route reaches a provider with the proxy's own credentials, so it
    # must not be a way around the ledger either.
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints

    monkeypatch.setattr(
        pass_through_endpoints.InitPassThroughEndpointHelpers,
        "is_registered_pass_through_route",
        staticmethod(lambda route: route == "/vertex_ai/custom"),
    )
    with pytest.raises(HTTPException) as pass_through:
        await accounts.require_request_admission(client, ACCOUNT, FLAT_TEAM_ID, "/vertex_ai/custom")
    assert pass_through.value.status_code == 503
    assert await accounts.require_request_admission(client, "not-enrolled", FLAT_TEAM_ID, "/vertex_ai/custom") is None
    assert await accounts.require_request_admission(client, ACCOUNT, FLAT_TEAM_ID, "/user/info") is None


@pytest.mark.usefixtures("managed_router")
@pytest.mark.asyncio
async def test_managed_reservation_binds_key_model_and_worst_case_cost(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.accounts.add(ACCOUNT)
    client: Final = SimpleNamespace(db=store)
    reserve: Final = AsyncMock(return_value=(("paid", 1),))
    estimator: Final = AsyncMock(return_value=2.5)
    monkeypatch.setattr("litellm.proxy.tokenin.ledger.reserve_account_request", reserve)
    monkeypatch.setattr("litellm.proxy.tokenin.enforcement.estimate_account_max_cost", estimator)
    body: Final = {"model": "model-a"}
    assert (
        await accounts.reserve_managed_request(client, ACCOUNT, FLAT_TEAM_ID, "/chat/completions", body, "sk-abc")
        is None
    )
    assert not reserve.await_count
    monkeypatch.setenv("TOKENIN_ACCOUNT_SPEND_ENABLED", "true")
    assert (
        await accounts.reserve_managed_request(client, "not-enrolled", FLAT_TEAM_ID, "/chat/completions", body, "sk")
        is None
    )
    embedded_hold: Final = await accounts.reserve_managed_request(
        client, ACCOUNT, FLAT_TEAM_ID, "/v1/embeddings", body, "sk-abc"
    )
    assert embedded_hold is not None and embedded_hold.startswith("tokenin-")
    hold_id: Final = await accounts.reserve_managed_request(
        client, ACCOUNT, FLAT_TEAM_ID, "/chat/completions", body, "sk-abc"
    )
    assert hold_id is not None and hold_id.startswith("tokenin-")
    assert reserve.await_args is not None
    assert reserve.await_args.kwargs["request_id"] == hold_id
    assert reserve.await_args.kwargs["model"] == "model-a"
    assert reserve.await_args.kwargs["estimated_cost"] == 2.5
    assert reserve.await_args.kwargs["key_hash"] == hashlib.sha256(b"sk-abc").hexdigest()
    estimator.side_effect = HTTPException(status_code=503, detail="Model has no supported maximum cost")
    with pytest.raises(HTTPException) as unpriced:
        await accounts.reserve_managed_request(client, ACCOUNT, FLAT_TEAM_ID, "/chat/completions", body, "sk-abc")
    assert unpriced.value.status_code == 503
    estimator.side_effect = None
    with pytest.raises(HTTPException) as no_model:
        await accounts.reserve_managed_request(
            client, ACCOUNT, FLAT_TEAM_ID, "/chat/completions", {"messages": []}, "sk-abc"
        )
    assert no_model.value.status_code == 503


@pytest.mark.parametrize(
    "fallback",
    [
        {"fallbacks": ["model-b"]},
        {"context_window_fallbacks": [{"model-a": ["model-b"]}]},
        {"content_policy_fallbacks": [{"model-a": ["model-b"]}]},
        {"router_settings_override": {"fallbacks": [{"model-a": ["model-b"]}]}},
    ],
)
@pytest.mark.usefixtures("managed_router")
@pytest.mark.asyncio
async def test_managed_reservation_refuses_client_fallbacks_before_holding_credit(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch, fallback: dict[str, object]
) -> None:
    store.accounts.add(ACCOUNT)
    monkeypatch.setenv("TOKENIN_ACCOUNT_SPEND_ENABLED", "true")
    reserve: Final = AsyncMock(return_value=(("paid", 1),))
    monkeypatch.setattr("litellm.proxy.tokenin.ledger.reserve_account_request", reserve)
    monkeypatch.setattr("litellm.proxy.tokenin.enforcement.estimate_account_max_cost", AsyncMock(return_value=2.5))
    client: Final = SimpleNamespace(db=store)
    with pytest.raises(HTTPException) as refused:
        await accounts.reserve_managed_request(
            client, ACCOUNT, FLAT_TEAM_ID, "/chat/completions", {"model": "model-a", **fallback}, "sk-abc"
        )
    assert refused.value.status_code == 400
    assert not reserve.await_count
    assert (
        await accounts.reserve_managed_request(
            client, "not-enrolled", FLAT_TEAM_ID, "/chat/completions", {"model": "model-a", **fallback}, "sk-abc"
        )
        is None
    )
    assert await accounts.reserve_managed_request(
        client, ACCOUNT, FLAT_TEAM_ID, "/chat/completions", {"model": "model-a", "fallbacks": []}, "sk-abc"
    )
    assert reserve.await_count == 1


@pytest.mark.parametrize("config", ["alias", "silent_model"])
@pytest.mark.asyncio
async def test_managed_static_routing_mismatch_creates_no_hold(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch, config: str
) -> None:
    import litellm
    from litellm.proxy import proxy_server

    store.accounts.add(ACCOUNT)
    monkeypatch.setenv("TOKENIN_ACCOUNT_SPEND_ENABLED", "true")
    model_a: Final = {
        "model_name": "model-a",
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": "fake",
            **({"silent_model": "model-b"} if config == "silent_model" else {}),
        },
    }
    router: Final = litellm.Router(
        model_list=[
            model_a,
            {"model_name": "model-b", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake"}},
        ],
        model_group_alias={"model-a": "model-b"} if config == "alias" else None,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    estimator: Final = AsyncMock(return_value=1.0)
    reserve: Final = AsyncMock()
    monkeypatch.setattr("litellm.proxy.tokenin.enforcement.estimate_account_max_cost", estimator)
    monkeypatch.setattr("litellm.proxy.tokenin.ledger.reserve_account_request", reserve)
    client: Final = SimpleNamespace(db=store)
    with pytest.raises(HTTPException) as rejected:
        await accounts.reserve_managed_request(
            client,  # pyright: ignore[reportArgumentType]  # FakeTx-backed Prisma stand-in
            ACCOUNT,
            FLAT_TEAM_ID,
            "/chat/completions",
            {"model": "model-a"},
            "sk-abc",
        )
    assert rejected.value.status_code == 503
    estimator.assert_not_awaited()
    reserve.assert_not_awaited()
    assert (
        await accounts.reserve_managed_request(
            client,  # pyright: ignore[reportArgumentType]  # FakeTx-backed Prisma stand-in
            "not-enrolled",
            FLAT_TEAM_ID,
            "/chat/completions",
            {"model": "model-a"},
            "sk-abc",
        )
        is None
    )


@pytest.mark.asyncio
async def test_account_key_issuance_derives_identity_and_carries_no_limits(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.management_endpoints import key_management_endpoints

    store.accounts.add(ACCOUNT)
    issued: Final = AsyncMock(return_value={"token": "sk-account-key", "token_id": "key-1", "key_name": "account-1"})
    monkeypatch.setattr(key_management_endpoints, "generate_key_helper_fn", issued)
    app: Final = FastAPI()
    app.include_router(accounts.router)
    transport: Final = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        denied: Final = await client.post("/tokenin/account/keys", json={"user_id": ACCOUNT, "alias": "laptop"})
        assert denied.status_code == 403
        forbidden_fields: Final = await client.post(
            "/tokenin/account/keys",
            headers={"Authorization": "Bearer very-long-test-secret-32-characters-minimum"},
            json={"user_id": ACCOUNT, "alias": "laptop", "plan_id": "medium", "max_budget": 50, "rpm_limit": 99},
        )
        assert forbidden_fields.status_code == 422
        unknown_account: Final = await client.post(
            "/tokenin/account/keys",
            headers={"Authorization": "Bearer very-long-test-secret-32-characters-minimum"},
            json={"user_id": "not-enrolled", "alias": "laptop"},
        )
        assert unknown_account.status_code == 409
        created: Final = await client.post(
            "/tokenin/account/keys",
            headers={"Authorization": "Bearer very-long-test-secret-32-characters-minimum"},
            json={"user_id": ACCOUNT, "alias": "laptop"},
        )
        assert created.status_code == 200
        assert created.json()["key"] == "sk-account-key"
        assert created.json()["user_id"] == ACCOUNT
        assert created.json()["expires"] is None
    assert issued.await_args is not None
    assert issued.await_args.kwargs["user_id"] == ACCOUNT
    assert issued.await_args.kwargs["team_id"] == FLAT_TEAM_ID
    assert issued.await_args.kwargs["models"] == KEY_MODELS
    assert not {
        "max_budget",
        "budget_duration",
        "budget_reset_at",
        "budget_limits",
        "expires",
        "rpm_limit",
        "max_parallel_requests",
        "plan_id",
    } & set(issued.await_args.kwargs)


@pytest.mark.asyncio
async def test_bad_plan_amount_period_and_model_scope_fail_closed(store: FakeTx) -> None:
    for request in (
        accounts.GrantRequest(user_id=ACCOUNT, idempotency_key="u", plan_id="unknown", kind="fixed"),
        accounts.GrantRequest(
            user_id=ACCOUNT,
            idempotency_key="m",
            plan_id="medium",
            kind="fixed",
            subscription_id="s",
            period_index=0,
            period_start=PERIOD_START,
            period_end=PERIOD_END,
            topup_usd=Decimal("9"),
        ),
    ):
        with pytest.raises(HTTPException) as denied:
            await accounts.grant_account(request)
        assert denied.value.status_code == 400
    with pytest.raises(HTTPException) as bad_policy:
        await accounts.set_account_policy(
            accounts.PolicyRequest(
                user_id=ACCOUNT,
                idempotency_key="p",
                plan_id="medium",
                models=["*"],
                rpm_limit=20,
                max_parallel_requests=1,
                effective_at=PERIOD_START,
            )
        )
    assert bad_policy.value.status_code == 400
    with pytest.raises(ValueError):
        accounts.GrantRequest(
            user_id=ACCOUNT, idempotency_key="n", plan_id="payg", kind="payg", topup_usd=Decimal("NaN")
        )
    assert not store.grants


@pytest.mark.asyncio
async def test_weekly_catalog_without_monthly_value_rejects_grant(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        accounts,
        "load_plans",
        lambda: [
            TokeninPlan(
                id="medium",
                name="Medium",
                kind="fixed",
                rpm_limit=20,
                max_parallel_requests=1,
                max_budget=2.5,
                budget_duration="7d",
            )
        ],
    )
    with pytest.raises(HTTPException) as denied:
        await accounts.grant_account(_fixed("weekly-without-monthly-credit"))
    assert denied.value.status_code == 400
    assert not store.grants


@pytest.mark.asyncio
async def test_http_routes_reject_generic_key_and_accept_service_secret(store: FakeTx) -> None:
    app: Final = FastAPI()
    app.include_router(accounts.router)
    transport: Final = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized: Final = await client.post(
            "/tokenin/account/grants",
            headers={"Authorization": "Bearer generic-proxy-master-key"},
            json=_fixed("http-purchase:0").model_dump(mode="json"),
        )
        assert unauthorized.status_code == 403
        granted: Final = await client.post(
            "/tokenin/account/grants",
            headers={"Authorization": "Bearer very-long-test-secret-32-characters-minimum"},
            json=_fixed("http-purchase:0").model_dump(mode="json"),
        )
        assert granted.status_code == 200
        assert granted.json()["credited"] == 10.0
        recovered: Final = await client.get(
            "/tokenin/account/grants/http-purchase:0",
            params={"user_id": ACCOUNT},
            headers={"Authorization": "Bearer very-long-test-secret-32-characters-minimum"},
        )
        assert recovered.status_code == 200
        assert recovered.json()["duplicate"] is True
        assert recovered.json()["period_start"].endswith("Z")
        policy: Final = await client.post(
            "/tokenin/account/policy",
            headers={"Authorization": "Bearer very-long-test-secret-32-characters-minimum"},
            json={
                "user_id": ACCOUNT,
                "idempotency_key": "policy-http",
                "plan_id": "medium",
                "models": ["model-a"],
                "rpm_limit": 20,
                "max_parallel_requests": 1,
                "effective_at": "2026-02-28T00:00:00.000Z",
            },
        )
        assert policy.status_code == 200
        assert policy.json()["effective_at"].endswith("Z")
        catalog: Final = await client.get(
            "/tokenin/account/catalog",
            headers={"Authorization": "Bearer very-long-test-secret-32-characters-minimum"},
        )
        assert catalog.status_code == 200
        assert catalog.json()["plans"][0]["monthly_value_usd"] == 10.0
        assert catalog.json()["plans"][1]["monthly_value_usd"] is None


@pytest.mark.asyncio
async def test_status_reports_the_switches_that_gate_account_spend(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch
) -> None:
    app: Final = FastAPI()
    app.include_router(accounts.router)
    transport: Final = httpx.ASGITransport(app=app)
    monkeypatch.setenv("TOKENIN_ACCOUNT_V2_ENABLED", "true")
    monkeypatch.setenv("TOKENIN_ACCOUNT_SERVICE_TOKEN", "very-long-test-secret-32-characters-minimum")
    auth: Final = {"Authorization": "Bearer very-long-test-secret-32-characters-minimum"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        denied: Final = await client.get("/tokenin/account/status")
        assert denied.status_code == 403
        records: Final = await client.get("/tokenin/account/status", headers=auth)
        assert records.status_code == 200
        assert records.json() == {
            "v2_enabled": True,
            "spend_enabled": False,
            "enforcement_active": False,
            "supported_routes": [
                "/chat/completions",
                "/embeddings",
                "/images/generations",
                "/models",
                "/responses",
                "/v1/chat/completions",
                "/v1/embeddings",
                "/v1/images/generations",
                "/v1/messages",
                "/v1/models",
                "/v1/responses",
            ],
        }
        monkeypatch.setenv("TOKENIN_ACCOUNT_SPEND_ENABLED", "true")
        armed: Final = await client.get("/tokenin/account/status", headers=auth)
        assert armed.json()["enforcement_active"] is True


@pytest.mark.asyncio
async def test_managed_model_lists_are_request_local_policy_scopes(store: FakeTx, monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    await accounts.grant_account(_fixed("model-list"))
    assert (await accounts.set_account_policy(
        accounts.PolicyRequest(
            user_id=ACCOUNT,
            idempotency_key="model-list-policy",
            plan_id="medium",
            models=["model-a"],
            rpm_limit=20,
            max_parallel_requests=1,
            effective_at=PERIOD_START,
            expected_policy_id=None,
        )
    ))["duplicate"] is False
    router: Final = SimpleNamespace(
        get_model_names=lambda: ["model-a", "model-b"],
        get_model_access_groups=lambda: {},
        get_fully_blocked_model_names=lambda: set(),
        get_model_list=lambda: [],
        get_configured_token_limits=lambda model: (None, None),
    )
    token: Final = UserAPIKeyAuth(user_id=ACCOUNT, team_id=FLAT_TEAM_ID, models=[*KEY_MODELS], team_models=[])
    request: Final = Request({"type": "http", "method": "GET", "path": "/v1/models", "headers": []})
    monkeypatch.setenv("TOKENIN_ACCOUNT_SPEND_ENABLED", "true")
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=store))
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server, "get_hidden_unhealthy_model_names", AsyncMock(return_value=set()))

    for path in ("/v1/models", "/models"):
        listed: Final = await proxy_server.model_list(request=request, user_api_key_dict=token)
        assert [entry["id"] for entry in listed["data"]] == ["model-a"]
        assert path in accounts._MANAGED_MODEL_READ_ROUTES  # pyright: ignore[reportPrivateUsage]
    expanded: Final = await proxy_server.model_list(
        request=request, user_api_key_dict=token, scope="expand", include_model_access_groups=True
    )
    assert [entry["id"] for entry in expanded["data"]] == ["model-a"]
    assert token.models == [*KEY_MODELS]
    assert token.team_models == []


def test_dynamic_model_read_route_is_info_for_common_checks() -> None:
    from litellm.proxy.auth.route_checks import RouteChecks

    token: Final = UserAPIKeyAuth(user_id=ACCOUNT, team_id=FLAT_TEAM_ID, user_role=None, models=[])
    request: Final = Request({"type": "http", "method": "GET", "path": "/v1/models/model-a", "headers": []})
    for route in ("/v1/models/model-a", "/models/model-a"):
        assert RouteChecks.is_info_route(route) is True
        assert RouteChecks.is_llm_api_route(route) is False
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=None,
            route=route,
            request=request,
            valid_token=token,
            request_data={},
        )
    assert RouteChecks.is_info_route("/v1/models/model-a/nested") is False


@pytest.mark.asyncio
async def test_managed_model_detail_filters_to_policy_without_admin_model_info(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server

    await accounts.grant_account(_fixed("model-detail"))
    await accounts.set_account_policy(
        accounts.PolicyRequest(
            user_id=ACCOUNT,
            idempotency_key="model-detail-policy",
            plan_id="medium",
            models=["model-a"],
            rpm_limit=20,
            max_parallel_requests=1,
            effective_at=PERIOD_START,
        )
    )
    token: Final = UserAPIKeyAuth(user_id=ACCOUNT, team_id=FLAT_TEAM_ID, models=[*KEY_MODELS], team_models=[])
    router: Final = SimpleNamespace(
        get_model_names=lambda: ["model-a", "model-b"],
        get_model_access_groups=lambda: {},
        get_fully_blocked_model_names=lambda: set(),
        get_model_list=lambda: [],
        get_configured_token_limits=lambda model: (None, None),
        get_deployment_by_model_group_name=lambda model: SimpleNamespace(
            litellm_params=SimpleNamespace(model=f"openai/{model}"), model_info={}
        ),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=store))
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server, "get_hidden_unhealthy_model_names", AsyncMock(return_value=set()))
    detail: Final = await proxy_server.model_info(model_id="model-a", user_api_key_dict=token)
    assert detail["id"] == "model-a"
    with pytest.raises(HTTPException) as denied:
        await proxy_server.model_info(model_id="model-b", user_api_key_dict=token)
    assert denied.value.status_code == 404


@pytest.mark.asyncio
async def test_model_catalog_lists_only_configured_aliases_for_service_token(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", SimpleNamespace(get_model_names=lambda: ["model-b", "model-a"]))
    app: Final = FastAPI()
    app.include_router(accounts.router)
    transport: Final = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        denied: Final = await client.get("/tokenin/account/models")
        assert denied.status_code == 403
        models: Final = await client.get(
            "/tokenin/account/models",
            headers={"Authorization": "Bearer very-long-test-secret-32-characters-minimum"},
        )
        assert models.status_code == 200
        assert models.json() == {"models": ["model-a", "model-b"], "enforcement_active": False}
    monkeypatch.setattr(proxy_server, "llm_router", None)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unavailable: Final = await client.get(
            "/tokenin/account/models",
            headers={"Authorization": "Bearer very-long-test-secret-32-characters-minimum"},
        )
        assert unavailable.status_code == 503


@pytest.mark.asyncio
async def test_enrolled_account_calls_fail_closed_and_legacy_mode_is_unchanged(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch
) -> None:
    db: Final = SimpleNamespace(db=store)
    await accounts.grant_account(_fixed("enrollment"))
    with pytest.raises(HTTPException) as denied:
        await accounts.require_request_admission(db, ACCOUNT, None, "/v1/chat/completions")
    assert denied.value.status_code == 503
    with pytest.raises(HTTPException) as unowned:
        await accounts.require_request_admission(db, None, FLAT_TEAM_ID, "/v1/chat/completions")
    assert unowned.value.status_code == 503
    with pytest.raises(HTTPException) as outage:
        await accounts.require_request_admission(None, ACCOUNT, None, "/v1/chat/completions")
    assert outage.value.status_code == 503
    await accounts.require_request_admission(db, "unenrolled", None, "/v1/chat/completions")
    monkeypatch.setenv("TOKENIN_ACCOUNT_V2_ENABLED", "false")
    await accounts.require_request_admission(None, ACCOUNT, None, "/v1/chat/completions")
    assert await accounts.is_managed_account(db, ACCOUNT) is False


@pytest.mark.asyncio
async def test_account_auth_denial_cannot_use_db_outage_fallback_or_admin_role(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server

    await accounts.grant_account(_fixed("auth-managed"))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=store))
    monkeypatch.setattr(proxy_server, "general_settings", {"allow_requests_on_db_unavailable": True})
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        SimpleNamespace(post_call_failure_hook=AsyncMock(return_value=None)),
    )
    admin: Final = UserAPIKeyAuth(user_id=ACCOUNT, team_id=FLAT_TEAM_ID, user_role=LitellmUserRoles.PROXY_ADMIN)
    request: Final = _request("managed-key")
    with pytest.raises(ProxyException) as denied:
        await _authorize_authenticated_request(
            user_api_key_auth_obj=admin,
            request=request,
            request_data={"model": "model-a", "messages": []},
            route="/v1/chat/completions",
            api_key="managed-key",
        )
    assert denied.value.code == "503"


@pytest.mark.parametrize("v2_enabled", [True, False])
@pytest.mark.asyncio
async def test_early_auth_db_fallback_identity_cannot_skip_v2_admission(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch, v2_enabled: bool
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.auth import user_api_key_auth as auth_module
    from litellm.proxy.auth.auth_exception_handler import (
        DB_UNAVAILABLE_FALLBACK_USER_ID,
        UserAPIKeyAuthExceptionHandler,
    )

    monkeypatch.setenv("TOKENIN_ACCOUNT_V2_ENABLED", "true" if v2_enabled else "false")
    monkeypatch.setenv("TOKENIN_ACCOUNT_SPEND_ENABLED", "true")
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=store))
    monkeypatch.setattr(proxy_server, "general_settings", {"allow_requests_on_db_unavailable": True})
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        SimpleNamespace(
            post_call_failure_hook=AsyncMock(return_value=None),
            service_logging_obj=SimpleNamespace(service_failure_hook=MagicMock()),
        ),
    )
    monkeypatch.setattr(auth_module, "_run_centralized_common_checks", AsyncMock(return_value=None))
    reserve: Final = AsyncMock()
    monkeypatch.setattr("litellm.proxy.tokenin.ledger.reserve_account_request", reserve)
    request: Final = _request("managed-key")
    recovered: Final = await UserAPIKeyAuthExceptionHandler._handle_authentication_error(  # pyright: ignore[reportPrivateUsage]  # exercise the builder's DB fallback handler
        e=httpx.ConnectError("database unreachable"),
        request=request,
        request_data={"model": "model-a"},
        route="/v1/chat/completions",
        parent_otel_span=None,
        api_key="managed-key",
    )
    assert recovered.user_id == DB_UNAVAILABLE_FALLBACK_USER_ID
    admission: Final = _authorize_authenticated_request(
        user_api_key_auth_obj=recovered,
        request=request,
        request_data={"model": "model-a"},
        route="/v1/chat/completions",
        api_key="managed-key",
    )
    if v2_enabled:
        with pytest.raises(ProxyException) as refused:
            await admission
        assert refused.value.code == "503"
    else:
        assert await admission is None
    reserve.assert_not_awaited()


@pytest.mark.parametrize("v2_enabled", [True, False])
@pytest.mark.asyncio
async def test_db_outage_during_account_reservation_never_mints_the_fallback_identity(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch, v2_enabled: bool
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.auth import user_api_key_auth as auth_module
    from litellm.proxy.auth.auth_exception_handler import DB_UNAVAILABLE_FALLBACK_USER_ID

    monkeypatch.setenv("TOKENIN_ACCOUNT_V2_ENABLED", "true" if v2_enabled else "false")
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=store))
    monkeypatch.setattr(proxy_server, "general_settings", {"allow_requests_on_db_unavailable": True})
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        SimpleNamespace(
            post_call_failure_hook=AsyncMock(return_value=None),
            service_logging_obj=SimpleNamespace(service_failure_hook=MagicMock()),
        ),
    )
    monkeypatch.setattr(auth_module, "_run_centralized_common_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "litellm.proxy.tokenin.accounts.reserve_managed_request",
        AsyncMock(side_effect=httpx.ConnectError("database unreachable")),
    )
    managed: Final = UserAPIKeyAuth(user_id=ACCOUNT, team_id=FLAT_TEAM_ID)
    call: Final = _authorize_authenticated_request(
        user_api_key_auth_obj=managed,
        request=_request("managed-key"),
        request_data={"model": "model-a", "messages": []},
        route="/v1/chat/completions",
        api_key="managed-key",
    )
    if v2_enabled:
        with pytest.raises(ProxyException) as denied:
            await call
        assert denied.value.code == "503"
    else:
        recovered: Final = await call
        assert recovered is not None and recovered.user_id == DB_UNAVAILABLE_FALLBACK_USER_ID


@pytest.mark.parametrize("route", ["/v1/audio/transcriptions", "/v1/audio/speech", "/model/info"])
@pytest.mark.asyncio
async def test_paid_llm_route_fails_closed_for_enrolled_account(store: FakeTx, monkeypatch: pytest.MonkeyPatch, route: str) -> None:
    await accounts.grant_account(_fixed("route-scope"))
    monkeypatch.setenv("TOKENIN_ACCOUNT_SPEND_ENABLED", "true")
    with pytest.raises(HTTPException) as denied:
        await accounts.require_request_admission(
            SimpleNamespace(db=store), ACCOUNT, FLAT_TEAM_ID, route, "GET" if route == "/model/info" else "POST"
        )
    assert denied.value.status_code == 503


@pytest.mark.parametrize(
    "route",
    [
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/responses",
        "/v1/images/generations",
        "/v1/messages",
    ],
)
@pytest.mark.asyncio
async def test_paid_estimable_llm_routes_admit_enrolled_account(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    await accounts.grant_account(_fixed("route-scope"))
    monkeypatch.setenv("TOKENIN_ACCOUNT_SPEND_ENABLED", "true")
    assert await accounts.require_request_admission(SimpleNamespace(db=store), ACCOUNT, FLAT_TEAM_ID, route, "POST") is None


def _request(token: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/tokenin/account/grants",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        }
    )


def test_new_api_requires_separate_service_secret_and_feature_gate(
    store: FakeTx, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(HTTPException) as generic:
        accounts._service_only(_request("generic-proxy-key"))
    assert generic.value.status_code == 403
    accounts._service_only(_request("very-long-test-secret-32-characters-minimum"))
    monkeypatch.setenv("TOKENIN_ACCOUNT_V2_ENABLED", "false")
    with pytest.raises(HTTPException) as disabled:
        accounts._service_only(_request("very-long-test-secret-32-characters-minimum"))
    assert disabled.value.status_code == 404


NANO: Final = 1_000_000_000
MAR31: Final = datetime(2026, 3, 31, tzinfo=timezone.utc)
APR30: Final = datetime(2026, 4, 30, tzinfo=timezone.utc)
SERVICE_AUTH: Final = {"Authorization": "Bearer very-long-test-secret-32-characters-minimum"}


def _payg(key: str, amount: str) -> accounts.GrantRequest:
    return accounts.GrantRequest(
        user_id=ACCOUNT, idempotency_key=key, plan_id="payg", kind="payg", topup_usd=Decimal(amount)
    )


async def _paid_months(*indexes: int) -> None:
    bounds: Final = (PERIOD_START, PERIOD_END, MAR31, APR30)
    for index in indexes:
        await accounts.grant_account(_fixed(f"p-{index}", index=index, start=bounds[index], end=bounds[index + 1]))


def _buckets(balance: dict[str, object]) -> dict[str, dict[str, object]]:
    rows: Final = balance["buckets"]
    assert isinstance(rows, list)
    return {str(row["grant_id"]): row for row in rows}


@pytest.mark.asyncio
async def test_balance_current_month_only_expires_at_its_period_end(store: FakeTx) -> None:
    await _paid_months(0)
    store.now = datetime(2026, 2, 1)
    balance: Final = await accounts.account_balance(ACCOUNT)
    bucket: Final = _buckets(balance)["p-0"]
    assert bucket["state"] == "active"
    assert bucket["expires_at"] == PERIOD_END
    assert bucket["period_start"] == PERIOD_START
    assert (bucket["amount_usd"], bucket["used_usd"], bucket["remaining_usd"]) == (10.0, 0.0, 10.0)
    assert balance["as_of"] == datetime(2026, 2, 1, tzinfo=timezone.utc)
    assert balance["fixed_available_usd"] == balance["available_usd"] == 10.0
    assert balance["payg_available_usd"] == balance["debt_usd"] == 0.0
    assert balance["enforcement_active"] is False


@pytest.mark.asyncio
async def test_balance_paid_next_month_extends_expiry_and_is_upcoming_until_boundary(store: FakeTx) -> None:
    await _paid_months(0, 1)
    store.now = datetime(2026, 2, 1)
    before: Final = await accounts.account_balance(ACCOUNT)
    assert _buckets(before)["p-0"]["state"] == "active"
    assert _buckets(before)["p-0"]["expires_at"] == MAR31
    assert _buckets(before)["p-1"]["state"] == "upcoming"
    assert _buckets(before)["p-1"]["expires_at"] == MAR31
    assert before["fixed_available_usd"] == 10.0
    store.now = PERIOD_END.replace(tzinfo=None)
    after: Final = await accounts.account_balance(ACCOUNT)
    assert {row["state"] for row in _buckets(after).values()} == {"active"}
    assert after["fixed_available_usd"] == 20.0


@pytest.mark.asyncio
async def test_balance_after_boundary_with_three_paid_months_expires_the_first(store: FakeTx) -> None:
    from litellm.proxy.tokenin.ledger import eligible_grants, load_grants

    await _paid_months(0, 1, 2)
    store.now = MAR31.replace(tzinfo=None)
    balance: Final = await accounts.account_balance(ACCOUNT)
    buckets: Final = _buckets(balance)
    assert (buckets["p-0"]["state"], buckets["p-0"]["expires_at"]) == ("expired", MAR31)
    assert (buckets["p-1"]["state"], buckets["p-1"]["expires_at"]) == ("active", APR30)
    assert (buckets["p-2"]["state"], buckets["p-2"]["expires_at"]) == ("active", APR30)
    assert balance["fixed_available_usd"] == balance["available_usd"] == 20.0
    # Spending and the balance read share one window rule, so they can never disagree.
    spendable: Final = eligible_grants(await load_grants(store, ACCOUNT), store.now)
    assert {grant.grant_id for grant in spendable} == {key for key, row in buckets.items() if row["state"] == "active"}


@pytest.mark.asyncio
async def test_balance_payment_gap_expires_at_own_period_end(store: FakeTx) -> None:
    await _paid_months(0, 2)
    store.now = PERIOD_END.replace(tzinfo=None)
    buckets: Final = _buckets(await accounts.account_balance(ACCOUNT))
    assert (buckets["p-0"]["state"], buckets["p-0"]["expires_at"]) == ("expired", PERIOD_END)
    assert (buckets["p-2"]["state"], buckets["p-2"]["expires_at"]) == ("upcoming", APR30)


@pytest.mark.asyncio
async def test_balance_payg_is_always_active_without_expiry(store: FakeTx) -> None:
    await accounts.grant_account(_payg("topup-1", "8.25"))
    for now in (datetime(2020, 1, 1), datetime(2099, 1, 1)):
        store.now = now
        balance = await accounts.account_balance(ACCOUNT)
        bucket = _buckets(balance)["topup-1"]
        assert (bucket["state"], bucket["expires_at"], bucket["period_start"]) == ("active", None, None)
        assert balance["payg_available_usd"] == balance["available_usd"] == 8.25
        assert balance["fixed_available_usd"] == 0.0


@pytest.mark.asyncio
async def test_balance_counts_held_reservations_as_used(store: FakeTx) -> None:
    await _paid_months(0)
    await accounts.grant_account(_payg("topup-1", "5"))
    store.allocations[("held-request", "p-0")] = 3 * NANO
    store.allocations[("settled-request", "p-0")] = NANO // 2
    store.allocations[("overrun", "topup-1")] = 6 * NANO
    balance: Final = await accounts.account_balance(ACCOUNT)
    buckets: Final = _buckets(balance)
    assert (buckets["p-0"]["used_usd"], buckets["p-0"]["remaining_usd"]) == (3.5, 6.5)
    assert (buckets["topup-1"]["used_usd"], buckets["topup-1"]["remaining_usd"]) == (6.0, 0.0)
    assert balance["fixed_available_usd"] == balance["available_usd"] == 6.5


@pytest.mark.asyncio
async def test_balance_debt_reduces_available_and_floors_at_zero(store: FakeTx) -> None:
    await _paid_months(0)
    await accounts.grant_account(_payg("topup-1", "2"))
    store.debt[ACCOUNT] = 4 * NANO
    owed: Final = await accounts.account_balance(ACCOUNT)
    assert (owed["debt_usd"], owed["available_usd"]) == (4.0, 8.0)
    assert (owed["fixed_available_usd"], owed["payg_available_usd"]) == (10.0, 2.0)
    store.debt[ACCOUNT] = 50 * NANO
    assert (await accounts.account_balance(ACCOUNT))["available_usd"] == 0.0


@pytest.mark.asyncio
async def test_balance_unknown_account_is_404_but_enrolled_account_reads_zero(store: FakeTx) -> None:
    with pytest.raises(HTTPException) as missing:
        await accounts.account_balance("nobody")
    assert (missing.value.status_code, missing.value.detail) == (404, "Account not found")
    store.accounts.add(ACCOUNT)
    empty: Final = await accounts.account_balance(ACCOUNT)
    assert (empty["available_usd"], empty["debt_usd"], empty["buckets"]) == (0.0, 0.0, [])


@pytest.mark.asyncio
async def test_balance_http_route_is_service_only(store: FakeTx, monkeypatch: pytest.MonkeyPatch) -> None:
    await _paid_months(0)
    app: Final = FastAPI()
    app.include_router(accounts.router)
    transport: Final = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        anonymous: Final = await client.get("/tokenin/account/balance", params={"user_id": ACCOUNT})
        assert anonymous.status_code == 403
        generic: Final = await client.get(
            "/tokenin/account/balance",
            params={"user_id": ACCOUNT},
            headers={"Authorization": "Bearer generic-proxy-master-key"},
        )
        assert generic.status_code == 403
        served: Final = await client.get("/tokenin/account/balance", params={"user_id": ACCOUNT}, headers=SERVICE_AUTH)
        assert served.status_code == 200
        body: Final = served.json()
        assert body["as_of"].endswith("Z")
        assert body["buckets"][0]["expires_at"] == "2026-02-28T00:00:00Z"
        assert body["available_usd"] == 10.0
        missing: Final = await client.get(
            "/tokenin/account/balance", params={"user_id": "nobody"}, headers=SERVICE_AUTH
        )
        assert missing.status_code == 404
        monkeypatch.setenv("TOKENIN_ACCOUNT_V2_ENABLED", "false")
        disabled: Final = await client.get(
            "/tokenin/account/balance", params={"user_id": ACCOUNT}, headers=SERVICE_AUTH
        )
        assert disabled.status_code == 404
