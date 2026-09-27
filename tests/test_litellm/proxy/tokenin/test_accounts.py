from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import _authorize_authenticated_request
from litellm.proxy.tokenin import accounts
from litellm.proxy.tokenin.plans import FLAT_TEAM_ID, TokeninPlan

ACCOUNT: Final = "customer-123"
PERIOD_START: Final = datetime(2026, 1, 31, tzinfo=timezone.utc)
PERIOD_END: Final = datetime(2026, 2, 28, tzinfo=timezone.utc)


class FakeTx:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.accounts: set[str] = set()
        self.grants: dict[str, dict[str, object]] = {}
        self.policies: dict[str, dict[str, object]] = {}

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
                "period_start": start,
                "period_end": end,
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
                "effective_at": effective,
                "created_at": datetime(2026, 1, 1) + timedelta(seconds=len(self.policies)),
            }
            return 1
        raise AssertionError(sql)

    async def query_raw(self, sql: str, *values: object) -> list[dict[str, object]]:
        if '"LiteLLM_TokeninAccount"' in sql:
            return [{"user_id": values[0]}] if values[0] in self.accounts else []
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
            return newest_first[:1] if "LIMIT 1" in sql else newest_first
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


@pytest.mark.parametrize(
    "route",
    ["/v1/chat/completions", "/v1/embeddings", "/v1/responses", "/v1/images/generations", "/v1/messages"],
)
@pytest.mark.asyncio
async def test_paid_llm_route_fails_closed_for_enrolled_account(store: FakeTx, route: str) -> None:
    await accounts.grant_account(_fixed("route-scope"))
    with pytest.raises(HTTPException) as denied:
        await accounts.require_request_admission(SimpleNamespace(db=store), ACCOUNT, FLAT_TEAM_ID, route)
    assert denied.value.status_code == 503


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
