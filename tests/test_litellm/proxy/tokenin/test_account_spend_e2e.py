"""Local end-to-end acceptance run for managed-account spend.

Runs the real proxy app (ASGI) against a local ephemeral Postgres, with the V2 and
spend switches on and a priced model served by litellm's mock provider, so no external
call is made. Run it with:

    scripts/tokenin_e2e_acceptance.sh

Requires the Prisma python query engine for this platform: macOS arm64 only ships the
node addon, so these tests run on the Linux CI/VPS image where `prisma py fetch`
installs `prisma-query-engine-<platform>` and the proxy's own prisma client works.

Every assertion here is a money-path invariant: what a customer is charged, what is
refused, and that nothing is refunded for work that may have been billed.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Final

import httpx
import pytest
import pytest_asyncio
import yaml

SERVICE_TOKEN: Final = "e2e-service-token-32-characters-minimum"
MASTER_KEY: Final = "sk-e2e-master-key"
STUB_MODEL: Final = "gpt-4o-mini"
OTHER_MODEL: Final = "other-priced-model"

pytestmark = pytest.mark.skipif(
    not os.environ.get("TOKENIN_E2E_DATABASE_URL"),
    reason="set TOKENIN_E2E_DATABASE_URL to a local ephemeral Postgres to run this acceptance test",
)


def _config(database_url: str) -> dict[str, Any]:
    return {
        "model_list": [
            {
                "model_name": OTHER_MODEL,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-stub-provider",
                    "mock_response": "stubbed reply",
                },
            },
            {
                "model_name": STUB_MODEL,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-stub-provider",
                    "mock_response": "stubbed reply",
                },
            },
        ],
        "general_settings": {"master_key": MASTER_KEY, "database_url": database_url},
        "litellm_settings": {"num_retries": 0},
        "tokenin_plans": [
            {
                "id": "medium",
                "name": "Medium",
                "kind": "fixed",
                "rpm_limit": 100,
                "max_parallel_requests": 10,
                "max_budget": 5.0,
                "monthly_value": 5.0,
            },
            {
                "id": "payg",
                "name": "PAYG",
                "kind": "payg",
                "rpm_limit": 100,
                "max_parallel_requests": 10,
                "max_budget": 0.0,
            },
        ],
    }


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def proxy_app():
    database_url: Final = os.environ["TOKENIN_E2E_DATABASE_URL"]
    os.environ["TOKENIN_ACCOUNT_V2_ENABLED"] = "true"
    os.environ["TOKENIN_ACCOUNT_SPEND_ENABLED"] = "true"
    os.environ["TOKENIN_ACCOUNT_SERVICE_TOKEN"] = SERVICE_TOKEN
    os.environ["LITELLM_MASTER_KEY"] = MASTER_KEY

    config_path: Final = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False).name
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.dump(_config(database_url), handle)
    os.environ["CONFIG_FILE_PATH"] = config_path

    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import app, cleanup_router_config_variables, initialize, proxy_startup_event

    cleanup_router_config_variables()
    await initialize(config=config_path)
    async with proxy_startup_event(app):
        assert proxy_server.prisma_client is not None
        await proxy_server.prisma_client.check_view_exists()
        yield app


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client(proxy_app) -> httpx.AsyncClient:
    transport: Final = httpx.ASGITransport(app=proxy_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://e2e", timeout=30.0) as http_client:
        yield http_client


@pytest_asyncio.fixture(loop_scope="module")
async def prisma(proxy_app):
    from litellm.proxy import proxy_server

    return proxy_server.prisma_client


async def _service_post(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> httpx.Response:
    return await client.post(path, json=payload, headers={"Authorization": f"Bearer {SERVICE_TOKEN}"})


async def _service_get(client: httpx.AsyncClient, path: str, params: dict[str, Any]) -> httpx.Response:
    return await client.get(path, params=params, headers={"Authorization": f"Bearer {SERVICE_TOKEN}"})


async def _enrolled_account(
    client: httpx.AsyncClient, prisma, *, models: list[str], rpm: int = 100, parallel: int = 10
) -> tuple[str, str]:
    """Enroll a fresh account, give it a policy, and return (user_id, key)."""
    user_id: Final = f"e2e-{uuid.uuid4().hex[:12]}"
    await prisma.db.execute_raw(
        'INSERT INTO "LiteLLM_UserTable" ("user_id", "user_email") VALUES ($1, $2) ON CONFLICT DO NOTHING',
        user_id,
        f"{user_id}@example.test",
    )
    policy: Final = await _service_post(
        client,
        "/tokenin/account/policy",
        {
            "user_id": user_id,
            "idempotency_key": f"policy-{user_id}",
            "plan_id": "payg",
            "models": models,
            "rpm_limit": rpm,
            "max_parallel_requests": parallel,
            "effective_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        },
    )
    assert policy.status_code == 200, policy.text
    issued: Final = await _service_post(client, "/tokenin/account/keys", {"user_id": user_id, "alias": f"{user_id}-k1"})
    assert issued.status_code == 200, issued.text
    return user_id, issued.json()["key"]


async def _issue_key(client: httpx.AsyncClient, user_id: str, alias: str) -> str:
    issued: Final = await _service_post(client, "/tokenin/account/keys", {"user_id": user_id, "alias": alias})
    assert issued.status_code == 200, issued.text
    return issued.json()["key"]


async def _grant(client: httpx.AsyncClient, user_id: str, usd: str) -> httpx.Response:
    return await _service_post(
        client,
        "/tokenin/account/grants",
        {
            "user_id": user_id,
            "idempotency_key": f"topup-{uuid.uuid4().hex}",
            "plan_id": "payg",
            "kind": "payg",
            "topup_usd": usd,
        },
    )


async def _chat(client: httpx.AsyncClient, key: str, *, model: str = STUB_MODEL, stream: bool = False):
    return await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": stream},
    )


async def _account_state(prisma, user_id: str) -> dict[str, Any]:
    holds: Final = await prisma.db.query_raw(
        'SELECT h."request_id", h."state", h."estimated_nano", h."charged_nano", '
        'COALESCE(SUM(a."amount_nano"),0) AS "held_nano" FROM "LiteLLM_TokeninHold" h '
        'LEFT JOIN "LiteLLM_TokeninAllocation" a ON a."request_id" = h."request_id" '
        'WHERE h."user_id" = $1 GROUP BY h."request_id"',
        user_id,
    )
    debt: Final = await prisma.db.query_raw(
        'SELECT "debt_nano" FROM "LiteLLM_TokeninAccount" WHERE "user_id" = $1', user_id
    )
    return {
        "debt_nano": int(debt[0]["debt_nano"]) if debt else 0,
        "holds": holds,
        "charged_nano": sum(int(row["charged_nano"] or 0) for row in holds),
        "open_nano": sum(int(row["held_nano"]) for row in holds if row["state"] == "held"),
    }


@pytest.mark.asyncio(loop_scope="module")
async def test_zero_credit_account_is_refused_and_writes_no_spend(client, prisma) -> None:
    user_id, key = await _enrolled_account(client, prisma, models=[STUB_MODEL])
    denied: Final = await _chat(client, key)
    assert denied.status_code == 402, denied.text
    spend_logs: Final = await prisma.db.query_raw(
        'SELECT COUNT(*) AS "count" FROM "LiteLLM_SpendLogs" WHERE "user" = $1', user_id
    )
    assert int(spend_logs[0]["count"]) == 0
    assert (await _account_state(prisma, user_id))["charged_nano"] == 0


@pytest.mark.asyncio(loop_scope="module")
async def test_one_grant_is_shared_by_two_keys_and_never_overspent(client, prisma) -> None:
    user_id, key_one = await _enrolled_account(client, prisma, models=[STUB_MODEL])
    key_two: Final = await _issue_key(client, user_id, f"{user_id}-k2")
    granted: Final = await _grant(client, user_id, "0.00005")
    assert granted.status_code == 200, granted.text
    grant_nano: Final = int(round(float(granted.json()["credited"]) * 1_000_000_000))

    allowed: Final[list[int]] = []
    denied: Final[list[int]] = []
    for index in range(12):
        response: Final = await _chat(client, key_one if index % 2 == 0 else key_two)
        (allowed if response.status_code == 200 else denied).append(response.status_code)
    assert allowed, "the grant must be spendable through both keys"
    assert set(denied) <= {402}, denied
    state: Final = await _account_state(prisma, user_id)
    assert state["charged_nano"] <= grant_nano, state
    assert state["charged_nano"] > 0, state
    assert state["debt_nano"] == 0, state
    assert state["open_nano"] == 0, state


@pytest.mark.asyncio(loop_scope="module")
async def test_model_outside_the_account_policy_is_denied_for_every_key(client, prisma) -> None:
    user_id, key_one = await _enrolled_account(client, prisma, models=[OTHER_MODEL])
    key_two: Final = await _issue_key(client, user_id, f"{user_id}-k2")
    await _grant(client, user_id, "0.01")
    for key in (key_one, key_two):
        denied: Final = await _chat(client, key)
        assert denied.status_code == 403, denied.text


@pytest.mark.asyncio(loop_scope="module")
async def test_rotating_keys_moves_no_credit_and_legacy_routes_refuse_enrolled_accounts(client, prisma) -> None:
    user_id, _key = await _enrolled_account(client, prisma, models=[STUB_MODEL])
    await _grant(client, user_id, "0.01")
    before: Final = await prisma.db.query_raw(
        'SELECT COALESCE(SUM("amount_nano"),0) AS "total" FROM "LiteLLM_TokeninAllocation" a '
        'JOIN "LiteLLM_TokeninHold" h ON h."request_id" = a."request_id" WHERE h."user_id" = $1',
        user_id,
    )
    await _issue_key(client, user_id, f"{user_id}-rotated")
    after: Final = await prisma.db.query_raw(
        'SELECT COALESCE(SUM("amount_nano"),0) AS "total" FROM "LiteLLM_TokeninAllocation" a '
        'JOIN "LiteLLM_TokeninHold" h ON h."request_id" = a."request_id" WHERE h."user_id" = $1',
        user_id,
    )
    assert before == after
    legacy: Final = await client.post(
        "/tokenin/key/generate",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json={"user_id": user_id, "alias": f"{user_id}-legacy"},
    )
    assert legacy.status_code == 409, legacy.text


@pytest.mark.asyncio(loop_scope="module")
async def test_rpm_is_shared_across_keys_of_the_account(client, prisma) -> None:
    user_id, key_one = await _enrolled_account(client, prisma, models=[STUB_MODEL], rpm=1)
    key_two: Final = await _issue_key(client, user_id, f"{user_id}-k2")
    await _grant(client, user_id, "0.01")
    first: Final = await _chat(client, key_one)
    assert first.status_code == 200, first.text
    second: Final = await _chat(client, key_two)
    assert second.status_code == 429, second.text


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_completion_settles_the_actual_cost(client, prisma) -> None:
    user_id, key = await _enrolled_account(client, prisma, models=[STUB_MODEL])
    await _grant(client, user_id, "0.01")
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": STUB_MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        body: Final = "".join([chunk async for chunk in response.aiter_text()])
    assert "data:" in body
    state: Final = await _account_state(prisma, user_id)
    assert state["holds"], "a stream must leave a settled hold"
    assert all(row["state"] == "settled" for row in state["holds"]), state
    assert state["charged_nano"] > 0, state
    assert state["open_nano"] == 0, state


@pytest.mark.asyncio(loop_scope="module")
async def test_client_disconnect_never_drops_credit_silently(client, prisma) -> None:
    user_id, key = await _enrolled_account(client, prisma, models=[STUB_MODEL])
    await _grant(client, user_id, "0.01")
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": STUB_MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        async for _chunk in response.aiter_text():
            break
    state: Final = await _account_state(prisma, user_id)
    for row in state["holds"]:
        assert row["state"] in {"settled", "cancelled", "uncertain"}, state
        if row["state"] == "uncertain":
            assert int(row["held_nano"]) > 0, "an uncertain hold must keep its reservation"
    assert state["debt_nano"] == 0, state


@pytest.mark.asyncio(loop_scope="module")
async def test_operator_reconciliation_clears_a_pinned_hold(client, prisma) -> None:
    user_id, key = await _enrolled_account(client, prisma, models=[STUB_MODEL])
    await _grant(client, user_id, "0.01")
    request_id: Final = f"tokenin-pinned-{uuid.uuid4().hex[:8]}"
    await prisma.db.execute_raw(
        'INSERT INTO "LiteLLM_TokeninHold" '
        '("request_id", "user_id", "key_hash", "model", "state", "estimated_nano", "admitted_at") '
        "VALUES ($1,$2,'pinned','gpt-4o-mini','held',1000000000,$3)",
        request_id,
        user_id,
        datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=30),
    )
    grant_row: Final = await prisma.db.query_raw(
        'SELECT "idempotency_key" FROM "LiteLLM_TokeninGrant" WHERE "user_id" = $1 LIMIT 1', user_id
    )
    await prisma.db.execute_raw(
        'INSERT INTO "LiteLLM_TokeninAllocation" ("request_id", "grant_id", "amount_nano") VALUES ($1,$2,$3)',
        request_id,
        grant_row[0]["idempotency_key"],
        1_000_000_000,
    )
    listed: Final = await _service_get(
        client, "/tokenin/account/holds", {"user_id": user_id, "states": "held,uncertain", "older_than_minutes": 10}
    )
    assert listed.status_code == 200, listed.text
    assert [row["request_id"] for row in listed.json()["holds"]] == [request_id]
    resolved: Final = await _service_post(
        client,
        f"/tokenin/account/holds/{request_id}/resolve",
        {"action": "cancel", "note": "e2e: pinned hold cleared by operator"},
    )
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["state"] == "cancelled"
    state: Final = await _account_state(prisma, user_id)
    assert state["open_nano"] == 0, state
    assert (await _chat(client, key)).status_code == 200
