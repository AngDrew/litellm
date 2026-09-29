from __future__ import annotations

from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import TypeAdapter
from starlette.requests import Request

from litellm.proxy import litellm_pre_call_utils
from litellm.proxy.tokenin import enforcement

HOLD: Final = "tokenin-hold-abc"
AUTH: Final = SimpleNamespace(tokenin_account_hold_id=HOLD)


def _metadata(**buckets: object) -> dict[str, object]:
    return {
        "litellm_metadata": buckets.get("litellm_metadata", {}),
        "metadata": buckets.get("metadata", {}),
    }


def test_hold_id_is_found_in_either_bucket_or_on_the_auth_object() -> None:
    stamped: Final = {enforcement.TOKENIN_ACCOUNT_HOLD_METADATA_KEY: HOLD}
    assert enforcement.account_hold_id(stamped) == HOLD
    assert enforcement.account_hold_id({"user_api_key_auth": AUTH}) == HOLD
    assert enforcement.account_hold_id({"user_api_key_auth": {"tokenin_account_hold_id": HOLD}}) == HOLD
    assert enforcement.account_hold_id({}) is None
    assert enforcement.account_hold_id(None) is None
    assert enforcement.hold_metadata_from_request_data(_metadata(litellm_metadata=stamped)) == stamped
    assert enforcement.hold_metadata_from_request_data(_metadata(metadata=stamped)) == stamped
    assert enforcement.hold_metadata_from_request_data({"metadata": {}}) is None
    assert enforcement.hold_metadata_from_request_data(None) is None


def test_client_supplied_hold_id_is_stripped_from_both_buckets() -> None:
    forged: Final = {
        "metadata": {enforcement.TOKENIN_ACCOUNT_HOLD_METADATA_KEY: "someone-elses-hold"},
        "litellm_metadata": {enforcement.TOKENIN_ACCOUNT_HOLD_METADATA_KEY: "someone-elses-hold"},
    }
    litellm_pre_call_utils._strip_router_reserved_metadata(forged)
    assert forged == {"metadata": {}, "litellm_metadata": {}}


@pytest.mark.asyncio
async def test_settle_mark_and_cancel_reach_the_ledger_once(monkeypatch: pytest.MonkeyPatch) -> None:
    settle: Final = AsyncMock(return_value=1)
    uncertain: Final = AsyncMock(return_value=True)
    cancel: Final = AsyncMock(return_value=True)
    client: Final = object()
    monkeypatch.setattr(enforcement, "_prisma_client", lambda: client)
    monkeypatch.setattr("litellm.proxy.tokenin.ledger.settle_account_request", settle)
    monkeypatch.setattr("litellm.proxy.tokenin.ledger.mark_account_request_uncertain", uncertain)
    monkeypatch.setattr("litellm.proxy.tokenin.ledger.cancel_account_request", cancel)
    metadata: Final = {enforcement.TOKENIN_ACCOUNT_HOLD_METADATA_KEY: HOLD}
    await enforcement.settle_account_hold(metadata=metadata, actual_cost=1.25)
    await enforcement.mark_account_hold_uncertain(metadata=metadata)
    await enforcement.cancel_account_hold(metadata=metadata)
    settle.assert_awaited_once_with(client, HOLD, 1.25)
    uncertain.assert_awaited_once_with(client, HOLD)
    cancel.assert_awaited_once_with(client, HOLD)
    await enforcement.settle_account_hold(metadata={}, actual_cost=1.25)
    assert settle.await_count == 1


@pytest.mark.asyncio
async def test_client_cancel_refunds_only_when_billing_is_known_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel: Final = AsyncMock(return_value=True)
    uncertain: Final = AsyncMock(return_value=True)
    monkeypatch.setattr(enforcement, "_prisma_client", lambda: object())
    monkeypatch.setattr("litellm.proxy.tokenin.ledger.cancel_account_request", cancel)
    monkeypatch.setattr("litellm.proxy.tokenin.ledger.mark_account_request_uncertain", uncertain)
    metadata: Final = {enforcement.TOKENIN_ACCOUNT_HOLD_METADATA_KEY: HOLD}
    await enforcement.handle_account_hold_on_cancel(metadata, billing_known_absent=True)
    cancel.assert_awaited_once()
    uncertain.assert_not_awaited()
    cancel.reset_mock()
    await enforcement.handle_account_hold_on_cancel(metadata, billing_known_absent=False)
    uncertain.assert_awaited_once()
    cancel.assert_not_awaited()
    uncertain.reset_mock()
    await enforcement.handle_account_hold_on_cancel({}, billing_known_absent=True)
    cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_or_unavailable_settlement_never_raises_into_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "litellm.proxy.tokenin.ledger.settle_account_request",
        AsyncMock(side_effect=RuntimeError("database down")),
    )
    monkeypatch.setattr(
        "litellm.proxy.tokenin.ledger.cancel_account_request",
        AsyncMock(side_effect=RuntimeError("database down")),
    )
    metadata: Final = {enforcement.TOKENIN_ACCOUNT_HOLD_METADATA_KEY: HOLD}
    monkeypatch.setattr(enforcement, "_prisma_client", lambda: object())
    await enforcement.settle_account_hold(metadata=metadata, actual_cost=0.5)
    await enforcement.cancel_account_hold(metadata=metadata)
    monkeypatch.setattr(enforcement, "_prisma_client", lambda: None)
    await enforcement.settle_account_hold(metadata=metadata, actual_cost=0.5)


@pytest.mark.asyncio
async def test_max_cost_reserves_worst_case_deployment_and_rejects_unpriced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.spend_tracking import budget_reservation

    estimates: Final = {"cheap": 0.5, "expensive": 4.0}
    monkeypatch.setattr(budget_reservation, "_get_request_models", lambda **_: ("cheap", "expensive"))
    monkeypatch.setattr(budget_reservation, "count_request_input_tokens", AsyncMock(return_value={"cheap": 10}))
    monkeypatch.setattr(
        budget_reservation,
        "_estimate_request_max_cost_for_model",
        lambda model, **_: estimates.get(model),
    )
    assert await enforcement.estimate_account_max_cost({}, "/chat/completions", None) == 4.0

    estimates["expensive"] = None
    with pytest.raises(HTTPException) as unpriced:
        await enforcement.estimate_account_max_cost({}, "/chat/completions", None)
    assert unpriced.value.status_code == 503

    estimates["expensive"] = 0.0
    with pytest.raises(HTTPException) as uncapped:
        await enforcement.estimate_account_max_cost({}, "/chat/completions", None)
    assert uncapped.value.status_code == 503

    monkeypatch.setattr(budget_reservation, "_get_request_models", lambda **_: ())
    with pytest.raises(HTTPException) as unknown_model:
        await enforcement.estimate_account_max_cost({}, "/chat/completions", None)
    assert unknown_model.value.status_code == 503


def _pre_call_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


@pytest.mark.parametrize("hold_id", [HOLD, None])
@pytest.mark.asyncio
async def test_held_request_cannot_fall_back_even_when_key_metadata_reenables_it(hold_id: str | None) -> None:
    from unittest.mock import MagicMock

    from litellm.proxy._types import UserAPIKeyAuth

    auth: Final = UserAPIKeyAuth(
        api_key="hashed", user_id="customer", metadata={"disable_fallbacks": False}, tokenin_account_hold_id=hold_id
    )
    data: Final = TypeAdapter(dict[str, object]).validate_python(
        await litellm_pre_call_utils.add_litellm_data_to_request(
            data={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
            request=_pre_call_request(),
            user_api_key_dict=auth,
            proxy_config=MagicMock(),
            general_settings={},
        )
    )
    if hold_id is None:
        assert data["disable_fallbacks"] is False
        assert not any(field in data for field in enforcement.ACCOUNT_HOLD_FALLBACK_FIELDS)
    else:
        assert data["disable_fallbacks"] is True
        assert all(data[field] == [] for field in enforcement.ACCOUNT_HOLD_FALLBACK_FIELDS)


@pytest.mark.asyncio
async def test_pinned_request_ignores_router_configured_fallbacks() -> None:
    import litellm

    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake", "mock_response": name},
            }
            for name in ("model-a", "model-b")
        ],
        fallbacks=[{"model-a": ["model-b"]}],
        context_window_fallbacks=[{"model-a": ["model-b"]}],
        content_policy_fallbacks=[{"model-a": ["model-b"]}],
        num_retries=0,
    )
    unpinned: Final = await router.acompletion(
        model="model-a", messages=[{"role": "user", "content": "hi"}], mock_testing_fallbacks=True
    )
    assert isinstance(unpinned, litellm.ModelResponse)
    assert unpinned.choices[0].message.content == "model-b"
    pinned: Final[dict[str, object]] = {}
    enforcement.pin_account_hold_to_requested_model(pinned)
    # The exact kwargs the pre-call stage leaves on a held request, replayed against the router.
    assert pinned == {
        "disable_fallbacks": True,
        "fallbacks": [],
        "context_window_fallbacks": [],
        "content_policy_fallbacks": [],
    }
    with pytest.raises(litellm.InternalServerError):
        await router.acompletion(
            model="model-a",
            messages=[{"role": "user", "content": "hi"}],
            mock_testing_fallbacks=True,
            disable_fallbacks=True,
            fallbacks=[],
            context_window_fallbacks=[],
            content_policy_fallbacks=[],
        )
    with pytest.raises(litellm.InternalServerError):
        # Empty lists alone must also beat the router's configured fallbacks.
        await router.acompletion(
            model="model-a",
            messages=[{"role": "user", "content": "hi"}],
            mock_testing_fallbacks=True,
            fallbacks=[],
            context_window_fallbacks=[],
            content_policy_fallbacks=[],
        )


@pytest.mark.parametrize("alias", [True, False])
@pytest.mark.asyncio
async def test_held_route_refuses_alias_and_default_fallback_but_nonheld_routes(alias: bool) -> None:
    import litellm
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.route_llm_request import route_request

    deployment: Final = [
        {
            "model_name": "model-b",
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake", "mock_response": "model-b"},
        }
    ]
    held: Final = UserAPIKeyAuth(api_key="hashed", tokenin_account_hold_id=HOLD, tokenin_account_hold_model="model-a")
    direct_deployment: Final = {
        "model_name": "model-a",
        "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake", "mock_response": "model-a"},
    }
    router: Final = (
        litellm.Router(model_list=[direct_deployment, *deployment], model_group_alias={"model-a": "model-b"})
        if alias
        else litellm.Router(model_list=deployment, default_fallbacks=["model-b"])
    )
    request: Final = {"model": "model-a", "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(HTTPException) as blocked:
        await route_request(dict(request), router, None, "acompletion", held)
    assert blocked.value.status_code == 503
    if alias:
        aliased: Final = await (await route_request(dict(request), router, None, "acompletion"))
        assert isinstance(aliased, litellm.ModelResponse)
        assert aliased.choices[0].message.content == "model-b"
    else:
        fallback: Final = await router.acompletion(model="model-a", messages=[{"role": "user", "content": "hi"}])
        assert isinstance(fallback, litellm.ModelResponse)
        assert fallback.choices[0].message.content == "model-b"

    direct: Final = litellm.Router(model_list=[direct_deployment])
    allowed: Final = await (await route_request(dict(request), direct, None, "acompletion", held))
    assert isinstance(allowed, litellm.ModelResponse)
    assert allowed.choices[0].message.content == "model-a"
    with pytest.raises(HTTPException):
        await route_request({**request, "model": "model-b"}, direct, None, "acompletion", held)
    with pytest.raises(HTTPException):
        await route_request({**request, "user_config": {"model_list": deployment}}, direct, None, "acompletion", held)


@pytest.mark.parametrize(
    "body",
    [
        {"fallbacks": ["model-b"]},
        {"fallbacks": [{"model": "model-b"}]},
        {"context_window_fallbacks": [{"model-a": ["model-b"]}]},
        {"content_policy_fallbacks": [{"model-a": ["model-b"]}]},
        {"router_settings_override": {"fallbacks": [{"model-a": ["model-b"]}]}},
        {"fallbacks": "model-b"},
        {"router_settings_override": {}},
        {"router_settings_override": {"num_retries": 5}},
        {"router_settings_override": ["fallbacks"]},
        {"router_settings_override": "fallbacks"},
        {"router_settings_override": 0},
    ],
)
def test_any_named_fallback_target_or_router_override_is_detected(body: dict[str, object]) -> None:
    assert enforcement.requests_model_fallbacks({"model": "model-a", **body})


def test_absent_or_empty_fallbacks_are_not_a_request_for_one() -> None:
    assert not enforcement.requests_model_fallbacks({"model": "model-a"})
    assert not enforcement.requests_model_fallbacks(
        {"model": "model-a", "fallbacks": [], "context_window_fallbacks": None, "router_settings_override": None}
    )
