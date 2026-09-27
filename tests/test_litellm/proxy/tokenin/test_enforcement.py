from __future__ import annotations

from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

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
async def test_client_cancel_refunds_only_when_the_provider_produced_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel: Final = AsyncMock(return_value=True)
    uncertain: Final = AsyncMock(return_value=True)
    monkeypatch.setattr(enforcement, "_prisma_client", lambda: object())
    monkeypatch.setattr("litellm.proxy.tokenin.ledger.cancel_account_request", cancel)
    monkeypatch.setattr("litellm.proxy.tokenin.ledger.mark_account_request_uncertain", uncertain)
    metadata: Final = {enforcement.TOKENIN_ACCOUNT_HOLD_METADATA_KEY: HOLD}
    await enforcement.handle_account_hold_on_cancel(metadata, provider_output_delivered=False)
    cancel.assert_awaited_once()
    uncertain.assert_not_awaited()
    cancel.reset_mock()
    await enforcement.handle_account_hold_on_cancel(metadata, provider_output_delivered=True)
    uncertain.assert_awaited_once()
    cancel.assert_not_awaited()
    uncertain.reset_mock()
    await enforcement.handle_account_hold_on_cancel({}, provider_output_delivered=False)
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
