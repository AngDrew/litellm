"""Provider-reported cost is billed with the global cost margin, exactly once, and never shown raw. Offline."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Iterator
from typing import Any, Final

import httpx
import openai
import pytest

import litellm
from callbacks.jev_provider import _reported_cost  # pyright: ignore[reportPrivateUsage]
from litellm.cost_calculator import response_cost_calculator
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.spend_tracking.budget_reservation import _with_cost_margin  # pyright: ignore[reportPrivateUsage]
from litellm.types.utils import ModelResponse, Usage

HEADER: Final = "llm_provider-x-litellm-response-cost"
INPUT_RATE: Final = 1e-6
OUTPUT_RATE: Final = 5e-6
PROMPT_TOKENS: Final = 1000
COMPLETION_TOKENS: Final = 100
TOKEN_COST: Final = PROMPT_TOKENS * INPUT_RATE + COMPLETION_TOKENS * OUTPUT_RATE  # 0.0015
PROVIDER_COST: Final = 0.01
MARGIN: Final = 0.30
OR_MODEL: Final = "openrouter/margin-test/model"
DI_MODEL: Final = "deepinfra/margin-test/model"


@pytest.fixture(autouse=True)
def priced_models(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name, provider in ((OR_MODEL, "openrouter"), (DI_MODEL, "deepinfra")):
        monkeypatch.setitem(
            litellm.model_cost,
            name,
            {
                "input_cost_per_token": INPUT_RATE,
                "output_cost_per_token": OUTPUT_RATE,
                "litellm_provider": provider,
                "mode": "chat",
            },
        )
    monkeypatch.setattr(litellm, "cost_margin_config", {"global": MARGIN})
    monkeypatch.setattr(litellm, "cost_discount_config", {})
    yield


def _response(*, header: Any = None, estimated: Any = None) -> ModelResponse:
    response = ModelResponse(
        model="x",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        usage=Usage(
            prompt_tokens=PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
            total_tokens=PROMPT_TOKENS + COMPLETION_TOKENS,
        ),
    )
    if header is not None:
        response._hidden_params["additional_headers"] = {HEADER: header}
    if estimated is not None:
        setattr(response.usage, "estimated_cost", estimated)
    return response


def _bill(
    response: ModelResponse, *, model: str = OR_MODEL, provider: str = "openrouter", call_type: str = "completion"
) -> float:
    return response_cost_calculator(
        response_object=response,
        model=model.split("/", 1)[1],
        custom_llm_provider=provider,
        call_type=call_type,  # type: ignore[arg-type]
        optional_params={},
    )


class TestMarginOnProviderCost:
    def test_the_global_margin_is_added_to_the_providers_cost(self) -> None:
        assert _bill(_response(header=PROVIDER_COST)) == pytest.approx(PROVIDER_COST * 1.30)

    def test_pricing_twice_does_not_add_the_margin_twice(self) -> None:
        response = _response(header=PROVIDER_COST)
        assert _bill(response) == pytest.approx(_bill(response))
        assert response._hidden_params["additional_headers"][HEADER] == PROVIDER_COST  # header stays the raw number

    def test_a_provider_entry_replaces_the_global_margin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "cost_margin_config", {"global": MARGIN, "openrouter": 0.15})
        assert _bill(_response(header=PROVIDER_COST)) == pytest.approx(PROVIDER_COST * 1.15)

    def test_without_a_margin_the_providers_cost_is_billed_as_is(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "cost_margin_config", {})
        assert _bill(_response(header=PROVIDER_COST)) == PROVIDER_COST

    def test_a_cache_hit_is_free(self) -> None:
        assert (
            response_cost_calculator(
                response_object=_response(header=PROVIDER_COST),
                model="margin-test/model",
                custom_llm_provider="openrouter",
                call_type="completion",
                optional_params={},
                cache_hit=True,
            )
            == 0.0
        )

    def test_no_provider_cost_prices_from_tokens_with_the_margin(self) -> None:
        assert _bill(_response()) == pytest.approx(TOKEN_COST * 1.30)

    def test_the_breakdown_records_the_providers_cost_and_the_margin(self) -> None:
        recorded: dict[str, Any] = {}

        class Logging:
            def set_cost_breakdown(self, **kwargs: Any) -> None:
                recorded.update(kwargs)

        response_cost_calculator(
            response_object=_response(header=PROVIDER_COST),
            model="margin-test/model",
            custom_llm_provider="openrouter",
            call_type="completion",
            optional_params={},
            litellm_logging_obj=Logging(),  # type: ignore[arg-type]
        )
        assert recorded["original_cost"] == PROVIDER_COST
        assert recorded["total_cost"] == pytest.approx(PROVIDER_COST * 1.30)
        assert recorded["margin_total_amount"] == pytest.approx(PROVIDER_COST * 0.30)


class TestProviderCostValidity:
    @pytest.mark.parametrize("reported", [-0.01, math.nan, math.inf, -math.inf], ids=["negative", "nan", "inf", "-inf"])
    def test_a_malformed_cost_is_ignored_and_the_call_is_priced_from_tokens(self, reported: float) -> None:
        assert _bill(_response(header=reported)) == pytest.approx(TOKEN_COST * 1.30)

    @pytest.mark.parametrize(
        "reported",
        [0.0, TOKEN_COST / 100, 100 * TOKEN_COST],
        ids=["zero", "far-under-token-estimate", "far-over-token-estimate"],
    )
    def test_every_finite_non_negative_cost_is_billed_with_the_margin(self, reported: float) -> None:
        assert _bill(_response(header=reported)) == pytest.approx(reported * 1.30)

    def test_a_garbage_header_falls_back_to_tokens_instead_of_failing_the_request(self) -> None:
        assert _bill(_response(header="not-a-number")) == pytest.approx(TOKEN_COST * 1.30)

    def test_a_model_without_prices_has_nothing_to_compare_against_so_the_cost_is_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delitem(litellm.model_cost, OR_MODEL)
        assert _bill(_response(header=PROVIDER_COST)) == pytest.approx(PROVIDER_COST * 1.30)


class TestDeepInfraEstimatedCost:
    def test_it_is_billed_with_the_margin(self) -> None:
        assert _bill(_response(estimated=PROVIDER_COST), model=DI_MODEL, provider="deepinfra") == pytest.approx(
            PROVIDER_COST * 1.30
        )

    def test_other_providers_estimated_cost_is_not_a_charge(self) -> None:
        assert _bill(_response(estimated=PROVIDER_COST), model=OR_MODEL, provider="openrouter") == pytest.approx(
            TOKEN_COST * 1.30
        )

    def test_a_large_estimate_is_billed_not_clamped(self) -> None:
        assert _bill(_response(estimated=1000 * TOKEN_COST), model=DI_MODEL, provider="deepinfra") == pytest.approx(
            1000 * TOKEN_COST * 1.30
        )

    def test_a_negative_estimate_is_ignored(self) -> None:
        assert _bill(_response(estimated=-1.0), model=DI_MODEL, provider="deepinfra") == pytest.approx(
            TOKEN_COST * 1.30
        )


class TestHoldsCarryTheMargin:
    def test_a_hold_estimate_is_marked_up_like_the_settled_spend(self) -> None:
        assert _with_cost_margin(1.0, {"litellm_provider": "openrouter"}) == pytest.approx(1.30)

    def test_a_provider_entry_applies_to_its_holds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "cost_margin_config", {"global": MARGIN, "openrouter": 0.15})
        assert _with_cost_margin(1.0, {"litellm_provider": "openrouter"}) == pytest.approx(1.15)

    def test_no_estimate_stays_no_estimate(self) -> None:
        assert _with_cost_margin(None, {}) is None

    def test_no_margin_leaves_the_estimate_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "cost_margin_config", {})
        assert _with_cost_margin(1.0, {}) == 1.0


class TestJevClassifierCost:
    def test_the_reported_cost_carries_the_margin(self) -> None:
        assert _reported_cost({"usage": {"cost": 0.01}}) == pytest.approx(0.013)

    @pytest.mark.parametrize("cost", [None, "x", True, -1.0, math.nan, math.inf])
    def test_an_unusable_cost_lets_litellm_price_from_the_deployment(self, cost: object) -> None:
        assert _reported_cost({"usage": {"cost": cost}}) is None

    def test_no_margin_keeps_the_reported_cost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "cost_margin_config", {})
        assert _reported_cost({"usage": {"cost": 0.01}}) == 0.01


class TestHeaderOnlyEverHoldsTheProvidersNumber:
    """The margin is added on top of the header, so a billed amount landing there would be marked up twice."""

    @staticmethod
    def _propagate(response: ModelResponse) -> None:
        CustomStreamWrapper._propagate_usage_cost_to_hidden_params(response, "openrouter")  # pyright: ignore[reportPrivateUsage]

    def test_the_providers_usage_cost_is_copied_to_the_header(self) -> None:
        response = _response()
        setattr(response.usage, "cost", PROVIDER_COST)
        self._propagate(response)
        assert response._hidden_params["additional_headers"][HEADER] == PROVIDER_COST

    def test_a_usage_cost_that_is_already_the_final_cost_is_not_copied(self) -> None:
        response = _response()
        setattr(response.usage, "cost", PROVIDER_COST * 1.30)
        response._hidden_params["response_cost"] = PROVIDER_COST * 1.30
        self._propagate(response)
        assert HEADER not in response._hidden_params.get("additional_headers", {})

    def test_the_providers_number_already_in_the_header_is_not_overwritten(self) -> None:
        response = _response(header=PROVIDER_COST)
        setattr(response.usage, "cost", PROVIDER_COST * 1.30)
        self._propagate(response)
        assert response._hidden_params["additional_headers"][HEADER] == PROVIDER_COST


class _Capture(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.costs: list[float | None] = []

    async def async_log_success_event(
        self, kwargs: dict, response_obj: object, start_time: object, end_time: object
    ) -> None:
        # Logging tasks left over from other tests in this worker can fire here too; keep only our own calls.
        if "margin-test" in str(kwargs.get("model")):
            self.costs.append(kwargs.get("response_cost"))


CAPTURE: Final = _Capture()


def _usage_body(extra: dict) -> dict:
    return {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS, "total_tokens": 1100, **extra}


def _json_body(extra: dict) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": _usage_body(extra),
    }


def _sse_body(extra: dict) -> str:
    def line(data: dict) -> str:
        return "data: " + json.dumps(data) + "\n\n"

    base = {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    return (
        line(
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}]}
        )
        + line({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        + line({**base, "choices": [], "usage": _usage_body(extra)})
        + "data: [DONE]\n\n"
    )


async def _call(model: str, base_url: str, usage_extra: dict, *, stream: bool) -> list[float | None]:
    def handler(_request: httpx.Request) -> httpx.Response:
        if stream:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse_body(usage_extra).encode()
            )
        return httpx.Response(200, json=_json_body(usage_extra))

    transport = httpx.MockTransport(handler)
    client: Any
    if model.startswith("deepinfra/"):
        client = openai.AsyncOpenAI(api_key="k", base_url=base_url, http_client=httpx.AsyncClient(transport=transport))
    else:
        client = AsyncHTTPHandler()
        client.client = httpx.AsyncClient(transport=transport)
    CAPTURE.costs = []
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        api_key="k",
        api_base=base_url,
        stream=stream,
        client=client,
        **({"stream_options": {"include_usage": True}} if stream else {}),
    )
    if stream:
        async for _chunk in response:  # type: ignore[union-attr]
            pass
    for _ in range(50):  # the success callback runs after the response is returned
        if CAPTURE.costs:
            break
        await asyncio.sleep(0.05)
    return CAPTURE.costs


async def _client_usage(model: str, base_url: str, usage_extra: dict, *, stream: bool) -> Usage:
    """The usage object the caller of acompletion ends up holding (the last chunk's, when streaming)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        if stream:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse_body(usage_extra).encode()
            )
        return httpx.Response(200, json=_json_body(usage_extra))

    transport = httpx.MockTransport(handler)
    client: Any
    if model.startswith("deepinfra/"):
        client = openai.AsyncOpenAI(api_key="k", base_url=base_url, http_client=httpx.AsyncClient(transport=transport))
    else:
        client = AsyncHTTPHandler()
        client.client = httpx.AsyncClient(transport=transport)
    CAPTURE.costs = []
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        api_key="k",
        api_base=base_url,
        stream=stream,
        client=client,
        **({"stream_options": {"include_usage": True}} if stream else {}),
    )
    last: Usage | None = None
    if stream:
        async for chunk in response:  # type: ignore[union-attr]
            last = getattr(chunk, "usage", None) or last
    else:
        last = response.usage  # type: ignore[union-attr]
    for _ in range(50):  # let this call's success callback finish so it cannot land in the next test
        if CAPTURE.costs:
            break
        await asyncio.sleep(0.05)
    assert last is not None
    return last


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> _Capture:
    monkeypatch.setattr(litellm, "callbacks", [CAPTURE])
    return CAPTURE


OR_URL: Final = "https://openrouter.ai/api/v1"
DI_URL: Final = "https://api.deepinfra.com/v1/openai"


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streaming"])
class TestWhatTheClientSees:
    """The caller is never shown the provider's own cost: only what it is billed, or nothing."""

    async def test_openrouter_usage_cost_is_the_billed_amount(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch, stream: bool
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)  # as in the proxy config
        usage = await _client_usage(OR_MODEL, OR_URL, {"cost": PROVIDER_COST}, stream=stream)
        assert usage.cost == pytest.approx(PROVIDER_COST * 1.30)  # type: ignore[attr-defined]

    async def test_deepinfra_estimated_cost_is_not_shown(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch, stream: bool
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
        usage = await _client_usage(DI_MODEL, DI_URL, {"estimated_cost": PROVIDER_COST}, stream=stream)
        assert getattr(usage, "estimated_cost", None) is None


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streaming"])
class TestCostVisibleOnCacheHits:
    async def _second_call_usage(self, model: str, url: str, extra: dict, stream: bool) -> Usage:
        content = f"cache-{model}-{stream}-{id(extra)}"
        for _ in range(2):

            def handler(_request: httpx.Request) -> httpx.Response:
                if stream:
                    return httpx.Response(
                        200, headers={"content-type": "text/event-stream"}, content=_sse_body(extra).encode()
                    )
                return httpx.Response(200, json=_json_body(extra))

            transport = httpx.MockTransport(handler)
            client: Any
            if model.startswith("deepinfra/"):
                client = openai.AsyncOpenAI(
                    api_key="k", base_url=url, http_client=httpx.AsyncClient(transport=transport)
                )
            else:
                client = AsyncHTTPHandler()
                client.client = httpx.AsyncClient(transport=transport)
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": content}],
                api_key="k",
                api_base=url,
                stream=stream,
                client=client,
                caching=True,
                **({"stream_options": {"include_usage": True}} if stream else {}),
            )
            last: Usage | None = None
            if stream:
                async for chunk in response:  # type: ignore[union-attr]
                    last = getattr(chunk, "usage", None) or last
            else:
                last = response.usage  # type: ignore[union-attr]
            await asyncio.sleep(0.3)
        assert last is not None
        return last

    @pytest.fixture(autouse=True)
    def local_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "cache", litellm.Cache(type="local"))

    async def test_a_cached_openrouter_cost_is_zero_when_cost_is_included(
        self, monkeypatch: pytest.MonkeyPatch, stream: bool
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
        usage = await self._second_call_usage(OR_MODEL, OR_URL, {"cost": PROVIDER_COST}, stream)
        assert usage.cost == 0  # type: ignore[attr-defined]

    async def test_a_cached_openrouter_cost_is_omitted_when_cost_is_not_included(
        self, monkeypatch: pytest.MonkeyPatch, stream: bool
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", False)
        usage = await self._second_call_usage(OR_MODEL, OR_URL, {"cost": PROVIDER_COST}, stream)
        assert getattr(usage, "cost", None) is None

    @pytest.mark.parametrize("include_cost", [False, True], ids=["usage-cost-off", "usage-cost-on"])
    async def test_a_cached_deepinfra_estimated_cost_is_not_shown(
        self, monkeypatch: pytest.MonkeyPatch, stream: bool, include_cost: bool
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", include_cost)
        usage = await self._second_call_usage(DI_MODEL, DI_URL, {"estimated_cost": PROVIDER_COST}, stream)
        assert getattr(usage, "estimated_cost", None) is None


class TestStreamingUsageCostVisibility:
    async def test_the_billed_cost_replaces_the_provider_cost_even_when_cost_is_not_included(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", False)
        usage = await _client_usage(OR_MODEL, OR_URL, {"cost": PROVIDER_COST}, stream=True)
        assert usage.cost == pytest.approx(PROVIDER_COST * 1.30)  # type: ignore[attr-defined]

    async def test_a_token_priced_stream_shows_the_billed_cost_and_bills_it_once(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", False)
        usage = await _client_usage(OR_MODEL, OR_URL, {}, stream=True)
        assert usage.cost == pytest.approx(TOKEN_COST * 1.30)  # type: ignore[attr-defined]
        assert capture.costs == [pytest.approx(TOKEN_COST * 1.30)]

    async def test_a_zero_provider_cost_is_shown_as_the_billed_zero(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
        usage = await _client_usage(OR_MODEL, OR_URL, {"cost": 0.0}, stream=True)
        assert usage.cost == 0.0  # type: ignore[attr-defined]

    def test_reassembling_client_chunks_does_not_apply_the_margin_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
        billed: Final = PROVIDER_COST * (1 + MARGIN)
        chunks: Final = [
            {
                "id": "chatcmpl-margin",
                "created": 1,
                "model": "margin-test/model",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-margin",
                "created": 2,
                "model": "margin-test/model",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": PROMPT_TOKENS,
                    "completion_tokens": COMPLETION_TOKENS,
                    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                    "cost": billed,
                },
                "_hidden_params": {"response_cost": billed},
            },
        ]

        class Logging:
            model_call_details: Final = {"custom_llm_provider": "openrouter"}

            def _response_cost_calculator(self, result: ModelResponse) -> float:
                existing: Final = result._hidden_params.get("response_cost")
                if isinstance(existing, (int, float)) and not isinstance(existing, bool):
                    return float(existing)
                return response_cost_calculator(
                    response_object=result,
                    model="margin-test/model",
                    custom_llm_provider="openrouter",
                    call_type="completion",
                    optional_params={},
                )

        rebuilt = litellm.stream_chunk_builder(chunks=chunks, logging_obj=Logging())  # type: ignore[arg-type]
        assert rebuilt is not None
        assert rebuilt._hidden_params["response_cost"] == pytest.approx(billed)
        assert rebuilt.usage.cost == pytest.approx(billed)  # type: ignore[union-attr]


class TestProxyStreamedUsageCost:
    @staticmethod
    def _frame(cost: object) -> dict:
        return {
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS, "cost": cost},
        }

    def test_the_providers_reported_cost_is_billed_with_the_margin_not_re_estimated_from_tokens(self) -> None:
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        class Logging:
            def _response_cost_calculator(self, result: ModelResponse) -> float | None:
                return response_cost_calculator(
                    response_object=result,
                    model="margin-test/model",
                    custom_llm_provider="openrouter",
                    call_type="completion",
                    optional_params={},
                )

        out = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(
            self._frame(PROVIDER_COST),
            OR_MODEL,
            Logging(),  # type: ignore[arg-type]
        )
        assert out is not None
        assert out["usage"]["cost"] == pytest.approx(PROVIDER_COST * 1.30)

    def test_a_raw_cost_that_cannot_be_priced_is_not_forwarded(self) -> None:
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        class Logging:
            def _response_cost_calculator(self, result: ModelResponse) -> float | None:
                return None

        out = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(
            self._frame(PROVIDER_COST),
            "no-such-model",
            Logging(),  # type: ignore[arg-type]
        )
        assert out is not None
        assert "cost" not in out["usage"]


def test_the_proxy_does_not_forward_the_providers_cost_header() -> None:
    from litellm.cost_calculator import PROVIDER_COST_HEADER
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
        user_api_key_dict=UserAPIKeyAuth(),
        response_cost=PROVIDER_COST * 1.30,
        **{PROVIDER_COST_HEADER: str(PROVIDER_COST), "llm_provider-x-request-id": "abc"},
    )
    assert PROVIDER_COST_HEADER not in headers
    assert headers["x-litellm-response-cost"] == str(PROVIDER_COST * 1.30)
    assert headers["llm_provider-x-request-id"] == "abc"  # other provider headers still pass


@pytest.mark.parametrize("include_cost_in_stream", [False, True], ids=["usage-cost-off", "usage-cost-on"])
@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streaming"])
class TestEndToEnd:
    """Real acompletion calls against a mocked upstream; the cost is what the logging callback records."""

    async def test_openrouter_cost_is_billed_once_with_the_margin(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch, stream: bool, include_cost_in_stream: bool
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", include_cost_in_stream)
        assert await _call(OR_MODEL, OR_URL, {"cost": PROVIDER_COST}, stream=stream) == [
            pytest.approx(PROVIDER_COST * 1.30)
        ]

    async def test_openrouter_without_a_cost_is_priced_from_tokens_with_the_margin(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch, stream: bool, include_cost_in_stream: bool
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", include_cost_in_stream)
        assert await _call(OR_MODEL, OR_URL, {}, stream=stream) == [pytest.approx(TOKEN_COST * 1.30)]

    async def test_openrouter_negative_cost_is_priced_from_tokens(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch, stream: bool, include_cost_in_stream: bool
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", include_cost_in_stream)
        assert await _call(OR_MODEL, OR_URL, {"cost": -1.0}, stream=stream) == [pytest.approx(TOKEN_COST * 1.30)]

    async def test_openrouter_surcharge_far_above_token_cost_is_billed_with_the_margin(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch, stream: bool, include_cost_in_stream: bool
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", include_cost_in_stream)
        assert await _call(OR_MODEL, OR_URL, {"cost": 100 * TOKEN_COST}, stream=stream) == [
            pytest.approx(100 * TOKEN_COST * 1.30)
        ]

    async def test_openrouter_zero_cost_is_billed_as_zero(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch, stream: bool, include_cost_in_stream: bool
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", include_cost_in_stream)
        assert await _call(OR_MODEL, OR_URL, {"cost": 0.0}, stream=stream) == [0.0]

    async def test_deepinfra_estimated_cost_is_billed_once_with_the_margin(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch, stream: bool, include_cost_in_stream: bool
    ) -> None:
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", include_cost_in_stream)
        assert await _call(DI_MODEL, DI_URL, {"estimated_cost": PROVIDER_COST}, stream=stream) == [
            pytest.approx(PROVIDER_COST * 1.30)
        ]
