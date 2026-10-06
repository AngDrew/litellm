import asyncio
from decimal import Decimal
import json

import pytest

import litellm
from litellm import ModelResponse
from litellm.llms.custom_llm import CustomLLMError
from litellm.llms.neuralwatt.chat.handler import (
    NeuralWattChatCompletion,
    _apply_neuralwatt_metadata,
    _async_stream,
    _chunk_from_payload,
    _energy_headers,
    _final_metadata_chunks,
    _get_chat_completions_url,
    _get_markup_pct,
    _get_upstream_request_data,
    _get_upstream_model,
    _raise_for_status,
    _response_headers_with_retry_after,
    _record_neuralwatt_spend_metadata,
    _sanitize_cost,
)
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.proxy.proxy_server import _serialize_streaming_chunk, async_data_generator
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
from litellm.utils import CustomStreamWrapper


class MockLogging:
    def __init__(self):
        self.model_call_details = {}

    def pre_call(self, *args, **kwargs):
        self.pre_call_kwargs = kwargs

    def post_call(self, *args, **kwargs):
        self.post_call_kwargs = kwargs


class MockResponse:
    status_code = 200
    text = ""

    def iter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}'
        yield ': energy {"energy_joules": 4.99, "energy_kwh": 0.000001385, "avg_power_watts": 55.3, "duration_seconds": 0.361}'
        yield ': cost {"request_cost_usd": "0.01", "allowance_remaining_usd": 9}'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}'

    def json(self):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "glm-5.2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "energy": {
                "energy_joules": 4.99,
                "energy_kwh": 0.000001385,
                "avg_power_watts": 55.3,
                "duration_seconds": 0.361,
            },
            "cost": {"request_cost_usd": "0.01", "allowance_remaining_usd": 9},
        }


class MockErrorResponse:
    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class MockClient:
    def post(self, *, url, headers, data, **kwargs):
        self.url = url
        self.headers = headers
        self.data = json.loads(data)
        return MockResponse()


def test_neuralwatt_strips_provider_prefix():
    assert _get_upstream_model("neuralwatt/glm-5.2") == "glm-5.2"


def test_neuralwatt_completion_maps_request_and_response_metadata():
    client = MockClient()
    logging = MockLogging()

    response = NeuralWattChatCompletion().completion(
        model="neuralwatt/glm-5.2",
        messages=[{"role": "user", "content": "hello"}],
        api_base="https://api.neuralwatt.test/v1",
        custom_prompt_dict={},
        model_response=ModelResponse(),
        print_verbose=lambda _: None,
        encoding=None,
        api_key="sk-test",
        logging_obj=logging,
        optional_params={"temperature": 0.1, "markup_pct": 20},
        litellm_params={"markup_pct": 20},
        client=client,
    )

    assert client.url == "https://api.neuralwatt.test/v1/chat/completions"
    assert client.headers["Authorization"] == "Bearer sk-test"
    assert client.data == {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert response.usage.cost_usd == 0.012
    assert response.usage.cost == 0.012
    assert response.cost == {"request_cost_usd": 0.012}
    assert response._hidden_params["response_cost"] == 0.012
    assert logging.model_call_details["response_cost"] == 0.012
    assert response._hidden_params["additional_headers"] == {
        "X-Energy-Joules": "4.99",
        "X-Energy-Kwh": "1.385e-06",
        "X-Energy-Avg-Power-Watts": "55.3",
        "X-Energy-Duration-Seconds": "0.361",
    }
    assert logging.model_call_details["litellm_params"]["metadata"]["spend_logs_metadata"] == {
        "neuralwatt_energy": response.energy,
        "neuralwatt_cost": {
            "allowance_remaining_usd": 9,
            "upstream_request_cost_usd": "0.01",
            "request_cost_usd": 0.012,
            "markup_pct": 20.0,
        },
    }


def test_neuralwatt_records_spend_metadata_without_overwriting_existing_values():
    logging = MockLogging()
    logging.model_call_details["litellm_params"] = {"metadata": {"spend_logs_metadata": {"owner": "ops"}}}

    _record_neuralwatt_spend_metadata(
        logging_obj=logging,
        energy={"energy_joules": 2.13, "grid_id": "FI"},
        cost={"request_cost_usd": 3.75e-06},
    )

    assert logging.model_call_details["litellm_params"]["metadata"]["spend_logs_metadata"] == {
        "owner": "ops",
        "neuralwatt_energy": {"energy_joules": 2.13, "grid_id": "FI"},
        "neuralwatt_cost": {"request_cost_usd": 3.75e-06},
    }


def test_neuralwatt_requires_api_base():
    with pytest.raises(CustomLLMError):
        _get_chat_completions_url(None)


def test_neuralwatt_validates_markup_pct():
    assert _get_markup_pct({}) == Decimal("0")
    assert _get_markup_pct({"markup_pct": None}) == Decimal("0")
    assert _get_markup_pct({"markup_pct": "25"}) == Decimal("25")
    assert _get_markup_pct({"markup_pct": 10}, {"markup_pct": 20}) == Decimal("10")
    assert _get_markup_pct({}, {"markup_pct": 20}) == Decimal("20")

    with pytest.raises(CustomLLMError):
        _get_markup_pct({"markup_pct": -1})

    with pytest.raises(CustomLLMError):
        _get_markup_pct({"markup_pct": "bad"})


def test_neuralwatt_forces_stream_and_usage_without_clobbering_stream_options():
    request_data = _get_upstream_request_data(
        upstream_model="glm-5.2",
        messages=[
            {"role": "developer", "content": "follow instructions"},
            {"role": "user", "content": "hello"},
        ],
        optional_params={
            "stream": False,
            "stream_options": {"include_usage": False, "foo": "bar"},
            "markup_pct": 20,
        },
    )

    assert request_data == {
        "model": "glm-5.2",
        "messages": [
            {"role": "system", "content": "follow instructions"},
            {"role": "user", "content": "hello"},
        ],
        "stream": True,
        "stream_options": {"include_usage": True, "foo": "bar"},
    }


def test_neuralwatt_sanitizes_cost_and_applies_markup():
    cost = _sanitize_cost(
        {
            "request_cost_usd": "0.0000104",
            "allowance_remaining_usd": 32.68,
            "cache_savings_usd": 1,
        },
        Decimal("25"),
    )

    assert cost == {
        "allowance_remaining_usd": 32.68,
        "cache_savings_usd": 1,
        "upstream_request_cost_usd": "0.0000104",
        "request_cost_usd": 0.000013,
        "markup_pct": 25.0,
    }


def test_neuralwatt_markup_pct_calculation_uses_upstream_cost_once():
    assert _sanitize_cost({"request_cost_usd": "0.01"}, Decimal("0"))["request_cost_usd"] == 0.01
    assert _sanitize_cost({"request_cost_usd": "0.01"}, Decimal("25"))["request_cost_usd"] == 0.0125
    assert _sanitize_cost({"request_cost_usd": "0.01"}, Decimal("100"))["request_cost_usd"] == 0.02
    assert _sanitize_cost({"request_cost_usd": "0.01"}, Decimal("12.5"))["request_cost_usd"] == 0.01125


def test_neuralwatt_applies_usage_cost_usd():
    response_json = {
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "cost": {"request_cost_usd": "0.01", "allowance_remaining_usd": 9},
    }

    response_cost = _apply_neuralwatt_metadata(response_json, Decimal("10"))

    assert response_cost == 0.011
    assert response_json["usage"]["cost_usd"] == 0.011
    assert response_json["usage"]["cost"] == 0.011
    assert response_json["cost"] == {"request_cost_usd": 0.011}


def test_neuralwatt_requires_upstream_request_cost():
    with pytest.raises(CustomLLMError):
        _sanitize_cost({}, Decimal("0"))


def test_neuralwatt_rate_limit_error_sets_retry_after_header():
    response = MockErrorResponse(
        429,
        {
            "error": {
                "type": "rate_limit_error",
                "code": "tpm_uncached_exceeded",
                "message": "Uncached token rate limit exceeded for glm-5.2.",
                "retry_after": 2,
                "retryable": True,
                "retry_strategy": {"type": "tpm_uncached"},
                "context": {"model": "glm-5.2", "limit_type": "tpm_uncached"},
            }
        },
    )

    with pytest.raises(CustomLLMError) as exc_info:
        _raise_for_status(response)

    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.message == "Uncached token rate limit exceeded for glm-5.2."
    assert exc.headers["retry-after"] == "2"
    assert exc.response.headers["retry-after"] == "2"
    assert exc.code == "tpm_uncached_exceeded"
    assert exc.retryable is True
    assert exc.retry_strategy == {"type": "tpm_uncached"}
    assert exc.context == {"model": "glm-5.2", "limit_type": "tpm_uncached"}


def test_neuralwatt_503_maps_to_service_unavailable():
    response = MockErrorResponse(
        503,
        {
            "error": {
                "type": "service_unavailable",
                "code": "fleet_capacity_exceeded",
                "message": "Service temporarily at capacity.",
                "retry_after": 5,
                "retryable": True,
            }
        },
    )

    with pytest.raises(CustomLLMError) as exc_info:
        _raise_for_status(response)

    with pytest.raises(litellm.ServiceUnavailableError) as mapped_exc_info:
        exception_type(
            model="neuralwatt/glm-5.2",
            original_exception=exc_info.value,
            custom_llm_provider="neuralwatt",
        )

    assert mapped_exc_info.value.response.headers["retry-after"] == "5"


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (400, litellm.BadRequestError),
        (403, litellm.PermissionDeniedError),
        (404, litellm.NotFoundError),
        (422, litellm.BadRequestError),
        (429, litellm.RateLimitError),
        (500, litellm.InternalServerError),
    ],
)
def test_neuralwatt_errors_map_by_status_without_naming_the_provider(status_code, expected):
    response = MockErrorResponse(
        status_code,
        {"error": {"message": "NeuralWatt rejected the call at api.neuralwatt.com"}},
    )
    with pytest.raises(CustomLLMError) as exc_info:
        _raise_for_status(response)

    with pytest.raises(expected) as mapped_exc_info:
        exception_type(
            model="neuralwatt/glm-5.2",
            original_exception=exc_info.value,
            custom_llm_provider="neuralwatt",
        )

    assert "neuralwatt" not in str(mapped_exc_info.value).lower()


def test_neuralwatt_private_financial_headers_are_not_forwarded():
    response = MockErrorResponse(
        429,
        {"error": {"message": "rate limited", "retry_after": 2}},
        headers={
            "X-Request-Cost-USD": "0.003400",
            "X-Cache-Savings-USD": "0.027000",
            "X-Allowance-Remaining-USD": "47.66",
            "X-Session-Spent-USD": "1.230000",
            "X-Session-Allowance-Remaining-USD": "3.770000",
            "X-Provider-Trace-Id": "trace-123",
        },
    )

    headers = _response_headers_with_retry_after(response, 2)

    assert headers["retry-after"] == "2"
    assert headers["X-Provider-Trace-Id"] == "trace-123"
    assert "X-Request-Cost-USD" not in headers
    assert "X-Cache-Savings-USD" not in headers
    assert "X-Allowance-Remaining-USD" not in headers
    assert "X-Session-Spent-USD" not in headers
    assert "X-Session-Allowance-Remaining-USD" not in headers


def test_neuralwatt_energy_headers_omit_missing_values():
    headers = _energy_headers(
        {
            "energy_joules": 4.99,
            "energy_kwh": 0.000001385,
            "duration_seconds": 0.361,
        }
    )

    assert headers == {
        "X-Energy-Joules": "4.99",
        "X-Energy-Kwh": "1.385e-06",
        "X-Energy-Duration-Seconds": "0.361",
    }


def test_neuralwatt_stream_usage_chunk_gets_cost():
    payload = {
        "choices": [],
        "usage": {"prompt_tokens": 18, "completion_tokens": 50, "total_tokens": 68, "cost_usd": 0.013},
    }

    chunk = _chunk_from_payload(payload)

    assert chunk["usage"] == payload["usage"]
    assert chunk["text"] == ""


def test_neuralwatt_stream_preserves_tool_call_delta():
    payload = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }

    chunk = _chunk_from_payload(payload)

    assert chunk["tool_use"] == payload["choices"][0]["delta"]["tool_calls"][0]


def test_neuralwatt_stream_metadata_serializes_comments():
    energy = {"energy_joules": 4.99}
    cost = {"upstream_request_cost_usd": 0.01, "request_cost_usd": 0.012, "markup_pct": 20.0}
    comments = _final_metadata_chunks(energy=energy, cost=cost)

    serialized = []
    for item in comments:
        stream = ModelResponseStream()
        stream.choices = [StreamingChoices(delta=Delta(content=""))]
        stream.choices[0].delta.provider_specific_fields = item["provider_specific_fields"]
        serialized.append(_serialize_streaming_chunk(stream))

    assert len(serialized) == 2
    assert serialized[0] == ': energy {"energy_joules": 4.99}\n\n'
    assert serialized[1] == ': cost {"request_cost_usd": 0.012}\n\n'


class _NeuralWattSSELogging:
    def __init__(self):
        self.model_call_details = {}
        self._llm_caching_handler = None
        self.completion_start_time = None

    def pre_call(self, *args, **kwargs):
        pass

    def post_call(self, *args, **kwargs):
        pass

    def _update_completion_start_time(self, completion_start_time):
        self.completion_start_time = completion_start_time

    def failure_handler(self, *args, **kwargs):
        pass

    async def async_success_handler(self, *args, **kwargs):
        pass


class _NeuralWattSSEClient:
    def __init__(self, lines):
        self._lines = lines

    async def post(self, **kwargs):
        return self

    @property
    def status_code(self):
        return 200

    @property
    def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def test_neuralwatt_stream_no_double_sse_prefix():
    """
    NeuralWatt emits SSE comments after the normal OpenAI stream. Those lines
    must pass through the proxy unchanged; wrapping them with an extra `data:`
    prefix makes OpenAI clients try to parse
    `: energy {...}` as JSON and fail.
    """
    lines = [
        'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"glm-5.2","choices":[{"index":0,"delta":{"content":"Hello!"}}]}',
        'data: {"id":"c2","object":"chat.completion.chunk","created":1,"model":"glm-5.2","choices":[{"index":0,"delta":{"content":" How can I assist you today?"}}]}',
        ': energy {"energy_joules": 4.99}',
        ': cost {"request_cost_usd": "0.01", "allowance_remaining_usd": 47.66, "cache_savings_usd": 0.027, "session_spent_usd": 1.23}',
        'data: {"id":"c3","object":"chat.completion.chunk","created":1,"model":"glm-5.2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":9,"total_tokens":10}}',
    ]

    logging = _NeuralWattSSELogging()
    stream = _async_stream(
        url="https://api.neuralwatt.test/v1/chat/completions",
        headers={},
        request_data={"model": "glm-5.2", "stream": True},
        timeout=None,
        client=_NeuralWattSSEClient(lines),
        markup_pct=Decimal("0"),
        logging_obj=logging,
        emit_final_metadata=False,
    )
    wrapper = CustomStreamWrapper(
        completion_stream=stream,
        model="neuralwatt/glm-5.2",
        custom_llm_provider="neuralwatt",
        logging_obj=logging,
    )

    async def collect():
        out = []
        async for line in async_data_generator(wrapper, UserAPIKeyAuth(), {"litellm_call_id": "abc"}):
            out.append(line)
        return out

    emitted = asyncio.run(collect())
    emitted_str = "".join(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk for chunk in emitted)

    assert ": energy" not in emitted_str
    assert ": cost" not in emitted_str
    assert "data: : energy" not in emitted_str
    assert "data: event: neuralwatt" not in emitted_str
    assert "event: neuralwatt" not in emitted_str
    assert "data: : cost" not in emitted_str
    assert "allowance_remaining_usd" not in emitted_str
    assert "cache_savings_usd" not in emitted_str
    assert "session_spent_usd" not in emitted_str
    assert wrapper.logging_obj.model_call_details["litellm_params"]["metadata"]["spend_logs_metadata"] == {
        "neuralwatt_energy": {"energy_joules": 4.99},
        "neuralwatt_cost": {
            "allowance_remaining_usd": 47.66,
            "cache_savings_usd": 0.027,
            "session_spent_usd": 1.23,
            "upstream_request_cost_usd": "0.01",
            "request_cost_usd": 0.01,
            "markup_pct": 0.0,
        },
    }
    assert wrapper.logging_obj.model_call_details["response_cost"] == 0.01


def test_neuralwatt_stream_usage_gets_marked_up_cost_fields():
    lines = [
        ': cost {"request_cost_usd": "0.01"}',
        'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":9,"total_tokens":10}}',
    ]

    async def collect():
        chunks = []
        async for chunk in _async_stream(
            url="https://api.neuralwatt.test/v1/chat/completions",
            headers={},
            request_data={"model": "glm-5.2", "stream": True},
            timeout=None,
            client=_NeuralWattSSEClient(lines),
            markup_pct=Decimal("25"),
            logging_obj=_NeuralWattSSELogging(),
            emit_final_metadata=False,
        ):
            chunks.append(chunk)
        return chunks

    usage_chunk = next(chunk for chunk in asyncio.run(collect()) if chunk.get("usage"))

    assert usage_chunk["usage"]["cost_usd"] == 0.0125
    assert usage_chunk["usage"]["cost"] == 0.0125
