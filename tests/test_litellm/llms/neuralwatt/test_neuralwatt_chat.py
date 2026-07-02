import asyncio
from decimal import Decimal
import json

import pytest

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
    _get_upstream_model,
    _sanitize_cost,
)
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
    }
    assert response.usage.cost_usd == 0.012
    assert response.cost == {
        "upstream_request_cost_usd": "0.01",
        "request_cost_usd": 0.012,
        "markup_pct": 20.0,
    }
    assert response._hidden_params["response_cost"] == 0.012
    assert response._hidden_params["additional_headers"] == {
        "X-Energy-Joules": "4.99",
        "X-Energy-Kwh": "1.385e-06",
        "X-Energy-Avg-Power-Watts": "55.3",
        "X-Energy-Duration-Seconds": "0.361",
    }


def test_neuralwatt_requires_api_base():
    with pytest.raises(CustomLLMError):
        _get_chat_completions_url(None)


def test_neuralwatt_validates_markup_pct():
    assert _get_markup_pct({}) == Decimal("0")
    assert _get_markup_pct({"markup_pct": "25"}) == Decimal("25")

    with pytest.raises(CustomLLMError):
        _get_markup_pct({"markup_pct": -1})

    with pytest.raises(CustomLLMError):
        _get_markup_pct({"markup_pct": "bad"})


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
        "upstream_request_cost_usd": "0.0000104",
        "request_cost_usd": 0.000013,
        "markup_pct": 25.0,
    }


def test_neuralwatt_applies_usage_cost_usd():
    response_json = {
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "cost": {"request_cost_usd": "0.01", "allowance_remaining_usd": 9},
    }

    response_cost = _apply_neuralwatt_metadata(response_json, Decimal("10"))

    assert response_cost == 0.011
    assert response_json["usage"]["cost_usd"] == 0.011
    assert response_json["cost"] == {
        "upstream_request_cost_usd": "0.01",
        "request_cost_usd": 0.011,
        "markup_pct": 10.0,
    }


def test_neuralwatt_requires_upstream_request_cost():
    with pytest.raises(CustomLLMError):
        _sanitize_cost({}, Decimal("0"))


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


def test_neuralwatt_stream_metadata_serializes_comments_and_event():
    energy = {"energy_joules": 4.99}
    cost = {"upstream_request_cost_usd": 0.01, "request_cost_usd": 0.012, "markup_pct": 20.0}
    comments_and_event = _final_metadata_chunks(energy=energy, cost=cost)

    serialized = []
    for item in comments_and_event:
        stream = ModelResponseStream()
        stream.choices = [StreamingChoices(delta=Delta(content=""))]
        stream.choices[0].delta.provider_specific_fields = item["provider_specific_fields"]
        serialized.append(_serialize_streaming_chunk(stream))

    assert serialized[0] == ': energy {"energy_joules": 4.99}\n\n'
    assert serialized[1].startswith(': cost {"upstream_request_cost_usd": 0.01')
    assert serialized[2].startswith("event: neuralwatt\ndata: ")


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
    NeuralWatt emits SSE comments and a custom event after the normal OpenAI
    stream. Those lines must pass through the proxy unchanged; wrapping them
    with an extra `data:` prefix makes OpenAI clients try to parse
    `: energy {...}` as JSON and fail.
    """
    lines = [
        'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"glm-5.2","choices":[{"index":0,"delta":{"content":"Hello!"}}]}',
        'data: {"id":"c2","object":"chat.completion.chunk","created":1,"model":"glm-5.2","choices":[{"index":0,"delta":{"content":" How can I assist you today?"}}]}',
        ': energy {"energy_joules": 4.99}',
        ': cost {"request_cost_usd": "0.01"}',
        'data: {"id":"c3","object":"chat.completion.chunk","created":1,"model":"glm-5.2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":9,"total_tokens":10}}',
    ]

    stream = _async_stream(
        url="https://api.neuralwatt.test/v1/chat/completions",
        headers={},
        request_data={"model": "glm-5.2", "stream": True},
        timeout=None,
        client=_NeuralWattSSEClient(lines),
        markup_pct=Decimal("0"),
        logging_obj=_NeuralWattSSELogging(),
    )
    wrapper = CustomStreamWrapper(
        completion_stream=stream,
        model="neuralwatt/glm-5.2",
        custom_llm_provider="neuralwatt",
        logging_obj=_NeuralWattSSELogging(),
    )

    async def collect():
        out = []
        async for line in async_data_generator(
            wrapper, UserAPIKeyAuth(), {"litellm_call_id": "abc"}
        ):
            out.append(line)
        return out

    emitted = asyncio.run(collect())
    emitted_str = "".join(
        chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        for chunk in emitted
    )

    assert ": energy" in emitted_str
    assert "event: neuralwatt" in emitted_str
    assert "data: : energy" not in emitted_str
    assert "data: event: neuralwatt" not in emitted_str
    assert "data: : cost" not in emitted_str
