from decimal import Decimal
import json

import pytest

from litellm import ModelResponse
from litellm.llms.custom_llm import CustomLLMError
from litellm.llms.neuralwatt.chat.handler import (
    NeuralWattChatCompletion,
    _apply_neuralwatt_metadata,
    _chunk_from_payload,
    _energy_headers,
    _final_metadata_chunks,
    _get_chat_completions_url,
    _get_markup_pct,
    _get_upstream_model,
    _sanitize_cost,
)
from litellm.proxy.proxy_server import _serialize_streaming_chunk
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


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
