"""The decisions provider: transport contract, response mapping, and recorded cost. Offline."""

from __future__ import annotations

import json

import httpx
import pytest

from callbacks.jev_provider import TypeSafeDecisions, decisions_request, decisions_url
from litellm.llms.custom_llm import CustomLLMError
from litellm.types.utils import ModelResponse

MODEL = "typesafe/jev-1.13"
DECISIONS_BODY = {
    "state": {"conversation": [{"role": "user", "text": "hi"}]},
    "questions": {"complexity": {"type": "choice"}},
}
ANSWER = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {"complexity": {"type": "choice", "choice": "SIMPLE", "confidence": 0.99}},
    "usage": {"input_tokens": 427, "output_tokens": 73, "cost": 0.000017934},
    "id": "gen-dec-1",
}


class FakeLogging:
    def __init__(self) -> None:
        self.model_call_details: dict = {}


def _call(
    handler: TypeSafeDecisions, *, api_key: str | None = "test-key", messages: list | None = None, logging_obj=None
):
    return handler.acompletion(
        model=MODEL,
        messages=messages or [{"role": "user", "content": json.dumps(DECISIONS_BODY)}],
        api_base="https://openrouter.ai/api/v1",
        custom_prompt_dict={},
        model_response=ModelResponse(),
        print_verbose=lambda *args: None,
        encoding=None,
        api_key=api_key,
        logging_obj=logging_obj or FakeLogging(),
        optional_params={},
    )


def test_decisions_url_moves_off_the_v1_base_and_honors_an_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("JEV_DECISIONS_URL", raising=False)
    assert decisions_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/alpha/decisions"
    monkeypatch.setenv("JEV_DECISIONS_URL", "https://example.test/decisions")
    assert decisions_url("https://openrouter.ai/api/v1") == "https://example.test/decisions"


def test_decisions_request_carries_the_model_and_rejects_a_conversation():
    assert decisions_request([{"role": "user", "content": json.dumps(DECISIONS_BODY)}], MODEL) == {
        **DECISIONS_BODY,
        "model": MODEL,
    }
    with pytest.raises(CustomLLMError):
        decisions_request([{"role": "user", "content": "just a chat message"}], MODEL)
    with pytest.raises(CustomLLMError):
        decisions_request([{"role": "user", "content": "{}"}], MODEL)
    with pytest.raises(CustomLLMError):
        decisions_request([{"role": "user", "content": "{}"}, {"role": "user", "content": "{}"}], MODEL)


async def test_acompletion_records_openrouters_exact_cost(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("JEV_MIN_CONFIDENCE", raising=False)
    seen: dict = {}

    def respond(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ANSWER)

    logging_obj = FakeLogging()
    response = await _call(TypeSafeDecisions(transport=httpx.MockTransport(respond)), logging_obj=logging_obj)

    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"] == {**DECISIONS_BODY, "model": MODEL}
    # the decisions body comes back whole, so the classifier reads typed answers out of it
    assert json.loads(response.choices[0].message.content)["answers"]["complexity"]["choice"] == "SIMPLE"
    assert response.usage.prompt_tokens == 427
    assert response.usage.completion_tokens == 73
    # this is the number the spend log and the tier budgets read
    assert logging_obj.model_call_details["response_cost"] == 0.000017934
    assert response._hidden_params["response_cost"] == 0.000017934


async def test_acompletion_falls_back_to_the_deployment_price_when_cost_is_absent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("JEV_MIN_CONFIDENCE", raising=False)
    unpriced = {"answers": ANSWER["answers"], "usage": {"input_tokens": 1000, "output_tokens": 0}}
    logging_obj = FakeLogging()
    handler = TypeSafeDecisions(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=unpriced)))

    response = await _call(handler, logging_obj=logging_obj)

    # no cost reported: LiteLLM prices it from the deployment's own per-token rates instead
    assert "response_cost" not in logging_obj.model_call_details
    assert response.usage.prompt_tokens == 1000


async def test_acompletion_requires_a_key_and_reports_upstream_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OR_API_KEY", raising=False)
    with pytest.raises(CustomLLMError):
        await _call(
            TypeSafeDecisions(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=ANSWER))),
            api_key=None,
        )

    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "is a decisions model"}})

    with pytest.raises(CustomLLMError):
        await _call(TypeSafeDecisions(transport=httpx.MockTransport(reject)))
