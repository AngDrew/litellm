"""The Jev classifier plugin: request shape, verdict mapping, and the metered router call. Offline."""

from __future__ import annotations

import json

import pytest

from callbacks.jev_classifier import JevClassifier, build_payload, tier_from_response
from litellm.types.router import RoutingContext
from litellm.types.utils import ModelResponse

MESSAGES = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "hi"},
    {"role": "tool", "content": "200 pages of build output"},
    {"role": "assistant", "content": "I will read the proof again."},
    {"role": "user", "content": "Prove the halting problem is undecidable."},
]

ANSWERS = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "complexity": {"type": "choice", "choice": "COMPLEX", "probabilities": {"COMPLEX": 0.8}, "confidence": 0.8}
    },
    "usage": {"input_tokens": 427, "output_tokens": 73, "cost": 0.000017934},
}


class FakeRouter:
    """Stands in for the proxy router, recording what the classifier asks it for."""

    def __init__(self, answers: dict | None = None) -> None:
        self.calls: list[dict] = []
        self._answers = answers or ANSWERS

    async def acompletion(self, **kwargs) -> ModelResponse:
        self.calls.append(kwargs)
        return ModelResponse(
            model=kwargs["model"],
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(self._answers)},
                    "finish_reason": "stop",
                }
            ],
        )


def test_payload_asks_one_choice_question_over_the_router_tiers():
    payload = build_payload(MESSAGES)
    question = payload["questions"]["complexity"]
    assert question["type"] == "choice"
    assert sorted(question["criteria"]) == ["COMPLEX", "MEDIUM", "REASONING", "SIMPLE"]
    assert payload["state"]["system_prompt"] == "You are a coding agent."
    # tool output and assistant narration are dropped, the ask is the newest user turn
    assert [turn["text"] for turn in payload["state"]["conversation"]] == [
        "hi",
        "Prove the halting problem is undecidable.",
    ]


def test_payload_caps_the_system_prompt_an_agent_sends():
    payload = build_payload([{"role": "system", "content": "x" * 50_000}, MESSAGES[-1]])
    assert len(payload["state"]["system_prompt"]) == 4000


def test_verdict_maps_to_the_tier_and_declines_when_unsure():
    answered = {"answers": {"complexity": {"type": "choice", "choice": "REASONING", "confidence": 0.9}}}
    assert tier_from_response(answered, 0.5) == "REASONING"
    unsure = {"answers": {"complexity": {"type": "choice", "choice": "REASONING", "confidence": 0.2}}}
    assert tier_from_response(unsure, 0.5) is None
    assert tier_from_response({"answers": {}}, 0.5) is None
    assert tier_from_response({}, 0.5) is None


async def test_classify_asks_the_router_so_the_call_is_metered(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("JEV_MODEL", raising=False)
    router = FakeRouter()
    classifier = JevClassifier(router=router)
    context = RoutingContext(
        raw_messages=[], structured_messages=MESSAGES, candidate_models=[], metadata={"user_api_key": "hashed-key"}
    )

    assert await classifier.classify(context) == "COMPLEX"

    call = router.calls[0]
    assert call["model"] == "jev-1.13"  # the priced deployment, not the raw upstream id
    assert call["timeout"] == 10.0
    body = json.loads(call["messages"][0]["content"])
    assert set(body) == {"state", "questions"}  # the provider supplies the model it posts for
    assert body["state"]["conversation"][-1]["text"] == "Prove the halting problem is undecidable."
    # the sub-call carries the caller's identity so its spend is attributed to them
    assert call["metadata"]["user_api_key"] == "hashed-key"
    assert call["metadata"]["internal_call_origin"] == "autorouter_classifier"


async def test_classify_declines_on_low_confidence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JEV_MIN_CONFIDENCE", "0.9")
    unsure = {"model": "typesafe/jev-1.13", "answers": {"complexity": {"choice": "MEDIUM", "confidence": 0.5}}}
    classifier = JevClassifier(router=FakeRouter(unsure))
    context = RoutingContext(raw_messages=[], structured_messages=MESSAGES, candidate_models=[], metadata={})

    assert await classifier.classify(context) is None
