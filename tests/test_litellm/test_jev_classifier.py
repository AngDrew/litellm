"""The Jev classifier plugin: request shape, verdict mapping, and the decisions call. Offline."""

from __future__ import annotations

import json

import httpx
import pytest

from callbacks.jev_classifier import JevClassifier, build_payload, tier_from_response
from litellm.types.router import RoutingContext

MESSAGES = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "hi"},
    {"role": "tool", "content": "200 pages of build output"},
    {"role": "user", "content": "Prove the halting problem is undecidable."},
]


def test_payload_asks_one_choice_question_over_the_router_tiers():
    payload = build_payload(MESSAGES)
    question = payload["questions"]["complexity"]
    assert question["type"] == "choice"
    assert sorted(question["criteria"]) == ["COMPLEX", "MEDIUM", "REASONING", "SIMPLE"]
    assert payload["state"]["system_prompt"] == "You are a coding agent."
    assert [turn["text"] for turn in payload["state"]["conversation"]] == [
        "hi",
        "Prove the halting problem is undecidable.",
    ]


def test_verdict_maps_to_the_tier_and_declines_when_unsure():
    answered = {"answers": {"complexity": {"type": "choice", "choice": "REASONING", "confidence": 0.9}}}
    assert tier_from_response(answered, 0.5) == "REASONING"
    unsure = {"answers": {"complexity": {"type": "choice", "choice": "REASONING", "confidence": 0.2}}}
    assert tier_from_response(unsure, 0.5) is None
    assert tier_from_response({"answers": {}}, 0.5) is None
    assert tier_from_response({}, 0.5) is None


async def test_classify_posts_the_decisions_body_and_returns_the_tier(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OR_API_KEY", "test-key")
    monkeypatch.setenv("JEV_DECISIONS_URL", "https://example.test/api/alpha/decisions")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"answers": {"complexity": {"type": "choice", "choice": "COMPLEX", "confidence": 0.8}}}
        )

    classifier = JevClassifier(transport=httpx.MockTransport(handler))
    context = RoutingContext(raw_messages=[], structured_messages=MESSAGES, candidate_models=[])

    assert await classifier.classify(context) == "COMPLEX"
    assert seen["url"] == "https://example.test/api/alpha/decisions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "typesafe/jev-1.13"
    assert seen["body"]["state"]["conversation"][-1]["text"] == "Prove the halting problem is undecidable."
