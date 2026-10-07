"""
Tests for SystemOneAdapterMiddleware.

`POST /v1/systemone` is TypeSafe's System One protocol; Jev is served here as a chat deployment whose
single message is the same JSON and whose reply carries the answers body as the message content.
The middleware rewrites the request into that chat call before routing and unwraps a successful
reply, so everything that meters a chat call (auth, account holds, margins, spend log) sees one.
"""

import json
from collections.abc import Mapping
from typing import Any, Final

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from litellm.proxy.middleware.system_one_adapter_middleware import SystemOneAdapterMiddleware

ANSWERS: Final = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "kind": {"type": "choice", "choice": "bug", "probabilities": {"bug": 1, "feature": 0}, "confidence": 1}
    },
    "usage": {"input_tokens": 308, "output_tokens": 31, "cost": 1.2936e-05},
}
REQUEST: Final = {
    "model": "jev-1.13",
    "state": "login crashes with a null pointer",
    "questions": {
        "kind": {"type": "choice", "instructions": "What kind?", "criteria": {"bug": "defect", "feature": "ask"}},
        "urgent": {"type": "noul", "instructions": "Urgent?"},
    },
}


def _completion(content: str) -> dict[str, Any]:
    return {
        "id": "gen-dec-1",
        "object": "chat.completion",
        "model": "jev-1.13",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 308, "completion_tokens": 31, "total_tokens": 339},
    }


# Jev reports the provider's raw charge (ANSWERS["usage"]["cost"]); the proxy stamps what the account
# was actually charged, margin included, on the reply. Clients must see the second figure.
BILLED_COST: Final = 1.5e-05
BILLED_HEADERS: Final = {"x-litellm-response-cost": repr(BILLED_COST)}
EXPECTED: Final = {**ANSWERS, "usage": {**ANSWERS["usage"], "cost": BILLED_COST}}


def _completion_response(answers: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> JSONResponse:
    return JSONResponse(_completion(json.dumps(answers)), headers=dict(BILLED_HEADERS if headers is None else headers))


def _client(reply: Response | None = None) -> tuple[TestClient, list[dict[str, Any]]]:
    """An app that records what reached it, standing in for the chat endpoint and everything above it."""
    seen: Final[list[dict[str, Any]]] = []  # mutable-ok: test recorder

    async def chat(request: Request) -> Response:
        seen.append(
            {
                "path": request.url.path,
                "body": json.loads(await request.body()),
                "content_length": request.headers.get("content-length"),
                "content_type": request.headers.get("content-type"),
                "accept_encoding": request.headers.get("accept-encoding"),
                "authorization": request.headers.get("authorization"),
            }
        )
        return reply if reply is not None else _completion_response(ANSWERS)

    async def other(request: Request) -> Response:
        seen.append({"path": request.url.path, "method": request.method, "body": (await request.body()).decode()})
        return JSONResponse({"untouched": True})

    app: Final = Starlette(
        routes=[
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/v1/systemone", other, methods=["GET"]),
            Route("/v1/models", other, methods=["POST", "GET"]),
        ]
    )
    app.add_middleware(SystemOneAdapterMiddleware)
    return TestClient(app), seen


def test_system_one_becomes_a_jev_chat_completion() -> None:
    client, seen = _client()

    response: Final = client.post("/v1/systemone", json=REQUEST, headers={"Authorization": "Bearer sk-test"})

    assert response.status_code == 200
    assert len(seen) == 1
    forwarded: Final = seen[0]
    assert forwarded["path"] == "/v1/chat/completions"
    assert forwarded["authorization"] == "Bearer sk-test"
    assert forwarded["content_type"] == "application/json"
    assert forwarded["accept_encoding"] is None
    body: Final = forwarded["body"]
    assert set(body) == {"model", "messages"}
    assert body["model"] == "jev-1.13"
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    # exactly what callbacks.jev_provider.decisions_request expects, minus the model
    assert json.loads(body["messages"][0]["content"]) == {
        "state": REQUEST["state"],
        "questions": REQUEST["questions"],
    }


def test_content_length_matches_the_rewritten_body() -> None:
    client, seen = _client()

    client.post("/v1/systemone", json=REQUEST)

    assert int(seen[0]["content_length"]) == len(json.dumps(seen[0]["body"], separators=(",", ":")))


def test_reply_is_the_unwrapped_answers_body() -> None:
    client, _ = _client()

    response: Final = client.post("/v1/systemone", json=REQUEST)

    assert response.headers["content-type"] == "application/json"
    assert response.json() == EXPECTED
    assert int(response.headers["content-length"]) == len(response.content)


@pytest.mark.parametrize("path", ["/v1/systemone", "/systemone"])
def test_both_paths_are_served(path: str) -> None:
    client, seen = _client()

    response: Final = client.post(path, json=REQUEST)

    assert response.json() == EXPECTED
    assert seen[0]["path"] == "/v1/chat/completions"


def test_extra_system_one_fields_ride_along_to_the_decisions_endpoint() -> None:
    client, seen = _client()

    client.post("/v1/systemone", json={**REQUEST, "extra": {"a": 1}})

    assert json.loads(seen[0]["body"]["messages"][0]["content"])["extra"] == {"a": 1}


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        (b"not json", "must be JSON"),
        (b"[1]", "JSON object"),
        (json.dumps({**REQUEST, "model": "gpt-5.6-sol"}).encode(), "decisions model"),
        (json.dumps({k: v for k, v in REQUEST.items() if k != "model"}).encode(), "decisions model"),
        (json.dumps({**REQUEST, "state": 3}).encode(), "`state`"),
        (json.dumps({**REQUEST, "questions": {}}).encode(), "`questions`"),
        (json.dumps({**REQUEST, "questions": ["a"]}).encode(), "`questions`"),
    ],
)
def test_malformed_requests_are_refused_before_anything_is_billed(payload: bytes, needle: str) -> None:
    client, seen = _client()

    response: Final = client.post("/v1/systemone", content=payload, headers={"content-type": "application/json"})

    assert response.status_code == 400
    assert needle in response.json()["error"]["message"]
    assert response.json()["error"]["code"] == "400"
    assert seen == []


def test_oversized_bodies_are_refused() -> None:
    client, seen = _client()

    response: Final = client.post("/v1/systemone", json={**REQUEST, "state": "x" * (1024 * 1024 + 1)})

    assert response.status_code == 413
    assert seen == []


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_errors_pass_through_untouched(status: int) -> None:
    error: Final = {"error": {"message": "nope", "type": "auth_error", "param": None, "code": str(status)}}
    client, _ = _client(JSONResponse(error, status_code=status))

    response: Final = client.post("/v1/systemone", json=REQUEST)

    assert response.status_code == status
    assert response.json() == error


@pytest.mark.parametrize(
    "content",
    ["plain prose", json.dumps(["not", "an", "object"]), json.dumps({"no": "answers"}), json.dumps({"answers": 3})],
)
def test_an_unreadable_answer_is_a_bad_gateway(content: str) -> None:
    client, _ = _client(JSONResponse(_completion(content)))

    response: Final = client.post("/v1/systemone", json=REQUEST)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"


def test_the_client_sees_the_billed_cost_not_the_providers() -> None:
    client, _ = _client()

    usage: Final = client.post("/v1/systemone", json=REQUEST).json()["usage"]

    assert usage["cost"] == BILLED_COST
    assert usage["cost"] != ANSWERS["usage"]["cost"]
    assert (usage["input_tokens"], usage["output_tokens"]) == (308, 31)


def test_a_zero_cost_reply_is_reported_as_free() -> None:
    client, _ = _client(_completion_response(ANSWERS, {"x-litellm-response-cost": "0.0"}))

    assert client.post("/v1/systemone", json=REQUEST).json()["usage"]["cost"] == 0.0


def test_a_response_cache_hit_is_reported_as_free_even_though_the_header_keeps_the_original_price() -> None:
    # the cached reply still carries the price of the call that filled the cache; serving it is not charged
    headers: Final = {**BILLED_HEADERS, "x-litellm-cache-key": "bd2fa04f9372"}
    client, _ = _client(_completion_response(ANSWERS, headers))

    usage: Final = client.post("/v1/systemone", json=REQUEST).json()["usage"]

    assert usage["cost"] == 0.0
    assert (usage["input_tokens"], usage["output_tokens"]) == (308, 31)


def test_a_cache_hit_is_free_even_when_the_cost_header_is_missing() -> None:
    client, _ = _client(_completion_response(ANSWERS, {"x-litellm-cache-key": "bd2fa04f9372"}))

    assert client.post("/v1/systemone", json=REQUEST).json()["usage"]["cost"] == 0.0


def test_an_empty_cache_key_header_is_not_a_cache_hit() -> None:
    client, _ = _client(_completion_response(ANSWERS, {**BILLED_HEADERS, "x-litellm-cache-key": ""}))

    assert client.post("/v1/systemone", json=REQUEST).json()["usage"]["cost"] == BILLED_COST


@pytest.mark.parametrize("header", [None, "", "abc", "nan", "inf", "-1e-05"])
def test_cost_is_dropped_rather_than_showing_the_raw_price_when_billing_is_unknown(header: str | None) -> None:
    headers: Final = {} if header is None else {"x-litellm-response-cost": header}
    client, _ = _client(_completion_response(ANSWERS, headers))

    body: Final = client.post("/v1/systemone", json=REQUEST).json()

    assert "cost" not in body["usage"]
    assert body["usage"]["input_tokens"] == 308
    assert body["answers"] == ANSWERS["answers"]


def test_an_answer_without_usage_is_returned_as_it_is() -> None:
    answers: Final = {k: v for k, v in ANSWERS.items() if k != "usage"}
    client, _ = _client(_completion_response(answers))

    assert client.post("/v1/systemone", json=REQUEST).json() == answers


def test_a_reply_that_is_not_a_completion_is_a_bad_gateway() -> None:
    client, _ = _client(JSONResponse({"choices": []}))

    assert client.post("/v1/systemone", json=REQUEST).status_code == 502


def test_other_routes_and_methods_are_left_alone() -> None:
    client, seen = _client()

    get_response: Final = client.get("/v1/systemone")
    post_response: Final = client.post("/v1/models", content=b"raw")

    assert get_response.json() == {"untouched": True}
    assert post_response.json() == {"untouched": True}
    assert seen == [
        {"path": "/v1/systemone", "method": "GET", "body": ""},
        {"path": "/v1/models", "method": "POST", "body": "raw"},
    ]


@pytest.mark.asyncio
async def test_non_http_scopes_are_forwarded() -> None:
    forwarded: Final[list[Mapping[str, Any]]] = []  # mutable-ok: test recorder

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        forwarded.append(scope)

    async def receive() -> Message:
        return {"type": "lifespan.startup"}

    async def send(message: Message) -> None:
        raise AssertionError(message)

    await SystemOneAdapterMiddleware(inner)({"type": "lifespan"}, receive, send)

    assert forwarded == [{"type": "lifespan"}]


@pytest.mark.asyncio
async def test_a_chunked_body_is_reassembled() -> None:
    body: Final = json.dumps(REQUEST).encode()
    messages: Final[list[Message]] = [  # mutable-ok: consumed by the fake receive
        {"type": "http.request", "body": body[:20], "more_body": True},
        {"type": "http.request", "body": body[20:], "more_body": False},
    ]
    reached: Final[list[Any]] = []  # mutable-ok: test recorder
    sent: Final[list[Message]] = []  # mutable-ok: test recorder

    async def receive() -> Message:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    async def inner(scope: Scope, inner_receive: Receive, inner_send: Send) -> None:
        first: Final = await inner_receive()
        reached.append((scope["path"], scope["raw_path"], json.loads(first["body"])["model"], first["more_body"]))
        assert (await inner_receive())["type"] == "http.disconnect"
        content: Final = json.dumps(_completion(json.dumps(ANSWERS))).encode()
        await inner_send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"x-keep", b"1"), (b"x-litellm-response-cost", b"1.5e-05")],
            }
        )
        await inner_send({"type": "http.response.body", "body": content[:10], "more_body": True})
        await inner_send({"type": "http.response.body", "body": content[10:], "more_body": False})

    scope: Final = {"type": "http", "method": "POST", "path": "/v1/systemone", "headers": []}
    await SystemOneAdapterMiddleware(inner)(scope, receive, send)

    assert reached == [("/v1/chat/completions", b"/v1/chat/completions", "jev-1.13", False)]
    assert sent[0]["status"] == 200
    assert (b"x-keep", b"1") in sent[0]["headers"]
    assert json.loads(sent[1]["body"]) == EXPECTED
    assert sent[1]["more_body"] is False
    assert len(sent) == 2
