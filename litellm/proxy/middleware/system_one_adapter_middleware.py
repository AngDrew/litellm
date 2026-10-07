import json
import math
from collections.abc import Mapping
from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

SYSTEM_ONE_PATHS: Final = frozenset({"/v1/systemone", "/systemone"})
CHAT_COMPLETIONS_PATH: Final = "/v1/chat/completions"
# Only decisions models can answer a System One request. Anything else would be billed as an
# ordinary chat call and come back as prose the client cannot read, so it is refused up front.
SYSTEM_ONE_MODEL_PREFIX: Final = "jev-"
# A System One body is a state string plus a handful of questions. Jev's context is 32k tokens,
# so a megabyte is already far past anything the model could read.
MAX_BODY_BYTES: Final = 1024 * 1024
# The final charge for the call, margin included, which the proxy stamps on every priced reply.
RESPONSE_COST_HEADER: Final = b"x-litellm-response-cost"
# Present only on a reply served from the response cache. That reply still carries the cost of the
# call that originally filled the cache, but nothing is charged for serving it.
CACHE_HIT_HEADER: Final = b"x-litellm-cache-key"


class _BodyTooLarge(Exception):
    pass


class _ClientDisconnected(Exception):
    pass


class SystemOneAdapterMiddleware:
    """
    Serves TypeSafe's System One protocol (`POST /v1/systemone`) on top of the Jev chat deployment.

    A System One request is `{model, state, questions}` and its reply is the typed answers body.
    Jev reaches this proxy as a chat deployment (`callbacks.jev_provider`) whose one message is that
    same JSON, and whose reply carries the answers body as the message content. This middleware is
    only the translation between the two shapes: the request becomes an ordinary
    `POST /v1/chat/completions` before routing, and a successful reply is unwrapped from
    `choices[0].message.content` on the way out.

    It is a rewrite rather than a route of its own so that authentication, model access, managed
    account holds and settlement, tier budgets, cost margins and the spend log all run on the
    request exactly as they do for any chat call. A separate handler would have to re-implement
    each of those against a body they do not understand, and a gap in any one is unbilled spend.

    The `usage.cost` Jev reports is the provider's raw charge. The client is billed more than that
    (the configured margin), so the reply carries the amount actually charged for the call, read
    from the proxy's own `x-litellm-response-cost` header, in its place. A reply served from the
    response cache is charged nothing, so it reports 0 even though the header still carries the
    original call's price. When the cost is unknown the field is dropped rather than left showing a
    price the account was not charged.

    Errors (auth, budget, model access, upstream failures) are passed through untouched in the
    proxy's usual error shape.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in SYSTEM_ONE_PATHS:
            await self.app(scope, receive, send)
            return

        try:
            raw_body: Final = await _read_body(receive)
        except _BodyTooLarge:
            await _send_error(send, 413, "System One request body is too large")
            return
        except _ClientDisconnected:
            return

        rejection, chat_body = _chat_body_from_system_one(raw_body)
        if chat_body is None:
            await _send_error(send, 400, rejection)
            return

        await self.app(_chat_scope(scope, chat_body), _single_body_receive(chat_body, receive), _unwrapping_send(send))


def _chat_body_from_system_one(raw_body: bytes) -> tuple[str, bytes | None]:
    """The chat body carrying a System One request, or the reason the request cannot be served."""
    try:
        request: object = json.loads(raw_body)
    except ValueError:
        return "System One request body must be JSON", None
    if not isinstance(request, dict):
        return "System One request body must be a JSON object", None
    model: Final = request.get("model")
    if not isinstance(model, str) or not model.startswith(SYSTEM_ONE_MODEL_PREFIX):
        return f"System One requires a decisions model (`{SYSTEM_ONE_MODEL_PREFIX}*`), got {model!r}", None
    if not isinstance(request.get("state"), str):
        return "System One request needs a string `state`", None
    questions: Final = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        return "System One request needs a non-empty `questions` object", None
    transport: Final = {key: value for key, value in request.items() if key != "model"}
    chat_body: Final = {
        "model": model,
        "messages": [{"role": "user", "content": json.dumps(transport, separators=(",", ":"))}],
    }
    return "", json.dumps(chat_body, separators=(",", ":")).encode("utf-8")


async def _read_body(receive: Receive) -> bytes:
    chunks: Final[list[bytes]] = []  # mutable-ok: accumulates the streamed request body
    size = 0
    while True:
        message: Final = await receive()
        if message["type"] != "http.request":
            raise _ClientDisconnected
        chunk: Final = message.get("body", b"")
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise _BodyTooLarge
        chunks.append(chunk)
        if not message.get("more_body", False):
            return b"".join(chunks)


def _chat_scope(scope: Scope, chat_body: bytes) -> Scope:
    # The reply is parsed on the way out, so it must not be compressed by anything inside.
    headers: Final = [
        (name, value)
        for name, value in scope["headers"]
        if name.lower() not in {b"content-length", b"content-type", b"accept-encoding"}
    ]
    headers.extend(
        [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(chat_body)).encode("latin-1")),
        ]
    )
    return {
        **scope,
        "path": CHAT_COMPLETIONS_PATH,
        "raw_path": CHAT_COMPLETIONS_PATH.encode("ascii"),
        "headers": headers,
    }


def _single_body_receive(chat_body: bytes, receive: Receive) -> Receive:
    delivered = False

    async def rewritten_receive() -> Message:
        nonlocal delivered
        if delivered:
            # Only the disconnect is left to report, which the original receive still knows.
            return await receive()
        delivered = True
        return {"type": "http.request", "body": chat_body, "more_body": False}

    return rewritten_receive


def _unwrapping_send(send: Send) -> Send:
    """Pass errors through; hold a 200 until it can be unwrapped from the chat completion."""
    held_start: Message | None = None
    chunks: Final[list[bytes]] = []  # mutable-ok: accumulates the streamed response body

    async def unwrap_send(message: Message) -> None:
        nonlocal held_start
        if message["type"] == "http.response.start" and message["status"] == 200:
            held_start = message
            return
        if held_start is None or message["type"] != "http.response.body":
            await send(message)
            return
        chunks.append(message.get("body", b""))
        if message.get("more_body", False):
            return
        start: Final = held_start
        held_start = None
        answers: Final = _answers_from_completion(b"".join(chunks), _billed_cost(start["headers"]))
        if answers is None:
            await _send_error(send, 502, "Jev returned an unreadable System One answer")
            return
        headers: Final = [
            (name, value)
            for name, value in start["headers"]
            if name.lower() not in {b"content-length", b"content-type"}
        ]
        headers.extend(
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(answers)).encode("latin-1")),
            ]
        )
        await send({**start, "headers": headers})
        await send({"type": "http.response.body", "body": answers, "more_body": False})

    return unwrap_send


def _billed_cost(headers: list[tuple[bytes, bytes]]) -> float | None:
    """What the account was charged for this reply, or None when the proxy did not say."""
    reported: bytes | None = None
    for name, value in headers:
        lowered: Final = name.lower()
        if lowered == CACHE_HIT_HEADER and value.strip():
            return 0.0
        if lowered == RESPONSE_COST_HEADER:
            reported = value
    if reported is None:
        return None
    try:
        cost: Final = float(reported)
    except ValueError:
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def _answers_from_completion(body: bytes, billed_cost: float | None) -> bytes | None:
    """The System One body a chat completion carries as its message content, or None."""
    try:
        completion: object = json.loads(body)
        if not isinstance(completion, Mapping):
            return None
        answers: Final = json.loads(completion["choices"][0]["message"]["content"])
    except (ValueError, LookupError, TypeError):
        return None
    if not isinstance(answers, Mapping) or not isinstance(answers.get("answers"), Mapping):
        return None
    return json.dumps(_with_billed_cost(answers, billed_cost), separators=(",", ":")).encode("utf-8")


def _with_billed_cost(answers: Mapping[str, object], billed_cost: float | None) -> dict[str, object]:
    usage: Final = answers.get("usage")
    if not isinstance(usage, Mapping):
        return dict(answers)
    priced: Final = {key: value for key, value in usage.items() if key != "cost"}
    if billed_cost is not None:
        priced["cost"] = billed_cost
    return {**answers, "usage": priced}


async def _send_error(send: Send, status: int, message: str) -> None:
    body: Final = json.dumps(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "api_error",
                "param": None,
                "code": str(status),
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})
