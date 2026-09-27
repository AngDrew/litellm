"""Opt-in acceptance run for a client that hangs up mid-stream.

``httpx.ASGITransport`` drives the app to completion and buffers every body frame, so the
in-process acceptance run cannot deliver a client that leaves mid-response: the app never
sees a disconnect and the hold settles on the success path. This module serves the same
proxy app over a real socket (uvicorn on an ephemeral port) and streams from a stub
upstream that stays mid-flight longer than the test client lives, so the client's close is
a real TCP close and the proxy's cancel path runs.

Run it with:

    scripts/tokenin_e2e_disconnect.sh

Every assertion is a money-path invariant: provider output that may have been billed is
never refunded, a finalised reservation cannot be re-resolved, and an unpriced outcome
stays reserved until an operator clears it.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Final

import httpx
import pytest
import pytest_asyncio
import uvicorn
import yaml
from test_account_spend_e2e import (
    _CALLBACK_LISTS,
    MASTER_KEY,
    SERVICE_TOKEN,
    _account_state,
    _config,
    _enrolled_account,
    _grant,
    _service_post,
)

import litellm

pytestmark = pytest.mark.skipif(
    not os.environ.get("TOKENIN_E2E_DATABASE_URL"),
    reason="set TOKENIN_E2E_DATABASE_URL to a local ephemeral Postgres to run this acceptance test",
)

SLOW_MODEL: Final = "tokenin-slow-stream"
REQUEST_TIMEOUT: Final = httpx.Timeout(30.0)
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class SlowUpstream:
    """Minimal OpenAI-compatible SSE endpoint that keeps producing after the client leaves.

    ``first_chunk_delay`` and ``chunk_interval`` are set per test. The per-request counters
    are the proof that the interruption was real: a disconnect only means something while
    this endpoint still has chunks left to write and has not completed that response.
    Counting per request keeps a previous test's abandoned loop, which uvicorn lets run on
    after its socket closes, out of the current test's numbers.
    """

    def __init__(self) -> None:
        self.url: str = ""
        self.first_chunk_delay: float = 0.0
        self.chunk_interval: float = 1.0
        self.chunk_count: int = 5
        self.requests: int = 0
        self.chunks_sent: Final[dict[int, int]] = {}
        self.completed: Final[set[int]] = set()

    def _chunk(self, index: int) -> bytes:
        payload: Final = {
            "id": "chatcmpl-slow",
            "object": "chat.completion.chunk",
            "created": 1735689600,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"content": f"chunk-{index} "}, "finish_reason": None}],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    async def __call__(self, scope: Message, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        self.requests += 1
        request_number: Final = self.requests
        self.chunks_sent[request_number] = 0
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
        await asyncio.sleep(self.first_chunk_delay)
        for index in range(self.chunk_count):
            await send({"type": "http.response.body", "body": self._chunk(index), "more_body": True})
            self.chunks_sent[request_number] += 1
            await asyncio.sleep(self.chunk_interval)
        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})
        self.completed.add(request_number)


class _Served:
    """A uvicorn server on an ephemeral port, started inside the test's own event loop."""

    def __init__(self, app: Any) -> None:
        self._server: Final = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="off"))
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> str:
        self._task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            await asyncio.sleep(0.02)
        port: Final = self._server.servers[0].sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def stop(self) -> None:
        self._server.should_exit = True
        if self._task is not None:
            await self._task


_APP_GLOBALS: Final[dict[str, Any]] = {}
_APP_CALLBACKS: Final[dict[str, Any]] = {}


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def slow_upstream() -> AsyncIterator[SlowUpstream]:
    upstream: Final = SlowUpstream()
    served: Final = _Served(upstream)
    upstream.url = await served.start()
    yield upstream
    await served.stop()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def live_server(slow_upstream: SlowUpstream) -> AsyncIterator[str]:
    database_url: Final = os.environ["TOKENIN_E2E_DATABASE_URL"]
    os.environ["TOKENIN_ACCOUNT_V2_ENABLED"] = "true"
    os.environ["TOKENIN_ACCOUNT_SPEND_ENABLED"] = "true"
    os.environ["TOKENIN_ACCOUNT_SERVICE_TOKEN"] = SERVICE_TOKEN
    os.environ["LITELLM_MASTER_KEY"] = MASTER_KEY

    base_config: Final = _config(database_url)
    config: Final = {
        **base_config,
        "model_list": [
            *base_config["model_list"],
            {
                "model_name": SLOW_MODEL,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-stub-provider",
                    "api_base": slow_upstream.url,
                },
            },
        ],
    }
    config_path: Final = tempfile.NamedTemporaryFile(mode="w", suffix="-disconnect.yaml", delete=False).name
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.dump(config, handle)
    os.environ["CONFIG_FILE_PATH"] = config_path

    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import app, cleanup_router_config_variables, initialize, proxy_startup_event

    cleanup_router_config_variables()
    await initialize(config=config_path)
    async with proxy_startup_event(app):
        assert proxy_server.prisma_client is not None
        await proxy_server.prisma_client.check_view_exists()
        for name in ("master_key", "prisma_client", "llm_router"):
            _APP_GLOBALS[name] = getattr(proxy_server, name)
        for name in _CALLBACK_LISTS:
            _APP_CALLBACKS[name] = getattr(litellm, name)
        # uvicorn serves the already-started app: its own lifespan would run startup twice.
        served: Final = _Served(app)
        base_url: Final = await served.start()
        yield base_url
        await served.stop()
    os.unlink(config_path)


@pytest.fixture(autouse=True)
def _restore_app_globals(live_server) -> None:
    from litellm.proxy import proxy_server

    for name, value in _APP_GLOBALS.items():
        setattr(proxy_server, name, value)
    for name, value in _APP_CALLBACKS.items():
        setattr(litellm, name, value)
    yield


@pytest_asyncio.fixture(loop_scope="module")
async def prisma(live_server):
    from litellm.proxy import proxy_server

    return proxy_server.prisma_client


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client(live_server: str) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=live_server, timeout=REQUEST_TIMEOUT) as http_client:
        yield http_client


@pytest_asyncio.fixture(scope="module", loop_scope="module", autouse=True)
async def flat_team(client) -> None:
    """The flat team is provisioning the proxy assumes, not something it creates."""
    from litellm.proxy.tokenin.plans import FLAT_TEAM_ID

    created: Final = await client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json={"team_id": FLAT_TEAM_ID, "team_alias": "tokenin-flat", "models": []},
    )
    assert created.status_code == 200, created.text


async def _await_hold_finalized(prisma, user_id: str, *, timeout: float = 25.0) -> dict[str, Any]:
    """Wait until no reservation is still ``held``: an uncertain hold stays reserved forever."""
    deadline: Final = time.monotonic() + timeout
    state: Final = await _account_state(prisma, user_id)
    while any(row["state"] == "held" for row in state["holds"]) and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        state = await _account_state(prisma, user_id)
    return state


def _stream_body() -> dict[str, Any]:
    return {"model": SLOW_MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True}


async def _disconnect_after_first_chunk(base_url: str, key: str) -> None:
    """Read one data chunk over a real socket, then hang up while the proxy is still streaming."""
    async with httpx.AsyncClient(base_url=base_url, timeout=REQUEST_TIMEOUT) as client:
        async with client.stream(
            "POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=_stream_body()
        ) as response:
            assert response.status_code == 200, await response.aread()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    return
            raise AssertionError("stream ended before the client received a data chunk")


async def _abort_mid_request(base_url: str, key: str, upstream: SlowUpstream, baseline: int) -> int:
    """Abort a raw TCP request after the upstream is waiting for its first chunk."""
    host, port = base_url.removeprefix("http://").split(":")
    writer: Final = (await asyncio.open_connection(host, int(port)))[1]
    body: Final = json.dumps(_stream_body()).encode()
    writer.write(
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        + f"Host: {host}:{port}\r\n".encode()
        + f"Authorization: Bearer {key}\r\n".encode()
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"\r\n"
        + body
    )
    await writer.drain()
    request_number: Final = await _await_upstream_request(upstream, baseline)
    assert request_number == baseline + 1
    assert upstream.chunks_sent[request_number] == 0, "the client must abort before the first provider chunk"
    await asyncio.sleep(0.1)
    assert upstream.chunks_sent[request_number] == 0
    writer.transport.abort()
    writer.close()
    return request_number


async def _await_upstream_request(upstream: SlowUpstream, baseline: int, *, timeout: float = 8.0) -> int:
    """Wait until the proxy has opened the upstream request."""
    deadline: Final = time.monotonic() + timeout
    while upstream.requests <= baseline and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    return upstream.requests


async def _resolved(client: httpx.AsyncClient, request_id: str, action: str) -> httpx.Response:
    return await _service_post(
        client,
        f"/tokenin/account/holds/{request_id}/resolve",
        {"action": action, "note": "disconnect e2e: operator remedy"},
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_disconnect_after_a_chunk_charges_the_partial_stream(
    client, live_server: str, slow_upstream: SlowUpstream, prisma
) -> None:
    """The client takes one chunk and leaves. The provider may have billed for it, so the
    reservation may never be refunded: the disconnect assembles the partial stream, settles
    the cost it can price, and a settled hold refuses a refund. This is also what proves the
    in-flight stream guard in ``proxy_track_cost_callback``: a per-chunk callback that marked
    the still-live hold uncertain would make that settle fail and leave the hold uncertain."""
    user_id, key = await _enrolled_account(client, prisma, models=[SLOW_MODEL])
    await _grant(client, user_id, "0.05")
    slow_upstream.first_chunk_delay = 0.0
    slow_upstream.chunk_interval = 2.0
    slow_upstream.chunk_count = 5

    await _disconnect_after_first_chunk(live_server, key)

    request_number: Final = slow_upstream.requests
    assert slow_upstream.chunks_sent[request_number] == 1, slow_upstream.chunks_sent
    assert request_number not in slow_upstream.completed, (
        "the client must have left while the upstream was still producing"
    )

    state: Final = await _await_hold_finalized(prisma, user_id)
    rows: Final = state["holds"]
    assert rows, "a disconnect after provider output must leave a hold behind"
    charged: Final = int(rows[0]["charged_nano"] or 0)
    assert rows[0]["state"] == "settled", state
    assert 0 < charged < int(rows[0]["estimated_nano"]), state
    assert state["open_nano"] == 0, state
    refused: Final = await _resolved(client, rows[0]["request_id"], "cancel")
    assert refused.status_code == 409, refused.text
    assert (await _account_state(prisma, user_id))["charged_nano"] == charged, "a refund would move credit"


@pytest.mark.asyncio(loop_scope="module")
async def test_disconnect_before_provider_output_refunds_the_reservation(
    client, live_server: str, slow_upstream: SlowUpstream, prisma, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A client aborts before the provider emits output; the shielded finalizer refunds once."""
    from litellm.proxy.tokenin import enforcement

    user_id, key = await _enrolled_account(client, prisma, models=[SLOW_MODEL])
    await _grant(client, user_id, "0.05")
    slow_upstream.first_chunk_delay = 8.0
    slow_upstream.chunk_interval = 2.0
    slow_upstream.chunk_count = 5

    hold_resolutions: Final[list[bool]] = []
    original_handler: Final = enforcement.handle_account_hold_on_cancel

    async def record_hold_resolution(metadata: Mapping[str, object] | None, *, provider_output_delivered: bool) -> None:
        hold_resolutions.append(provider_output_delivered)
        await original_handler(metadata, provider_output_delivered=provider_output_delivered)

    monkeypatch.setattr(enforcement, "handle_account_hold_on_cancel", record_hold_resolution)
    baseline: Final = slow_upstream.requests
    request_number: Final = await _abort_mid_request(live_server, key, slow_upstream, baseline)
    assert slow_upstream.chunks_sent[request_number] == 0, "the upstream emitted no chunk when the client aborted"
    assert request_number not in slow_upstream.completed

    state: Final = await _await_hold_finalized(prisma, user_id)
    rows: Final = state["holds"]
    assert rows, state
    assert rows[0]["state"] == "cancelled", state
    assert int(rows[0]["charged_nano"] or 0) == 0, state
    assert int(rows[0]["held_nano"]) == 0, "cancellation must release the allocation"
    assert state["open_nano"] == 0, state
    assert state["charged_nano"] == 0, state
    assert hold_resolutions == [False], "the shielded finalizer must resolve the hold exactly once"


@pytest.mark.asyncio(loop_scope="module")
async def test_unpriceable_disconnect_stays_reserved_until_the_operator_clears_it(
    client, live_server: str, slow_upstream: SlowUpstream, prisma, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No streaming logging means the partial response can never be priced, so the hold
    stays reserved with its whole amount, and the documented operator remedy is what
    releases it."""
    user_id, key = await _enrolled_account(client, prisma, models=[SLOW_MODEL])
    await _grant(client, user_id, "0.05")
    slow_upstream.first_chunk_delay = 0.0
    slow_upstream.chunk_interval = 2.0
    slow_upstream.chunk_count = 5
    monkeypatch.setattr(litellm, "disable_streaming_logging", True)

    await _disconnect_after_first_chunk(live_server, key)

    state: Final = await _await_hold_finalized(prisma, user_id)
    rows: Final = state["holds"]
    assert rows, state
    assert rows[0]["state"] == "uncertain", state
    assert int(rows[0]["charged_nano"] or 0) == 0, state
    assert int(rows[0]["held_nano"]) == int(rows[0]["estimated_nano"]) > 0, state

    resolved: Final = await _resolved(client, rows[0]["request_id"], "cancel")
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["state"] == "cancelled", resolved.text
    after: Final = await _account_state(prisma, user_id)
    assert after["open_nano"] == 0, after
    assert after["charged_nano"] == 0, after
    assert after["debt_nano"] == 0, after
