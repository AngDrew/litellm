import asyncio
import sys
from typing import Any, Dict, Optional

import pytest
from fastapi import HTTPException

from litellm.proxy.hooks.tokenin.concurrency_queue import TokeninConcurrencyQueue


class FakeTimeProvider:
    def __init__(self) -> None:
        self.now = 0.0
        self.calls = 0

    def time(self) -> float:
        self.calls += 1
        return self.now


class FakeUserAPIKeyAuth:
    def __init__(
        self,
        api_key: Optional[str] = "hashed-key-123",
        max_parallel_requests: Optional[int] = 1,
    ) -> None:
        self.api_key = api_key
        self.max_parallel_requests = max_parallel_requests


def _user(api_key: str = "hashed-key-123") -> FakeUserAPIKeyAuth:
    return FakeUserAPIKeyAuth(api_key=api_key, max_parallel_requests=1)


def _event_kwargs(data: dict) -> dict:
    return {"litellm_params": {"metadata": data["metadata"]}}


async def _release_success(hook: TokeninConcurrencyQueue, data: dict) -> None:
    await hook.async_log_success_event(_event_kwargs(data), None, None, None)


@pytest.mark.asyncio
async def test_request_passes_under_limit() -> None:
    time_provider = FakeTimeProvider()
    hook = TokeninConcurrencyQueue(time_provider=time_provider)
    data: Dict[str, Any] = {}
    user = _user("under-limit-key")

    result = await hook.async_pre_call_hook(user, None, data, "acompletion")

    assert result is data
    assert user.max_parallel_requests == sys.maxsize
    assert time_provider.calls == 2
    await _release_success(hook, data)


@pytest.mark.asyncio
async def test_second_request_waits_until_first_finishes() -> None:
    hook = TokeninConcurrencyQueue(max_wait_seconds=1)
    first_data: Dict[str, Any] = {}
    second_data: Dict[str, Any] = {}
    await hook.async_pre_call_hook(_user("waiting-key"), None, first_data, "acompletion")

    second_request = asyncio.create_task(
        hook.async_pre_call_hook(_user("waiting-key"), None, second_data, "acompletion")
    )
    await asyncio.sleep(0)
    assert not second_request.done()

    await _release_success(hook, first_data)
    assert await asyncio.wait_for(second_request, timeout=0.1) is second_data
    await _release_success(hook, second_data)


@pytest.mark.asyncio
async def test_request_times_out_when_no_slot_frees() -> None:
    hook = TokeninConcurrencyQueue(max_wait_seconds=0.01)
    first_data: Dict[str, Any] = {}
    await hook.async_pre_call_hook(_user("timeout-key"), None, first_data, "acompletion")

    with pytest.raises(HTTPException) as exc_info:
        await hook.async_pre_call_hook(_user("timeout-key"), None, {}, "acompletion")

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail == "Concurrency queue timeout: request waited too long for an available slot"
    await _release_success(hook, first_data)


@pytest.mark.asyncio
async def test_streaming_request_releases_slot_when_stream_completes() -> None:
    hook = TokeninConcurrencyQueue(max_wait_seconds=1)
    stream_data: Dict[str, Any] = {"stream": True}
    await hook.async_pre_call_hook(_user("stream-key"), None, stream_data, "acompletion")

    async def stream():
        yield "first"
        yield "second"

    chunks = [
        chunk
        async for chunk in hook.async_post_call_streaming_iterator_hook(_user("stream-key"), stream(), stream_data)
    ]
    assert chunks == ["first", "second"]

    next_data: Dict[str, Any] = {}
    await asyncio.wait_for(
        hook.async_pre_call_hook(_user("stream-key"), None, next_data, "acompletion"),
        timeout=0.1,
    )
    await _release_success(hook, next_data)


@pytest.mark.asyncio
async def test_streaming_request_releases_slot_when_stream_fails() -> None:
    hook = TokeninConcurrencyQueue(max_wait_seconds=1)
    stream_data: Dict[str, Any] = {"stream": True}
    await hook.async_pre_call_hook(_user("failed-stream-key"), None, stream_data, "acompletion")

    async def stream():
        yield "first"
        raise RuntimeError("stream failed")

    with pytest.raises(RuntimeError, match="stream failed"):
        async for _ in hook.async_post_call_streaming_iterator_hook(_user("failed-stream-key"), stream(), stream_data):
            pass

    next_data: Dict[str, Any] = {}
    await asyncio.wait_for(
        hook.async_pre_call_hook(_user("failed-stream-key"), None, next_data, "acompletion"),
        timeout=0.1,
    )
    await _release_success(hook, next_data)


@pytest.mark.asyncio
async def test_success_and_failure_events_do_not_release_twice() -> None:
    hook = TokeninConcurrencyQueue(max_wait_seconds=0.01)
    data: Dict[str, Any] = {}
    await hook.async_pre_call_hook(_user("single-release-key"), None, data, "acompletion")
    kwargs = _event_kwargs(data)

    await hook.async_log_success_event(kwargs, None, None, None)
    await hook.async_log_failure_event(kwargs, None, None, None)

    first: Dict[str, Any] = {}
    second: Dict[str, Any] = {}
    await hook.async_pre_call_hook(_user("single-release-key"), None, first, "acompletion")
    with pytest.raises(HTTPException):
        await hook.async_pre_call_hook(_user("single-release-key"), None, second, "acompletion")
    await _release_success(hook, first)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    ["moderation", "audio_transcription", "batch"],
)
async def test_unsupported_call_types_are_unchanged(call_type: str) -> None:
    hook = TokeninConcurrencyQueue()
    data: Dict[str, Any] = {}

    assert await hook.async_pre_call_hook(_user(), None, data, call_type) is data
    assert data == {}


@pytest.mark.asyncio
async def test_disabled_or_unlimited_keys_are_unchanged() -> None:
    data: Dict[str, Any] = {}
    disabled_hook = TokeninConcurrencyQueue(enabled=False)
    assert await disabled_hook.async_pre_call_hook(_user(), None, data, "acompletion") is data

    unlimited_hook = TokeninConcurrencyQueue()
    unlimited_user = FakeUserAPIKeyAuth(max_parallel_requests=None)
    assert await unlimited_hook.async_pre_call_hook(unlimited_user, None, data, "acompletion") is data
    assert data == {}
