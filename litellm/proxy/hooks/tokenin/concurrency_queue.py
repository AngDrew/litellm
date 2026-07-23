import asyncio
import time
import weakref
from typing import Any, Dict, Optional, Protocol, Tuple
from uuid import uuid4

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger

_SLOT_METADATA_KEY = "_tokenin_concurrency_queue_slot"
_SUPPORTED_CALL_TYPES = {
    "acompletion",
    "completion",
    "embeddings",
    "aembedding",
    "aresponses",
    "aimage_generation",
    "arealtime_calls",
}


class _TimeProvider(Protocol):
    def time(self) -> float: ...


class TokeninConcurrencyQueue(CustomLogger):
    callback_specific_params_key = "tokenin_concurrency_queue"
    _semaphores: weakref.WeakValueDictionary[str, asyncio.Semaphore] = weakref.WeakValueDictionary()
    _semaphore_lock = asyncio.Lock()
    _active_slots: Dict[str, Tuple[str, asyncio.Semaphore]] = {}

    def __init__(
        self,
        max_wait_seconds: float = 300,
        enabled: bool = True,
        time_provider: Optional[_TimeProvider] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_wait_seconds = max_wait_seconds
        self.enabled = enabled
        self._time_provider = time_provider or time

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict:
        if not self.enabled or call_type not in _SUPPORTED_CALL_TYPES:
            return data

        max_parallel_requests = getattr(user_api_key_dict, "max_parallel_requests", None)
        api_key = getattr(user_api_key_dict, "api_key", None)
        if max_parallel_requests is None or max_parallel_requests <= 0 or not api_key:
            return data

        semaphore = await self._get_or_create_semaphore(api_key=api_key, limit=max_parallel_requests)
        key_id = api_key[:8]
        verbose_proxy_logger.debug(
            "TokeninConcurrencyQueue: semaphore acquisition start key=%s call_type=%s",
            key_id,
            call_type,
        )
        if semaphore.locked():
            verbose_proxy_logger.debug(
                "TokeninConcurrencyQueue: waiting for semaphore key=%s timeout=%s",
                key_id,
                self.max_wait_seconds,
            )

        started_at = self._time_provider.time()
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=self.max_wait_seconds)
        except TimeoutError:
            verbose_proxy_logger.debug(
                "TokeninConcurrencyQueue: semaphore timeout key=%s timeout=%s",
                key_id,
                self.max_wait_seconds,
            )
            raise HTTPException(
                status_code=429,
                detail="Concurrency queue timeout: request waited too long for an available slot",
            )

        try:
            self._record_slot(data=data, api_key=api_key, semaphore=semaphore)
        except Exception:
            semaphore.release()
            raise

        verbose_proxy_logger.debug(
            "TokeninConcurrencyQueue: semaphore acquired key=%s waited=%s",
            key_id,
            self._time_provider.time() - started_at,
        )
        return data

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        self._release_slot(kwargs)

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        self._release_slot(kwargs)

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict,
    ) -> Any:
        try:
            async for item in response:
                yield item
        finally:
            self._release_slot(request_data)

    async def _get_or_create_semaphore(self, api_key: str, limit: int) -> asyncio.Semaphore:
        semaphore = self._semaphores.get(api_key)
        if semaphore is not None:
            return semaphore

        async with self._semaphore_lock:
            semaphore = self._semaphores.get(api_key)
            if semaphore is None:
                semaphore = asyncio.Semaphore(limit)
                self._semaphores[api_key] = semaphore
            return semaphore

    def _record_slot(
        self,
        data: dict,
        api_key: str,
        semaphore: asyncio.Semaphore,
    ) -> None:
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            data["metadata"] = metadata

        slot_id = uuid4().hex
        type(self)._active_slots[slot_id] = (api_key, semaphore)
        metadata[_SLOT_METADATA_KEY] = slot_id

    def _release_slot(self, event_data: dict) -> None:
        metadata = self._metadata_from_event(event_data)
        if metadata is None:
            return

        slot_id = metadata.pop(_SLOT_METADATA_KEY, None)
        if not isinstance(slot_id, str):
            return

        active_slot = type(self)._active_slots.pop(slot_id, None)
        if active_slot is None:
            verbose_proxy_logger.debug("TokeninConcurrencyQueue: semaphore slot already released")
            return

        api_key, semaphore = active_slot
        semaphore.release()
        verbose_proxy_logger.debug("TokeninConcurrencyQueue: semaphore released key=%s", api_key[:8])

    @classmethod
    def request_has_active_slot(cls, data: dict) -> bool:
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            return False
        slot_id = metadata.get(_SLOT_METADATA_KEY)
        return isinstance(slot_id, str) and slot_id in cls._active_slots

    @staticmethod
    def _metadata_from_event(event_data: dict) -> Optional[dict]:
        litellm_params = event_data.get("litellm_params")
        if isinstance(litellm_params, dict):
            metadata = litellm_params.get("metadata")
            if isinstance(metadata, dict):
                return metadata

        metadata = event_data.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        return None
