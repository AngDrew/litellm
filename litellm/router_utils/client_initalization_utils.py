import asyncio
import hashlib
from typing import TYPE_CHECKING, Any, Final

from litellm.utils import calculate_max_parallel_requests

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


class InitalizeCachedClient:
    @staticmethod
    def get_max_parallel_requests_cache_key(model: dict) -> str:
        """
        Cache key for the max-parallel-requests semaphore.

        Deployments with ``provider_max_parallel_requests`` share one semaphore
        per provider (same api_base + api_key); all others keep a per-deployment
        semaphore keyed by model id.
        """
        litellm_params: Final = model.get("litellm_params", {})
        model_id: Final = model["model_info"]["id"]
        if litellm_params.get("provider_max_parallel_requests") is not None:
            provider_identity: Final = f"{litellm_params.get('api_base')}\0{litellm_params.get('api_key')}"
            return f"{hashlib.sha256(provider_identity.encode()).hexdigest()}_max_parallel_requests_client"
        return f"{model_id}_max_parallel_requests_client"

    @staticmethod
    def set_max_parallel_requests_client(litellm_router_instance: LitellmRouter, model: dict):
        litellm_params: Final = model.get("litellm_params", {})
        model_id: Final = model["model_info"]["id"]
        rpm: Final = litellm_params.get("rpm", None)
        tpm: Final = litellm_params.get("tpm", None)
        max_parallel_requests: Final = litellm_params.get("max_parallel_requests", None)
        provider_max_parallel_requests: Final = litellm_params.get("provider_max_parallel_requests", None)
        calculated_max_parallel_requests: Final = calculate_max_parallel_requests(
            rpm=rpm,
            max_parallel_requests=max_parallel_requests,
            tpm=tpm,
            default_max_parallel_requests=litellm_router_instance.default_max_parallel_requests,
            provider_max_parallel_requests=provider_max_parallel_requests,
        )
        cache_key: Final = InitalizeCachedClient.get_max_parallel_requests_cache_key(model)
        shared_caps: Final = (
            [
                cap
                for cap in (
                    calculate_max_parallel_requests(
                        rpm=lp.get("rpm", None),
                        max_parallel_requests=lp.get("max_parallel_requests", None),
                        tpm=lp.get("tpm", None),
                        default_max_parallel_requests=litellm_router_instance.default_max_parallel_requests,
                        provider_max_parallel_requests=lp.get("provider_max_parallel_requests", None),
                    )
                    for lp in (
                        d.get("litellm_params", {})
                        for d in litellm_router_instance.model_list
                        if InitalizeCachedClient.get_max_parallel_requests_cache_key(d) == cache_key
                    )
                    if lp.get("provider_max_parallel_requests") is not None
                )
                if cap is not None
            ]
            if provider_max_parallel_requests is not None
            else []
        )
        semaphore_max_parallel_requests: Final = min(shared_caps) if shared_caps else calculated_max_parallel_requests
        if semaphore_max_parallel_requests:
            # provider_max_parallel_requests: one semaphore shared by every
            # deployment pointing at the same provider (same api_base + api_key),
            # e.g. N models on a single ollama cloud account. The shared cap is
            # the MIN of every deployment on the provider key — lazy per-deployment
            # creation made a mixed set non-deterministic (first-writer-wins); a
            # loose value on any one deployment must never widen the shared cap.
            # ponytail: semaphore is local-only, so multi-replica LiteLLM
            # multiplies the cap per replica — a Redis-backed counter is the
            # upgrade path.
            semaphore: Final = asyncio.Semaphore(semaphore_max_parallel_requests)
            litellm_router_instance.cache.set_cache(
                key=cache_key,
                value=semaphore,
                local_only=True,
            )
