import re
import time

from fastapi import HTTPException

from litellm import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

BUDGET_TIERS = {
    "free": {"max_budget": 0, "budget_duration": "1d"}, # 0
    "xiaodi": {"max_budget": 5, "budget_duration": "1d"}, # 50k
    "tauke": {"max_budget": 15, "budget_duration": "1d"}, # 150k
    "laoban": {"max_budget": 45, "budget_duration": "1d"}, # 450k
    "taipan": {"max_budget": 90, "budget_duration": "1d"}, # 900k
}

_DURATION_RE = re.compile(r"^(\d+)(s|m|h|d|mo)$")

_DURATION_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "mo": 2592000,
}


def _parse_duration(duration_str: str) -> int:
    match = _DURATION_RE.match(duration_str)
    if not match:
        raise ValueError(f"Invalid duration: {duration_str}")
    amount, unit = int(match.group(1)), match.group(2)
    return amount * _DURATION_SECONDS[unit]


def _window_key(token: str, tier: str, duration_seconds: int) -> str:
    window_start = int(time.time() // duration_seconds) * duration_seconds
    return f"tier_budget:{tier}:{token}:{window_start}"


def _get_tier_from_metadata(metadata: dict) -> str:
    tier = metadata.get("tier") or metadata.get("team_tier")
    return tier if tier and tier in BUDGET_TIERS else "default"


class TieredPricingCallback(CustomLogger):
    _cache = None

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self._cache = cache
        metadata = getattr(user_api_key_dict, "metadata", None) or {}
        tier = _get_tier_from_metadata(metadata)

        budget_config = BUDGET_TIERS.get(tier)
        if not budget_config:
            return

        max_budget = budget_config["max_budget"]
        if max_budget <= 0:
            if max_budget == 0:
                raise HTTPException(
                    status_code=429,
                    detail=f"Tier '{tier}' has $0 budget. Requests blocked.",
                )
            return

        token = getattr(user_api_key_dict, "token", None)
        if not token:
            return

        try:
            duration_seconds = _parse_duration(budget_config["budget_duration"])
        except ValueError:
            return

        cache_key = _window_key(token, tier, duration_seconds)
        current_spend = await cache.async_get_cache(key=cache_key)
        current_spend = float(current_spend) if current_spend else 0.0

        if current_spend >= max_budget:
            verbose_logger.info(
                f"[TieredPricing] BLOCKED tier={tier} key={token[:8]}... "
                f"spend=${current_spend:.4f} >= limit=${max_budget:.2f}"
            )
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Tier '{tier}' budget exceeded. "
                    f"Spent ${current_spend:.4f} of ${max_budget:.2f} "
                    f"({budget_config['budget_duration']} window)."
                ),
            )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        cost = kwargs.get("response_cost", 0) or 0
        await self._increment_budget(kwargs, cost)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        kwargs["response_cost"] = 0

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        kwargs["response_cost"] = 0

    async def _increment_budget(self, kwargs, cost):
        if not cost or cost <= 0:
            return

        metadata = kwargs.get("litellm_params", {}).get("metadata", {}) or {}
        tier = _get_tier_from_metadata(metadata)

        budget_config = BUDGET_TIERS.get(tier)
        if not budget_config or budget_config["max_budget"] <= 0:
            return

        token = kwargs.get("litellm_params", {}).get("metadata", {}).get("user_api_key")
        if not token:
            return

        try:
            duration_seconds = _parse_duration(budget_config["budget_duration"])
        except ValueError:
            return

        cache_key = _window_key(token, tier, duration_seconds)

        if self._cache:
            await self._cache.async_increment_cache(
                key=cache_key,
                value=cost,
                ttl=duration_seconds,
            )
