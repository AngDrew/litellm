"""Account-hold pricing and settlement seams for the Tokenin ledger.

Everything here is inert until admission stamps ``tokenin_account_hold_id`` on the
authenticated key; without that stamp the helpers return immediately, so the
callback paths behave exactly as before.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy.utils import PrismaClient

if TYPE_CHECKING:
    from litellm.router import Router

TOKENIN_ACCOUNT_HOLD_METADATA_KEY: Final = "user_api_key_tokenin_account_hold_id"
TOKENIN_ACCOUNT_HOLD_AUTH_FIELD: Final = "tokenin_account_hold_id"
# Every request field the router reads to send a failed call to another model group.
# The hold is priced and policy-checked for ``model`` alone, so any other target
# would spend credit on a model the account policy never approved.
ACCOUNT_HOLD_FALLBACK_FIELDS: Final = ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks")


def requests_model_fallbacks(request_data: Mapping[str, object]) -> bool:
    """Whether the body names fallback targets or carries a ``router_settings_override`` of any shape.

    The override is refused whole rather than parsed: ``route_llm_request`` merges its
    ``fallbacks`` into the router call, and every other setting in it is operator
    policy, never client input. Only an absent override is accepted.
    """
    return request_data.get("router_settings_override") is not None or any(
        _names_targets(request_data.get(field)) for field in ACCOUNT_HOLD_FALLBACK_FIELDS
    )


def _names_targets(value: object) -> bool:
    """Absent and empty lists request nothing; any other value is refused, including malformed ones."""
    return not (value is None or (isinstance(value, list) and not value))


def pin_account_hold_to_requested_model(
    data: dict[str, object],  # mutable-ok: add_litellm_data_to_request hands every stage the same request dict
) -> None:
    """Turn off every fallback path for a held request, including proxy-configured ones.

    Explicit empty lists win over the router's configured fallbacks on each read
    (``kwargs.get("fallbacks", self.fallbacks)``, the mid-stream retry included),
    and a key/team ``router_settings_override`` only fills fields the request lacks.
    ``disable_fallbacks`` also stops the proxy's local rate-limit fallback.
    """
    # An explicit empty list is what overrides the router's configured fallbacks.
    emptied: Final = {field: [] for field in ACCOUNT_HOLD_FALLBACK_FIELDS}  # mutable-ok: router reads lists
    data.update(emptied, disable_fallbacks=True)


def require_direct_held_model(
    *,
    approved_model: str | None,
    model: object,
    router: Router | None,
    team_id: str | None,
) -> None:
    """Reject held requests that could route to a model group other than the one priced at admission."""
    import litellm

    exact_deployments: Final = (
        router._get_all_deployments(  # pyright: ignore[reportPrivateUsage]  # exact lookup excludes wildcard/default routing
            model_name=approved_model, team_id=team_id
        )
        if router is not None and approved_model is not None
        else ()
    )
    if (
        not approved_model
        or not isinstance(model, str)
        or model != approved_model
        or router is None
        or model in router.model_group_alias
        or model in litellm.model_alias_map
        or router.has_model_id(model)
        or (team_id is not None and router.map_team_model(model, team_id) not in (None, model))
        or not exact_deployments
        or any(
            "silent_model" in deployment["litellm_params"] and deployment["litellm_params"]["silent_model"] is not None
            for deployment in exact_deployments
        )
    ):
        raise HTTPException(status_code=503, detail="Managed account model requires direct, single-model routing")


def account_hold_id(metadata: Mapping[str, object] | None) -> str | None:
    if not metadata:
        return None
    stamped: Final = metadata.get(TOKENIN_ACCOUNT_HOLD_METADATA_KEY)
    if isinstance(stamped, str) and stamped:
        return stamped
    auth: Final = metadata.get("user_api_key_auth")
    if isinstance(auth, Mapping):
        fallback: Final = auth.get(TOKENIN_ACCOUNT_HOLD_AUTH_FIELD)
    else:
        fallback = getattr(auth, TOKENIN_ACCOUNT_HOLD_AUTH_FIELD, None)
    return fallback if isinstance(fallback, str) and fallback else None


async def estimate_account_max_cost(request_body: dict, route: str, llm_router: Router | None) -> float:
    """Worst-case cost across every deployment the routed model group can select.

    The budget estimator drops unpriced candidates; the account ledger must not,
    because the deployment that ends up serving the request can be the unpriced
    one. Any unpriceable candidate fails the request closed instead.
    """
    from litellm.proxy.spend_tracking.budget_reservation import (
        _estimate_request_max_cost_for_model,
        _get_request_models,
        count_request_input_tokens,
    )

    models: Final[Sequence[str]] = tuple(
        _get_request_models(request_body=request_body, route=route, llm_router=llm_router)
    )
    if not models:
        raise HTTPException(status_code=503, detail="Requested model is not configured on this proxy")
    counts: Final = await count_request_input_tokens(request_body=request_body, route=route, llm_router=llm_router)
    estimates: Final[list[float]] = []
    for model_name in models:
        estimate = _estimate_request_max_cost_for_model(
            request_body=request_body,
            route=route,
            model=model_name,
            llm_router=llm_router,
            input_tokens=counts.get(model_name),
        )
        if estimate is None or estimate <= 0:
            raise HTTPException(status_code=503, detail=f"Model '{model_name}' has no supported maximum cost")
        estimates.append(estimate)
    return max(estimates)


def hold_metadata_from_request_data(request_data: Mapping[str, object] | None) -> Mapping[str, object] | None:
    """Return whichever request-data bucket carries the account hold stamp."""
    if not request_data:
        return None
    for bucket in ("litellm_metadata", "metadata"):
        candidate: Final = request_data.get(bucket)
        if isinstance(candidate, Mapping) and account_hold_id(candidate) is not None:
            return candidate
    return None


def _prisma_client() -> PrismaClient | None:
    from litellm.proxy.proxy_server import prisma_client

    return prisma_client


async def settle_account_hold(metadata: Mapping[str, object] | None, actual_cost: float) -> None:
    """Charge a completed request. A failure here leaves the hold reserved, never refunded."""
    hold_id: Final = account_hold_id(metadata)
    if hold_id is None:
        return
    prisma_client: Final = _prisma_client()
    if prisma_client is None:
        verbose_proxy_logger.error("Tokenin account hold %s left reserved: database unavailable", hold_id)
        return
    from litellm.proxy.tokenin.ledger import settle_account_request

    try:
        await settle_account_request(prisma_client, hold_id, actual_cost)
    except Exception:
        # ponytail: no retry loop here; a stuck hold is cleared by reconciliation, not by refunding it
        verbose_proxy_logger.exception("Tokenin account hold %s was not settled", hold_id)


async def mark_account_hold_uncertain(metadata: Mapping[str, object] | None) -> None:
    """Keep the reservation when the outcome is unknown, so nothing is refunded speculatively."""
    hold_id: Final = account_hold_id(metadata)
    if hold_id is None:
        return
    prisma_client: Final = _prisma_client()
    if prisma_client is None:
        verbose_proxy_logger.error("Tokenin account hold %s stays uncertain: database unavailable", hold_id)
        return
    from litellm.proxy.tokenin.ledger import mark_account_request_uncertain

    try:
        await mark_account_request_uncertain(prisma_client, hold_id)
    except Exception:
        verbose_proxy_logger.exception("Tokenin account hold %s was not marked uncertain", hold_id)


async def cancel_account_hold(metadata: Mapping[str, object] | None) -> None:
    """Release a reservation only when billing is known not to have happened."""
    hold_id: Final = account_hold_id(metadata)
    if hold_id is None:
        return
    prisma_client: Final = _prisma_client()
    if prisma_client is None:
        verbose_proxy_logger.error("Tokenin account hold %s left reserved: database unavailable", hold_id)
        return
    from litellm.proxy.tokenin.ledger import cancel_account_request

    try:
        await cancel_account_request(prisma_client, hold_id)
    except Exception:
        verbose_proxy_logger.exception("Tokenin account hold %s was not cancelled", hold_id)


async def handle_account_hold_on_cancel(metadata: Mapping[str, object] | None, *, billing_known_absent: bool) -> None:
    """Release only when a caller has established that provider billing did not occur."""
    if billing_known_absent:
        await cancel_account_hold(metadata=metadata)
    else:
        await mark_account_hold_uncertain(metadata=metadata)
