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


async def handle_account_hold_on_cancel(
    metadata: Mapping[str, object] | None, *, provider_output_delivered: bool
) -> None:
    """Client-cancel outcome: refund only when the provider produced no output.

    A delivered chunk means the provider may already have billed, and a stream that
    broke mid-flight cannot be priced from here, so the reservation stays uncertain
    and visible to reconciliation instead of being refunded.
    """
    if provider_output_delivered:
        await mark_account_hold_uncertain(metadata=metadata)
    else:
        await cancel_account_hold(metadata=metadata)
