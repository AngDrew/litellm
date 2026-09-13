"""
TOKENIN ACCOUNT WALLET

One prepaid balance per account, hosted on the user row's `max_budget` ceiling and
shared by every key the account owns. Purchases and top-ups raise the ceiling, the
spend writer accrues usage to the same row, and no key carries a budget of its own.
The account's team keys are gated by this wallet through
general_settings.apply_user_budget_to_team_keys.
"""

from typing import TYPE_CHECKING, Final

from fastapi import HTTPException, status

from litellm.repositories.user_repository import UserRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_EMPTY_WALLET_BUDGET: Final = 0.0


async def ensure_wallet_user(prisma_client: "PrismaClient", user_id: str) -> None:
    """Create the account's user row when it is missing. An existing row keeps its max_budget and spend."""
    await prisma_client.insert_data(
        data={"user_id": user_id, "max_budget": _EMPTY_WALLET_BUDGET, "models": []},  # mutable-ok: Prisma payloads are dict-shaped
        table_name="user",
    )


async def credit_wallet(prisma_client: "PrismaClient", user_id: str, amount: float) -> float:
    """
    Raise the account's wallet ceiling by `amount` and return the new ceiling.

    The raise is a single atomic increment, so concurrent top-ups cannot lose credit.
    A NULL ceiling (a user row created outside tokenin) is seeded with this credit
    instead, since SQL arithmetic on NULL is NULL, and that seed is itself
    conditional on the column still being NULL.
    """
    await ensure_wallet_user(prisma_client=prisma_client, user_id=user_id)
    table: Final = UserRepository(prisma_client).table

    seeded: Final = await table.update_many(
        where={"user_id": user_id, "max_budget": None},  # mutable-ok: Prisma filters are dict-shaped
        data={"max_budget": amount},  # mutable-ok: Prisma payloads are dict-shaped
    )
    if seeded:
        await _invalidate_cached_user(user_id=user_id)
        return amount

    updated: Final = await table.update(
        where={"user_id": user_id},  # mutable-ok: Prisma filters are dict-shaped
        data={"max_budget": {"increment": amount}},  # mutable-ok: Prisma payloads are dict-shaped
    )
    await _invalidate_cached_user(user_id=user_id)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account wallet row could not be credited",
        )
    return float(updated.max_budget or 0.0)


async def _invalidate_cached_user(user_id: str) -> None:
    """Drop the auth-path user cache so the new ceiling is enforced on the next request."""
    from litellm.proxy.proxy_server import user_api_key_cache

    await user_api_key_cache.async_delete_cache(key=user_id)
