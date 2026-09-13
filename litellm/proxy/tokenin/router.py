"""
TOKENIN KEY PROVISIONING AND ACCOUNT WALLET

/tokenin/key/generate   issue a key for an account. Repeatable and harmless: it never
                        moves credit, so a customer can hold any number of keys.
/tokenin/wallet/topup   add a purchase to the account wallet. Touches no key.
/tokenin/key/update     compatibility shim for callers still posting key-oriented
                        purchase intents. It resolves the account from the key, credits
                        the wallet, and mutates nothing on the key.

Credit lives on the account wallet (the user row's max_budget), never on a key: no key
gets a max_budget, a budget window, or an expiry. Enforcement for the account's flat
-team keys is general_settings.apply_user_budget_to_team_keys.
"""

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Final, Literal

from fastapi import APIRouter, Depends, HTTPException, status

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _get_and_validate_existing_key,
    generate_key_helper_fn,
)
from litellm.proxy.tokenin.plans import (
    FLAT_TEAM_ID,
    KEY_MODELS,
    TokeninPlan,
    get_plan_by_id,
    is_payg,
)
from litellm.proxy.tokenin.wallet import credit_wallet, ensure_wallet_user
from litellm.proxy.utils import PrismaClient

router = APIRouter()


class TokeninGenerateRequest(LiteLLMPydanticObjectBase):
    user_id: str
    alias: str
    plan_id: str | None = None


class TokeninWalletTopupRequest(LiteLLMPydanticObjectBase):
    user_id: str
    plan_id: str
    topup_usd: float | None = None


class TokeninUpdateRequest(LiteLLMPydanticObjectBase):
    key: str
    plan_id: str
    action: Literal["extend", "payg_topup"]
    topup_usd: float | None = None


class TokeninGeneratedKey(LiteLLMPydanticObjectBase):
    """The key a customer is issued. Expiry and budgets are absent by design: the wallet holds the balance."""

    token_id: str
    key: str
    key_alias: str | None = None
    key_name: str | None = None
    expires: datetime | None = None
    created_at: datetime | None = None


class TokeninWalletCredit(LiteLLMPydanticObjectBase):
    user_id: str
    credited: float
    max_budget: float


@router.post(
    "/tokenin/key/generate",
    tags=["tokenin"],
    dependencies=[Depends(user_api_key_auth)],
)
async def tokenin_generate_key(data: TokeninGenerateRequest) -> TokeninGeneratedKey:
    """Issue a key that spends from the account wallet. No credit moves here, so this is safe to repeat."""
    prisma_client: Final = _prisma_client_or_500()
    limits: Final = get_plan_by_id(data.plan_id)

    await ensure_wallet_user(prisma_client=prisma_client, user_id=data.user_id)

    response: Final[Mapping[str, object]] = await generate_key_helper_fn(
        request_type="key",
        user_id=data.user_id,
        team_id=FLAT_TEAM_ID,
        models=KEY_MODELS,
        key_alias=data.alias,
        rpm_limit=limits.rpm_limit,
        max_parallel_requests=limits.max_parallel_requests,
        table_name="key",
    )

    return TokeninGeneratedKey.model_validate({**response, "key": response.get("token")})


@router.post(
    "/tokenin/wallet/topup",
    tags=["tokenin"],
    dependencies=[Depends(user_api_key_auth)],
)
async def tokenin_wallet_topup(data: TokeninWalletTopupRequest) -> TokeninWalletCredit:
    """Credit the account wallet with one purchase. Creates, extends, and mutates no key."""
    prisma_client: Final = _prisma_client_or_500()
    plan: Final = get_plan_by_id(data.plan_id)

    credit: Final = _purchase_credit(plan=plan, topup_usd=data.topup_usd)
    wallet_budget: Final = await credit_wallet(
        prisma_client=prisma_client,
        user_id=data.user_id,
        amount=credit,
    )

    return TokeninWalletCredit(user_id=data.user_id, credited=credit, max_budget=wallet_budget)


@router.post(
    "/tokenin/key/update",
    tags=["tokenin"],
    dependencies=[Depends(user_api_key_auth)],
)
async def tokenin_update_key(data: TokeninUpdateRequest) -> TokeninWalletCredit:
    """Compatibility shim: `key` only resolves the account. Prefer POST /tokenin/wallet/topup."""
    prisma_client: Final = _prisma_client_or_500()
    plan: Final = get_plan_by_id(data.plan_id)
    existing: Final[LiteLLM_VerificationToken] = await _get_and_validate_existing_key(
        token=data.key,
        prisma_client=prisma_client,
    )

    user_id: Final = existing.user_id
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This key is not linked to an account wallet",
        )

    credit: Final = _legacy_action_credit(plan=plan, action=data.action, topup_usd=data.topup_usd)
    wallet_budget: Final = await credit_wallet(
        prisma_client=prisma_client,
        user_id=user_id,
        amount=credit,
    )

    return TokeninWalletCredit(user_id=user_id, credited=credit, max_budget=wallet_budget)


def _purchase_credit(plan: TokeninPlan, topup_usd: float | None) -> float:
    """Credit one purchase carries: the catalog amount for a fixed plan.

    Pay-as-you-go has no catalog price, so its credit is the payment amount the caller
    supplies, and keeping that amount idempotent stays the platform's responsibility.
    """
    if is_payg(plan):
        if topup_usd is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"topup_usd is required for the payg plan '{plan.id}'",
            )
        return _positive_amount(value=topup_usd, field="topup_usd")

    if topup_usd is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"topup_usd is only valid for payg plans, not '{plan.id}'",
        )
    return _configured_credit(plan=plan)


def _legacy_action_credit(plan: TokeninPlan, action: str, topup_usd: float | None) -> float:
    """The shim's `action` only separates a catalog purchase from a pay-as-you-go payment."""
    if action == "extend":
        if is_payg(plan):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="extend is not valid for payg plans",
            )
        if topup_usd is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"topup_usd is only valid for payg plans, not '{plan.id}'",
            )
        return _configured_credit(plan=plan)

    if not is_payg(plan):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="payg_topup is only valid for payg plans",
        )
    return _purchase_credit(plan=plan, topup_usd=topup_usd)


def _configured_credit(plan: TokeninPlan) -> float:
    return _positive_amount(value=float(plan.max_budget or 0.0), field=f"plan '{plan.id}' credit")


def _positive_amount(value: float, field: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field} must be a positive number",
        )
    return value


def _prisma_client_or_500() -> PrismaClient:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No DB connected",
        )
    return prisma_client
