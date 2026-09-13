"""
TOKENIN KEY PROVISIONING

/tokenin/key/generate
/tokenin/key/update

Credit lives on the account wallet, never on a key: no key gets a max_budget, a
budget window, or an expiry. A purchase is one /generate call, which credits the
account with the plan amount and returns a key. An extension or top-up is one
/update call against any key the account owns, which the call only uses to resolve
the account.
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
    plan_id: str
    alias: str
    payg_initial_balance: float | None = None


class TokeninUpdateRequest(LiteLLMPydanticObjectBase):
    key: str
    plan_id: str
    action: Literal["extend", "payg_topup"]
    topup_usd: float | None = None


class TokeninGeneratedKey(LiteLLMPydanticObjectBase):
    """The key a purchase returns. Expiry and budgets are absent by design: the wallet holds the balance."""

    token_id: str
    key: str
    key_alias: str | None = None
    key_name: str | None = None
    expires: datetime | None = None
    created_at: datetime | None = None


class TokeninWalletCredit(LiteLLMPydanticObjectBase):
    action: Literal["extend", "payg_topup"]
    user_id: str
    credited: float
    max_budget: float


@router.post(
    "/tokenin/key/generate",
    tags=["tokenin"],
    dependencies=[Depends(user_api_key_auth)],
)
async def tokenin_generate_key(data: TokeninGenerateRequest) -> TokeninGeneratedKey:
    prisma_client: Final = _prisma_client_or_500()
    plan: Final = get_plan_by_id(data.plan_id)

    purchase_credit: Final = _purchase_credit(plan=plan, payg_initial_balance=data.payg_initial_balance)
    if purchase_credit > 0:
        await credit_wallet(prisma_client=prisma_client, user_id=data.user_id, amount=purchase_credit)
    else:
        # A pay-as-you-go purchase can arrive before its payment, but its key must still
        # resolve to a wallet, otherwise the account would be unconstrained until top-up.
        await ensure_wallet_user(prisma_client=prisma_client, user_id=data.user_id)

    response: Final[Mapping[str, object]] = await generate_key_helper_fn(
        request_type="key",
        user_id=data.user_id,
        team_id=FLAT_TEAM_ID,
        models=KEY_MODELS,
        key_alias=data.alias,
        rpm_limit=plan.rpm_limit,
        max_parallel_requests=plan.max_parallel_requests,
        table_name="key",
    )

    return TokeninGeneratedKey.model_validate({**response, "key": response.get("token")})


@router.post(
    "/tokenin/key/update",
    tags=["tokenin"],
    dependencies=[Depends(user_api_key_auth)],
)
async def tokenin_update_key(data: TokeninUpdateRequest) -> TokeninWalletCredit:
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

    credit: Final = _topup_amount(plan=plan, action=data.action, topup_usd=data.topup_usd)
    wallet_budget: Final = await credit_wallet(prisma_client=prisma_client, user_id=user_id, amount=credit)

    return TokeninWalletCredit(
        action=data.action,
        user_id=user_id,
        credited=credit,
        max_budget=wallet_budget,
    )


def _purchase_credit(plan: TokeninPlan, payg_initial_balance: float | None) -> float:
    """Wallet credit a purchase of `plan` carries.

    A fixed plan is worth its configured amount. Pay-as-you-go has no catalog price,
    so its credit is the payment amount the caller supplies.
    """
    if not is_payg(plan):
        return _configured_credit(plan=plan)
    if payg_initial_balance is None:
        return 0.0
    return _positive_amount(value=payg_initial_balance, field="payg_initial_balance")


def _topup_amount(plan: TokeninPlan, action: str, topup_usd: float | None) -> float:
    if action == "extend":
        return _configured_credit(plan=plan)
    if not is_payg(plan):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="payg_topup is only valid for payg plans",
        )
    if topup_usd is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="topup_usd must be a positive number",
        )
    return _positive_amount(value=topup_usd, field="topup_usd")


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
