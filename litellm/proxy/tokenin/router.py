"""
TOKENIN KEY PROVISIONING

/tokenin/key/generate
/tokenin/key/update

Wraps the existing key-management helpers with plan-derived params so the
platform sends {plan_id, action} instead of computed rpm/budget fields.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.key_management_endpoints import (
    UpdateKeyRequest,
    _get_and_validate_existing_key,
    _process_single_key_update,
    generate_key_helper_fn,
)
from litellm.proxy.tokenin.plans import (
    KEY_MODELS,
    _FLAT_TEAM_ID,
    get_plan_by_id,
    is_payg,
    key_budget_duration,
    key_duration,
)

router = APIRouter()


class TokeninGenerateRequest(LiteLLMPydanticObjectBase):
    user_id: str
    plan_id: str
    alias: str
    payg_initial_balance: Optional[float] = None


class TokeninUpdateRequest(LiteLLMPydanticObjectBase):
    key: str
    plan_id: str
    action: Literal["extend", "payg_topup"]
    topup_usd: Optional[float] = None


@router.post(
    "/tokenin/key/generate",
    tags=["tokenin"],
    dependencies=[Depends(user_api_key_auth)],
)
async def tokenin_generate_key(
    data: TokeninGenerateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    plan = get_plan_by_id(data.plan_id)

    max_budget = (
        data.payg_initial_balance if (is_payg(plan) and data.payg_initial_balance is not None) else plan.max_budget
    )

    response = await generate_key_helper_fn(
        request_type="key",
        user_id=data.user_id,
        team_id=_FLAT_TEAM_ID,
        models=KEY_MODELS,
        key_alias=data.alias,
        key_max_budget=max_budget,
        key_budget_duration=key_budget_duration(plan),
        rpm_limit=plan.rpm_limit,
        max_parallel_requests=plan.max_parallel_requests,
        duration=key_duration(plan),
        table_name="key",
    )

    return {
        "token_id": response.get("token_id"),
        "key": response.get("token"),
        "key_alias": response.get("key_alias"),
        "key_name": response.get("key_name"),
        "expires": response.get("expires"),
        "created_at": response.get("created_at"),
    }


@router.post(
    "/tokenin/key/update",
    tags=["tokenin"],
    dependencies=[Depends(user_api_key_auth)],
)
async def tokenin_update_key(
    data: TokeninUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    from litellm.proxy.proxy_server import (
        llm_router,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
        user_custom_key_update,
    )

    plan = get_plan_by_id(data.plan_id)
    existing = await _get_and_validate_existing_key(token=data.key, prisma_client=prisma_client)

    if data.action == "extend":
        _refuse_unexpired(plan, existing)
        update = UpdateKeyRequest(
            key=data.key,
            max_budget=plan.max_budget,
            budget_duration=key_budget_duration(plan),
            rpm_limit=plan.rpm_limit,
            max_parallel_requests=plan.max_parallel_requests,
            duration=key_duration(plan),
        )
    elif data.action == "payg_topup":
        if not is_payg(plan):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="payg_topup is only valid for payg plans",
            )
        if data.topup_usd is None or data.topup_usd <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="topup_usd must be a positive number",
            )
        current_cap = _current_payg_cap(existing.model_dump())
        update = UpdateKeyRequest(
            key=data.key,
            max_budget=current_cap + data.topup_usd,
            rpm_limit=plan.rpm_limit,
            max_parallel_requests=plan.max_parallel_requests,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="action must be extend or payg_topup",
        )

    return await _process_single_key_update(
        update_key_request=update,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by=None,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
        llm_router=llm_router,
        user_custom_key_update=user_custom_key_update,
        existing_key_row=existing,
    )


def _refuse_unexpired(plan, existing) -> None:
    if is_payg(plan):
        return
    expires = getattr(existing, "expires", None)
    if expires is None:
        return
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires > datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This key is still active. You can only extend a key after it has expired.",
        )


def _current_payg_cap(key_dict: Dict[str, Any]) -> float:
    limits = key_dict.get("budget_limits")
    windows: list = []
    if isinstance(limits, str):
        import json

        try:
            windows = json.loads(limits) or []
        except (ValueError, TypeError):
            windows = []
    elif isinstance(limits, list):
        windows = limits

    for window in windows:
        if not isinstance(window, dict):
            continue
        if not window.get("budget_duration"):
            return float(window.get("max_budget") or 0)

    flat = key_dict.get("max_budget")
    return float(flat) if flat is not None else 0.0
