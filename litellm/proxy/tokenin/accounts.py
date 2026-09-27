"""Gated Tokenin paid-period records. No request spend is authorized by this module yet."""

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from litellm.proxy.tokenin.plans import TokeninPlan, load_plans
from litellm.proxy.utils import PrismaClient

router = APIRouter()
_NANODOLLARS: Final = Decimal("1000000000")
_MAX_NANODOLLARS: Final = 2**63 - 1


def account_v2_enabled() -> bool:
    return os.environ.get("TOKENIN_ACCOUNT_V2_ENABLED") == "true"


def _service_only(request: Request) -> None:
    if not account_v2_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    secret: Final = os.environ.get("TOKENIN_ACCOUNT_SERVICE_TOKEN", "")
    if len(secret) < 32:
        raise HTTPException(status_code=503, detail="Tokenin service authentication unavailable")
    supplied: Final = request.headers.get("authorization", "")
    if not hmac.compare_digest(supplied, f"Bearer {secret}"):
        raise HTTPException(status_code=403, detail="Forbidden")


def _db() -> PrismaClient:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return prisma_client


def _plan(plan_id: str) -> TokeninPlan:
    for plan in load_plans():
        if plan.id == plan_id:
            return plan
    raise HTTPException(status_code=400, detail="Unknown plan_id")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise HTTPException(status_code=400, detail="Period timestamps require a timezone")
    converted: Final = value.astimezone(timezone.utc)
    return converted.replace(tzinfo=None, microsecond=(converted.microsecond // 1000) * 1000)


def _amount_nano(value: Decimal) -> int:
    if not value.is_finite() or value <= 0:
        raise HTTPException(status_code=400, detail="Credit must be positive and finite")
    scaled: Final = value * _NANODOLLARS
    if scaled != scaled.to_integral_value() or scaled > _MAX_NANODOLLARS:
        raise HTTPException(status_code=400, detail="Credit must have at most nine decimal places")
    return int(scaled)


def _dollars(amount_nano: int) -> Decimal:
    return Decimal(amount_nano) / _NANODOLLARS


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _fingerprint(payload: Mapping[str, object]) -> str:
    serialized: Final = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


class GrantRequest(BaseModel):
    user_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    kind: Literal["fixed", "payg"]
    subscription_id: str | None = None
    period_index: int | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
    topup_usd: Decimal | None = None


class PolicyRequest(BaseModel):
    user_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    models: list[str]
    rpm_limit: int = Field(gt=0)
    max_parallel_requests: int = Field(gt=0)
    effective_at: datetime


def _grant_details(data: GrantRequest) -> tuple[int, datetime | None, datetime | None, str]:
    plan: Final = _plan(data.plan_id)
    if data.kind == "payg":
        if (
            plan.kind != "payg"
            or data.topup_usd is None
            or any(
                field is not None
                for field in (data.subscription_id, data.period_index, data.period_start, data.period_end)
            )
        ):
            raise HTTPException(status_code=400, detail="PAYG needs a verified amount and no subscription period")
        return _amount_nano(data.topup_usd), None, None, "payg"

    if plan.kind == "payg" or data.topup_usd is not None:
        raise HTTPException(status_code=400, detail="Fixed plan uses its configured monthly credit")
    if (
        not data.subscription_id
        or data.period_index is None
        or data.period_index < 0
        or data.period_start is None
        or data.period_end is None
    ):
        raise HTTPException(status_code=400, detail="Fixed grant needs a paid subscription period")
    start: Final = _utc(data.period_start)
    end: Final = _utc(data.period_end)
    if not start < end or not 28 <= (end - start).total_seconds() / 86400 <= 32:
        raise HTTPException(status_code=400, detail="Fixed grant period must span one paid month")
    return _fixed_monthly_amount(plan), start, end, "fixed"


def _fixed_monthly_amount(plan: TokeninPlan) -> int:
    if plan.monthly_value is None and plan.budget_duration not in {"30d", "1mo"}:
        raise HTTPException(status_code=400, detail="Fixed plan lacks a monthly credit")
    try:
        monthly_credit: Final = Decimal(str(plan.monthly_value if plan.monthly_value is not None else plan.max_budget))
    except (InvalidOperation, ValueError):
        raise HTTPException(status_code=400, detail="Fixed plan lacks monthly credit")
    return _amount_nano(monthly_credit)


async def is_managed_account(prisma_client: PrismaClient, user_id: str | None) -> bool:
    if not account_v2_enabled() or user_id is None:
        return False
    try:
        rows: Final = await prisma_client.db.query_raw(
            'SELECT 1 FROM "LiteLLM_TokeninAccount" WHERE "user_id" = $1 LIMIT 1', user_id
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Account enrollment lookup unavailable") from exc
    return bool(rows)


async def require_request_admission(
    prisma_client: PrismaClient | None, user_id: str | None, team_id: str | None, route: str
) -> None:
    if not account_v2_enabled():
        return
    from litellm.proxy.auth.auth_exception_handler import DB_UNAVAILABLE_FALLBACK_USER_ID
    from litellm.proxy.auth.route_checks import RouteChecks
    from litellm.proxy.tokenin.plans import FLAT_TEAM_ID

    if not RouteChecks.is_llm_api_route(route):
        return
    if prisma_client is None or user_id == DB_UNAVAILABLE_FALLBACK_USER_ID:
        raise HTTPException(status_code=503, detail="Managed account authorization unavailable")
    if user_id is None and team_id == FLAT_TEAM_ID:
        raise HTTPException(status_code=503, detail="Unattributed Tokenin key cannot be authorized")
    if await is_managed_account(prisma_client, user_id):
        raise HTTPException(status_code=503, detail="Managed account spend is not yet enabled")


def _grant_response(row: Mapping[str, object], duplicate: bool) -> dict[str, object]:
    return {
        "user_id": row["user_id"],
        "idempotency_key": row["idempotency_key"],
        "credited": float(_dollars(int(row["amount_nano"]))),
        "plan_id": row.get("plan_id"),
        "kind": row.get("kind"),
        "subscription_id": row.get("subscription_id"),
        "period_index": row.get("period_index"),
        "period_start": _as_utc(row.get("period_start")),
        "period_end": _as_utc(row.get("period_end")),
        "duplicate": duplicate,
        "available_usd": None,
        "enforcement_active": False,
    }


@router.get("/tokenin/account/catalog", dependencies=[Depends(_service_only)], tags=["tokenin"])
async def account_catalog() -> dict[str, object]:
    plans: Final = load_plans()
    return {
        "plans": [
            {
                "id": plan.id,
                "kind": plan.kind,
                "monthly_value_usd": float(_dollars(_fixed_monthly_amount(plan))) if plan.kind == "fixed" else None,
                "rpm_limit": plan.rpm_limit,
                "max_parallel_requests": plan.max_parallel_requests,
            }
            for plan in plans
        ],
        "enforcement_active": False,
    }


@router.post("/tokenin/account/grants", dependencies=[Depends(_service_only)], tags=["tokenin"])
async def grant_account(data: GrantRequest) -> dict[str, object]:
    amount, start, end, kind = _grant_details(data)
    fingerprint: Final = _fingerprint(
        {
            "user_id": data.user_id,
            "idempotency_key": data.idempotency_key,
            "plan_id": data.plan_id,
            "kind": kind,
            "subscription_id": data.subscription_id,
            "period_index": data.period_index,
            "period_start": start.isoformat() if start is not None else None,
            "period_end": end.isoformat() if end is not None else None,
            "topup_usd": str(_dollars(amount)) if kind == "payg" else None,
        }
    )
    async with _db().db.tx() as tx:
        await tx.execute_raw(
            'INSERT INTO "LiteLLM_TokeninAccount" ("user_id") VALUES ($1) ON CONFLICT DO NOTHING', data.user_id
        )
        await tx.query_raw(
            'SELECT "user_id" FROM "LiteLLM_TokeninAccount" WHERE "user_id" = $1 FOR UPDATE', data.user_id
        )
        previous: Final = await tx.query_raw(
            'SELECT * FROM "LiteLLM_TokeninGrant" WHERE "idempotency_key" = $1', data.idempotency_key
        )
        if previous:
            if previous[0]["payload_hash"] != fingerprint:
                raise HTTPException(status_code=409, detail="Grant ID reused with a different payload")
            return _grant_response(previous[0], duplicate=True)
        if kind == "fixed":
            neighbor: Final = await tx.query_raw(
                'SELECT "period_index", "period_start", "period_end" FROM "LiteLLM_TokeninGrant" '
                'WHERE "user_id" = $1 AND "subscription_id" = $2 AND "kind" = $3 '
                'AND "period_index" IN ($4, $5)',
                data.user_id,
                data.subscription_id,
                kind,
                data.period_index - 1,
                data.period_index + 1,
            )
            for adjacent in neighbor:
                expected: Final = adjacent["period_end"] if adjacent["period_index"] == data.period_index - 1 else end
                actual: Final = start if adjacent["period_index"] == data.period_index - 1 else adjacent["period_start"]
                if expected != actual:
                    raise HTTPException(status_code=409, detail="Paid periods are not contiguous")
        inserted: Final = await tx.execute_raw(
            'INSERT INTO "LiteLLM_TokeninGrant" '
            '("idempotency_key", "user_id", "payload_hash", "plan_id", "kind", "amount_nano", '
            '"subscription_id", "period_index", "period_start", "period_end") '
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) ON CONFLICT DO NOTHING",
            data.idempotency_key,
            data.user_id,
            fingerprint,
            data.plan_id,
            kind,
            amount,
            data.subscription_id,
            data.period_index,
            start,
            end,
        )
        if inserted != 1:
            raise HTTPException(status_code=409, detail="Paid period already granted")
    return _grant_response(
        {
            "user_id": data.user_id,
            "idempotency_key": data.idempotency_key,
            "amount_nano": amount,
            "plan_id": data.plan_id,
            "kind": kind,
            "subscription_id": data.subscription_id,
            "period_index": data.period_index,
            "period_start": start,
            "period_end": end,
        },
        duplicate=False,
    )


@router.get("/tokenin/account/grants/{idempotency_key}", dependencies=[Depends(_service_only)], tags=["tokenin"])
async def get_grant(idempotency_key: str, user_id: str = Query(min_length=1)) -> dict[str, object]:
    rows: Final = await _db().db.query_raw(
        'SELECT * FROM "LiteLLM_TokeninGrant" WHERE "idempotency_key" = $1 AND "user_id" = $2',
        idempotency_key,
        user_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Grant not found")
    return _grant_response(rows[0], duplicate=True)


@router.post("/tokenin/account/policy", dependencies=[Depends(_service_only)], tags=["tokenin"])
async def set_account_policy(data: PolicyRequest) -> dict[str, object]:
    _plan(data.plan_id)
    effective: Final = _utc(data.effective_at)
    if len(data.models) != len(set(data.models)) or any(
        not model or model in {"*", "all-team-models"} for model in data.models
    ):
        raise HTTPException(status_code=400, detail="Models must be explicit unique aliases")
    fingerprint: Final = _fingerprint(
        {
            "user_id": data.user_id,
            "idempotency_key": data.idempotency_key,
            "plan_id": data.plan_id,
            "models": data.models,
            "rpm_limit": data.rpm_limit,
            "max_parallel_requests": data.max_parallel_requests,
            "effective_at": effective.isoformat(),
        }
    )
    async with _db().db.tx() as tx:
        await tx.execute_raw(
            'INSERT INTO "LiteLLM_TokeninAccount" ("user_id") VALUES ($1) ON CONFLICT DO NOTHING', data.user_id
        )
        await tx.query_raw(
            'SELECT "user_id" FROM "LiteLLM_TokeninAccount" WHERE "user_id" = $1 FOR UPDATE', data.user_id
        )
        previous: Final = await tx.query_raw(
            'SELECT * FROM "LiteLLM_TokeninPolicy" WHERE "idempotency_key" = $1', data.idempotency_key
        )
        if previous:
            if previous[0]["payload_hash"] != fingerprint:
                raise HTTPException(status_code=409, detail="Policy ID reused with a different payload")
            return _policy_response(previous[0], duplicate=True)
        await tx.execute_raw(
            'INSERT INTO "LiteLLM_TokeninPolicy" '
            '("idempotency_key", "user_id", "payload_hash", "plan_id", "models", '
            '"rpm_limit", "max_parallel_requests", "effective_at") '
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
            data.idempotency_key,
            data.user_id,
            fingerprint,
            data.plan_id,
            data.models,
            data.rpm_limit,
            data.max_parallel_requests,
            effective,
        )
    return _policy_response(
        {
            "user_id": data.user_id,
            "idempotency_key": data.idempotency_key,
            "plan_id": data.plan_id,
            "models": data.models,
            "rpm_limit": data.rpm_limit,
            "max_parallel_requests": data.max_parallel_requests,
            "effective_at": effective,
        },
        duplicate=False,
    )


def _policy_response(row: Mapping[str, object], duplicate: bool) -> dict[str, object]:
    return {
        "user_id": row["user_id"],
        "idempotency_key": row["idempotency_key"],
        "duplicate": duplicate,
        "plan_id": row["plan_id"],
        "models": row["models"],
        "rpm_limit": row["rpm_limit"],
        "max_parallel_requests": row["max_parallel_requests"],
        "effective_at": _as_utc(row["effective_at"]),
        "enforcement_active": False,
    }


@router.get("/tokenin/account/summary", dependencies=[Depends(_service_only)], tags=["tokenin"])
async def account_summary(user_id: str = Query(min_length=1)) -> dict[str, object]:
    db: Final = _db().db
    grants: Final = await db.query_raw(
        'SELECT "idempotency_key", "plan_id", "kind", "amount_nano", "subscription_id", '
        '"period_index", "period_start", "period_end" FROM "LiteLLM_TokeninGrant" '
        'WHERE "user_id" = $1 ORDER BY "period_start" NULLS LAST, "created_at", "idempotency_key"',
        user_id,
    )
    policies: Final = await db.query_raw(
        'SELECT "idempotency_key", "plan_id", "models", "rpm_limit", '
        '"max_parallel_requests", "effective_at" FROM "LiteLLM_TokeninPolicy" '
        'WHERE "user_id" = $1 ORDER BY "effective_at", "created_at", "idempotency_key"',
        user_id,
    )
    if not grants and not policies and not await is_managed_account(_db(), user_id):
        raise HTTPException(status_code=404, detail="Account not found")
    return {
        "user_id": user_id,
        "grants": [
            {
                **row,
                "period_start": _as_utc(row["period_start"]),
                "period_end": _as_utc(row["period_end"]),
                "amount_usd": float(_dollars(int(row["amount_nano"]))),
            }
            for row in grants
        ],
        "policies": [_policy_response(row, duplicate=False) for row in policies],
        "available_usd": None,
        "enforcement_active": False,
    }
