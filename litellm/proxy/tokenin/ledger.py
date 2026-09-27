"""Atomic account-level prepaid holds. Admission remains dark until cost settlement is wired to every supported route."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import ROUND_CEILING, Decimal
from typing import Final
from uuid import uuid4

from fastapi import HTTPException

from litellm.proxy.utils import PrismaClient

_SCALE: Final = Decimal(1_000_000_000)
_MAX_NANO: Final = 2**63 - 1
_GRANTS_WITH_SPEND_QUERY: Final = (
    'SELECT g."idempotency_key", g."kind", g."plan_id", g."amount_nano", '
    'g."subscription_id", g."period_index", g."period_start", g."period_end", '
    'COALESCE(SUM(a."amount_nano"),0) AS "allocated_nano" '
    'FROM "LiteLLM_TokeninGrant" g LEFT JOIN "LiteLLM_TokeninAllocation" a '
    'ON a."grant_id" = g."idempotency_key" WHERE g."user_id" = $1 '
    'GROUP BY g."idempotency_key"'
)


@dataclass(frozen=True, slots=True)
class GrantBalance:
    grant_id: str
    kind: str
    plan_id: str
    amount_nano: int
    allocated_nano: int
    subscription_id: str | None
    period_index: int | None
    period_start: datetime | None
    period_end: datetime | None

    @property
    def available_nano(self) -> int:
        return max(self.amount_nano - self.allocated_nano, 0)


def to_nano_ceil(value: float, allow_zero: bool = False) -> int:
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise HTTPException(status_code=503, detail="Priced request cost is unavailable")
    amount: Final = int((Decimal(str(value)) * _SCALE).to_integral_value(rounding=ROUND_CEILING))
    if amount > _MAX_NANO or (amount == 0 and not allow_zero):
        raise HTTPException(status_code=503, detail="Priced request cost exceeds supported range")
    return amount


def _grant(row: Mapping[str, object]) -> GrantBalance:
    return GrantBalance(
        grant_id=str(row["idempotency_key"]),
        kind=str(row["kind"]),
        plan_id=str(row["plan_id"]),
        amount_nano=int(row["amount_nano"]),
        allocated_nano=int(row["allocated_nano"]),
        subscription_id=str(row["subscription_id"]) if row["subscription_id"] is not None else None,
        period_index=int(row["period_index"]) if row["period_index"] is not None else None,
        period_start=row["period_start"] if isinstance(row["period_start"], datetime) else None,
        period_end=row["period_end"] if isinstance(row["period_end"], datetime) else None,
    )


def eligible_grants(grants: Sequence[GrantBalance], now: datetime) -> tuple[GrantBalance, ...]:
    following: Final = {
        (grant.subscription_id, grant.period_index): grant
        for grant in grants
        if grant.kind == "fixed" and grant.subscription_id is not None and grant.period_index is not None
    }
    eligible: Final = (
        grant
        for grant in grants
        if grant.available_nano > 0
        and (
            grant.kind == "payg"
            or (
                grant.kind == "fixed"
                and grant.period_start is not None
                and grant.period_end is not None
                and grant.period_index is not None
                and (
                    grant.period_start <= now < grant.period_end
                    or (
                        (next_grant := following.get((grant.subscription_id, grant.period_index + 1))) is not None
                        and next_grant.period_start == grant.period_end
                        and next_grant.period_start is not None
                        and next_grant.period_end is not None
                        and next_grant.period_start <= now < next_grant.period_end
                    )
                )
            )
        )
    )
    return tuple(
        sorted(
            eligible,
            key=lambda grant: (
                grant.kind == "payg",
                grant.period_start or datetime.max,
                grant.grant_id,
            ),
        )
    )


def allocate_fifo(grants: Sequence[GrantBalance], amount_nano: int) -> tuple[tuple[str, int], ...]:
    allocations: Final[list[tuple[str, int]]] = []
    remaining = amount_nano
    for grant in grants:
        if remaining <= 0:
            break
        taken: Final = min(grant.available_nano, remaining)
        if taken > 0:
            allocations.append((grant.grant_id, taken))
            remaining -= taken  # rebind-ok: FIFO balance decreases once per bucket
    if remaining > 0:
        raise HTTPException(status_code=402, detail="Insufficient account credit")
    return tuple(allocations)


def _active_plan(grants: Sequence[GrantBalance], now: datetime) -> tuple[str, datetime | None]:
    paid: Final = (
        grant
        for grant in grants
        if grant.kind == "fixed"
        and grant.period_start is not None
        and grant.period_end is not None
        and grant.period_start <= now < grant.period_end
    )
    current: Final = tuple(paid)
    if len(current) > 1:
        raise HTTPException(status_code=503, detail="Overlapping paid plans require reconciliation")
    return (current[0].plan_id, current[0].period_start) if current else ("payg", None)


def _matching_policy(
    policies: Sequence[Mapping[str, object]], plan_id: str, now: datetime, period_start: datetime | None
) -> Mapping[str, object]:
    effective: Final = (
        policy
        for policy in policies
        if policy["plan_id"] == plan_id
        and isinstance(policy["effective_at"], datetime)
        and policy["effective_at"] <= now
        and (period_start is None or policy["effective_at"] >= period_start)
    )
    candidate_policies: Final = tuple(effective)
    if period_start is not None and not any(policy["effective_at"] == period_start for policy in candidate_policies):
        raise HTTPException(status_code=503, detail="Paid period has no matching initial policy")
    latest: Final = max(candidate_policies, key=lambda policy: policy["effective_at"], default=None)
    if latest is None:
        raise HTTPException(status_code=503, detail="Account plan policy is not active")
    return latest


async def reserve_account_request(
    prisma_client: PrismaClient,
    user_id: str,
    request_id: str,
    model: str,
    estimated_cost: float,
    *,
    key_hash: str,
) -> tuple[tuple[str, int], ...]:
    estimated_nano: Final = to_nano_ceil(estimated_cost)
    if not user_id or not request_id or not key_hash or not model:
        raise HTTPException(status_code=503, detail="Account request attribution unavailable")
    async with prisma_client.db.tx() as tx:
        account: Final = await tx.query_raw(
            'SELECT "user_id", "debt_nano" FROM "LiteLLM_TokeninAccount" WHERE "user_id" = $1 FOR UPDATE',
            user_id,
        )
        if not account:
            raise HTTPException(status_code=503, detail="Managed account is not enrolled")
        clock: Final = await tx.query_raw("SELECT clock_timestamp() AT TIME ZONE 'UTC' AS now")
        now: Final = clock[0]["now"]
        previous: Final = await tx.query_raw(
            'SELECT "user_id", "key_hash", "model", "estimated_nano", "state" '
            'FROM "LiteLLM_TokeninHold" WHERE "request_id" = $1',
            request_id,
        )
        if previous:
            if (
                previous[0]["user_id"] != user_id
                or previous[0]["key_hash"] != key_hash
                or previous[0]["model"] != model
                or int(previous[0]["estimated_nano"]) != estimated_nano
                or previous[0]["state"] != "held"
            ):
                raise HTTPException(status_code=409, detail="Request ID reused or finalized")
            existing: Final = await tx.query_raw(
                'SELECT "grant_id", "amount_nano" FROM "LiteLLM_TokeninAllocation" '
                'WHERE "request_id" = $1 ORDER BY "grant_id"',
                request_id,
            )
            return tuple((str(row["grant_id"]), int(row["amount_nano"])) for row in existing)

        raw_grants: Final = await tx.query_raw(_GRANTS_WITH_SPEND_QUERY, user_id)
        grants: Final = tuple(_grant(row) for row in raw_grants)
        debt: Final = int(account[0]["debt_nano"])
        debt_allocation: Final = allocate_fifo(eligible_grants(grants=grants, now=now), debt) if debt else ()
        if debt:
            debt_id: Final = f"tokenin-debt:{uuid4().hex}"
            await tx.execute_raw(
                'INSERT INTO "LiteLLM_TokeninHold" '
                '("request_id", "user_id", "key_hash", "model", "state", "estimated_nano", '
                '"charged_nano", "admitted_at", "settled_at") '
                "VALUES ($1,$2,'tokenin-debt','tokenin-debt','settled',$3,$3,$4,$5)",
                debt_id,
                user_id,
                debt,
                datetime(1970, 1, 1),
                now,
            )
            for grant_id, amount in debt_allocation:
                await tx.execute_raw(
                    'INSERT INTO "LiteLLM_TokeninAllocation" ("request_id", "grant_id", "amount_nano") '
                    "VALUES ($1,$2,$3)",
                    debt_id,
                    grant_id,
                    amount,
                )
            await tx.execute_raw('UPDATE "LiteLLM_TokeninAccount" SET "debt_nano" = 0 WHERE "user_id" = $1', user_id)
        debt_paid: Final = dict(debt_allocation)
        remaining_grants: Final = tuple(
            replace(grant, allocated_nano=grant.allocated_nano + debt_paid.get(grant.grant_id, 0)) for grant in grants
        )
        plan_id, period_start = _active_plan(grants=remaining_grants, now=now)
        policies: Final = await tx.query_raw(
            'SELECT "plan_id", "models", "rpm_limit", "max_parallel_requests", "effective_at" '
            'FROM "LiteLLM_TokeninPolicy" WHERE "user_id" = $1 AND "effective_at" <= $2',
            user_id,
            now,
        )
        policy: Final = _matching_policy(policies=policies, plan_id=plan_id, now=now, period_start=period_start)
        if model not in policy["models"]:
            raise HTTPException(status_code=403, detail="Model not allowed by account plan")
        limits: Final = await tx.query_raw(
            "SELECT COUNT(*) FILTER (WHERE \"admitted_at\" > $2 - INTERVAL '60 seconds') AS rpm, "
            "COUNT(*) FILTER (WHERE \"state\" IN ('held', 'uncertain')) AS concurrent "
            'FROM "LiteLLM_TokeninHold" WHERE "user_id" = $1',
            user_id,
            now,
        )
        if int(limits[0]["rpm"]) >= int(policy["rpm_limit"]):
            raise HTTPException(status_code=429, detail="Account RPM limit exceeded")
        if int(limits[0]["concurrent"]) >= int(policy["max_parallel_requests"]):
            raise HTTPException(status_code=429, detail="Account concurrency limit exceeded")
        allocation: Final = allocate_fifo(eligible_grants(grants=remaining_grants, now=now), estimated_nano)
        await tx.execute_raw(
            'INSERT INTO "LiteLLM_TokeninHold" '
            '("request_id", "user_id", "key_hash", "model", "state", "estimated_nano", "admitted_at") '
            "VALUES ($1,$2,$3,$4,'held',$5,$6)",
            request_id,
            user_id,
            key_hash,
            model,
            estimated_nano,
            now,
        )
        for grant_id, amount in allocation:
            await tx.execute_raw(
                'INSERT INTO "LiteLLM_TokeninAllocation" ("request_id", "grant_id", "amount_nano") VALUES ($1,$2,$3)',
                request_id,
                grant_id,
                amount,
            )
        return allocation


async def settle_account_request(prisma_client: PrismaClient, request_id: str, actual_cost: float) -> int:
    charged_nano: Final = to_nano_ceil(actual_cost, allow_zero=True)
    async with prisma_client.db.tx() as tx:
        identity: Final = await tx.query_raw(
            'SELECT "user_id" FROM "LiteLLM_TokeninHold" WHERE "request_id" = $1', request_id
        )
        if not identity:
            raise HTTPException(status_code=503, detail="Account reservation is missing")
        user_id: Final = identity[0]["user_id"]
        account: Final = await tx.query_raw(
            'SELECT "user_id", "debt_nano" FROM "LiteLLM_TokeninAccount" WHERE "user_id" = $1 FOR UPDATE', user_id
        )
        existing: Final = await tx.query_raw(
            'SELECT "state", "charged_nano" FROM "LiteLLM_TokeninHold" WHERE "request_id" = $1', request_id
        )
        if existing[0]["state"] == "settled":
            if int(existing[0]["charged_nano"]) != charged_nano:
                raise HTTPException(status_code=409, detail="Request cost changed after settlement")
            return charged_nano
        if existing[0]["state"] != "held":
            raise HTTPException(status_code=503, detail="Reservation needs manual reconciliation")
        allocation: Final = await tx.query_raw(
            'SELECT a."grant_id", a."amount_nano" FROM "LiteLLM_TokeninAllocation" a '
            'JOIN "LiteLLM_TokeninGrant" g ON g."idempotency_key" = a."grant_id" '
            'WHERE a."request_id" = $1 ORDER BY g."period_start" NULLS LAST, g."created_at", g."idempotency_key"',
            request_id,
        )
        remaining = charged_nano
        for entry in allocation:
            spent: Final = min(remaining, int(entry["amount_nano"]))
            await tx.execute_raw(
                'UPDATE "LiteLLM_TokeninAllocation" SET "amount_nano" = $3 WHERE "request_id" = $1 AND "grant_id" = $2',
                request_id,
                entry["grant_id"],
                spent,
            )
            remaining -= spent  # rebind-ok: refund any unused reservation after FIFO actual charge
        if remaining:
            clock: Final = await tx.query_raw("SELECT clock_timestamp() AT TIME ZONE 'UTC' AS now")
            fresh_grants: Final = await tx.query_raw(_GRANTS_WITH_SPEND_QUERY, user_id)
            eligible: Final = eligible_grants(tuple(_grant(row) for row in fresh_grants), clock[0]["now"])
            payable: Final = min(remaining, sum(grant.available_nano for grant in eligible))
            if payable:
                for grant_id, amount in allocate_fifo(eligible, payable):
                    await tx.execute_raw(
                        'INSERT INTO "LiteLLM_TokeninAllocation" ("request_id", "grant_id", "amount_nano") '
                        'VALUES ($1,$2,$3) ON CONFLICT ("request_id", "grant_id") DO UPDATE '
                        'SET "amount_nano" = "LiteLLM_TokeninAllocation"."amount_nano" + EXCLUDED."amount_nano"',
                        request_id,
                        grant_id,
                        amount,
                    )
                remaining -= payable  # rebind-ok: settle excess from currently eligible credit before debt
        if remaining:
            if int(account[0]["debt_nano"]) + remaining > _MAX_NANO:
                raise HTTPException(status_code=503, detail="Account debt exceeds supported range")
            await tx.execute_raw(
                'UPDATE "LiteLLM_TokeninAccount" SET "debt_nano" = "debt_nano" + $2 WHERE "user_id" = $1',
                user_id,
                remaining,
            )
        await tx.execute_raw(
            'UPDATE "LiteLLM_TokeninHold" SET "state" = \'settled\', '
            '"charged_nano" = $2, "settled_at" = clock_timestamp() AT TIME ZONE \'UTC\' '
            'WHERE "request_id" = $1',
            request_id,
            charged_nano,
        )
        return charged_nano


async def cancel_account_request(prisma_client: PrismaClient, request_id: str) -> bool:
    """Release a reservation whose work is known not to have billed. Unknown holds stay reserved."""
    async with prisma_client.db.tx() as tx:
        identity: Final = await tx.query_raw(
            'SELECT "user_id", "state" FROM "LiteLLM_TokeninHold" WHERE "request_id" = $1', request_id
        )
        if not identity:
            return False
        await tx.query_raw(
            'SELECT "user_id" FROM "LiteLLM_TokeninAccount" WHERE "user_id" = $1 FOR UPDATE',
            identity[0]["user_id"],
        )
        state: Final = identity[0]["state"]
        if state == "cancelled":
            return True
        if state != "held":
            return False
        await tx.execute_raw(
            'UPDATE "LiteLLM_TokeninAllocation" SET "amount_nano" = 0 WHERE "request_id" = $1', request_id
        )
        await tx.execute_raw(
            'UPDATE "LiteLLM_TokeninHold" SET "state" = \'cancelled\', "charged_nano" = 0, '
            '"settled_at" = clock_timestamp() AT TIME ZONE \'UTC\' WHERE "request_id" = $1',
            request_id,
        )
        return True


async def mark_account_request_uncertain(prisma_client: PrismaClient, request_id: str) -> bool:
    """Keep the reservation when billing outcome is unknown, so nothing is refunded speculatively."""
    async with prisma_client.db.tx() as tx:
        identity: Final = await tx.query_raw(
            'SELECT "user_id", "state" FROM "LiteLLM_TokeninHold" WHERE "request_id" = $1', request_id
        )
        if not identity:
            return False
        await tx.query_raw(
            'SELECT "user_id" FROM "LiteLLM_TokeninAccount" WHERE "user_id" = $1 FOR UPDATE',
            identity[0]["user_id"],
        )
        state: Final = identity[0]["state"]
        if state == "uncertain":
            return True
        if state != "held":
            return False
        await tx.execute_raw(
            'UPDATE "LiteLLM_TokeninHold" SET "state" = \'uncertain\' WHERE "request_id" = $1', request_id
        )
        return True
