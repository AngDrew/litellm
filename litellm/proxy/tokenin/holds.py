"""Stale account-hold reconciliation. Admission may free a latch, but this job closes the billed state."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Final, NamedTuple

from litellm._logging import verbose_proxy_logger
from litellm.proxy.tokenin.ledger import (
    HOLD_INFLIGHT_TTL_SECONDS,
    as_sql_timestamp,
    mark_account_request_uncertain_tx,
    settle_account_request_tx,
)
from litellm.proxy.utils import PrismaClient

UNCERTAIN_SETTLE_AFTER_SECONDS: Final = max(int(os.environ.get("TOKENIN_ACCOUNT_UNCERTAIN_SETTLE_AFTER_SECONDS", "86400")), 1)
HOLD_REAPER_INTERVAL_SECONDS: Final = max(int(os.environ.get("TOKENIN_ACCOUNT_HOLD_REAPER_INTERVAL_SECONDS", "300")), 60)
HOLD_OLDER_THAN_SECONDS: Final = max(int(os.environ.get("TOKENIN_ACCOUNT_HOLD_OLDER_THAN_SECONDS", "3600")), 1)
REAPER_BATCH_SIZE: Final = max(int(os.environ.get("TOKENIN_ACCOUNT_HOLD_REAPER_BATCH_SIZE", "100")), 1)
_NANO_TO_USD: Final = 1_000_000_000


class _Candidate(NamedTuple):
    request_id: str
    state: str
    estimated_nano: int
    admitted_at: datetime


def _candidate(row: Mapping[str, object]) -> _Candidate:
    from litellm.proxy.tokenin.ledger import as_naive_utc

    admitted: Final = as_naive_utc(row["admitted_at"])
    if admitted is None:
        raise ValueError("hold admitted_at is missing")
    return _Candidate(
        request_id=str(row["request_id"]),
        state=str(row["state"]),
        estimated_nano=int(row["estimated_nano"]),
        admitted_at=admitted,
    )


async def _lock_contenders(tx: Any, now: datetime, offset: int = 0) -> list[_Candidate]:
    ttl_cutoff: Final = now - timedelta(seconds=HOLD_INFLIGHT_TTL_SECONDS)
    rows: Final = await tx.query_raw(
        'SELECT "request_id", "state", "estimated_nano", "admitted_at" '
        'FROM "LiteLLM_TokeninHold" WHERE ("state" = \'held\' AND "admitted_at" < $1::timestamp) '
        "OR \"state\" = 'uncertain' "
        'ORDER BY "admitted_at" LIMIT 1 OFFSET $2',
        as_sql_timestamp(ttl_cutoff),
        offset,
    )
    return [_candidate(row) for row in rows]


async def _spend_cost(prisma_client: PrismaClient, request_id: str) -> float | None:
    rows: Final = await prisma_client.db.query_raw(
        'SELECT s."request_id", s."spend" FROM "LiteLLM_SpendLogs" s '
        "WHERE s.\"metadata\"::jsonb #>> '{spend_logs_metadata,tokenin_account_hold_id}' = $1 "
        "OR s.\"metadata\"::jsonb #>> '{user_api_key_tokenin_account_hold_id}' = $1 "
        "ORDER BY s.\"endTime\" DESC LIMIT 2",
        request_id,
    )
    costs: Final = {float(row["spend"]) for row in rows}
    return costs.pop() if len(costs) == 1 else None


async def _reconcile_one(
    prisma_client: PrismaClient, candidate: _Candidate, now: datetime
) -> bool | None:
    """Process one hold under a transaction-scoped advisory lock. ``None`` means another worker owns it."""
    if candidate.admitted_at < now - timedelta(seconds=HOLD_OLDER_THAN_SECONDS):
        verbose_proxy_logger.warning(
            "Tokenin hold %s stale state=%s age_seconds=%s",
            candidate.request_id,
            candidate.state,
            int((now - candidate.admitted_at).total_seconds()),
        )
    async with prisma_client.db.tx() as tx:
        lock: Final = await tx.query_raw("SELECT pg_try_advisory_xact_lock(hashtextextended($1, 0)) AS locked", candidate.request_id)
        if not lock or not lock[0]["locked"]:
            return None
        if candidate.state == "held":
            changed: Final = await mark_account_request_uncertain_tx(tx, candidate.request_id)
            if changed:
                verbose_proxy_logger.warning(
                    "Tokenin hold %s resolved action=uncertain cost_usd=None note=reaper-stale-held",
                    candidate.request_id,
                )
            return changed

        actual: Final = await _spend_cost(prisma_client=prisma_client, request_id=candidate.request_id)
        if actual is None and candidate.admitted_at >= now - timedelta(seconds=UNCERTAIN_SETTLE_AFTER_SECONDS):
            return False
        cost: Final = actual if actual is not None else candidate.estimated_nano / _NANO_TO_USD
        await settle_account_request_tx(tx, candidate.request_id, cost, allow_uncertain=True)
        verbose_proxy_logger.warning(
            "Tokenin hold %s resolved action=settle cost_usd=%s note=reaper-%s",
            candidate.request_id,
            json.dumps(cost),
            "spend-log" if actual is not None else "reserved-estimate",
        )
        return True


async def reconcile_account_holds(
    prisma_client: PrismaClient, *, now_override: datetime | None = None
) -> dict[str, int]:
    """Resolve stale holds while the reservation itself stays on the ledger books."""
    now: Final = now_override or datetime.utcnow()
    promoted = settled = unreconciled = 0
    cursor = 0
    while promoted + settled + unreconciled < REAPER_BATCH_SIZE:
        current = cursor
        try:
            first: Final = await _lock_contenders(prisma_client.db, now, current)
        except Exception:
            unreconciled += 1
            verbose_proxy_logger.exception("Tokenin hold reaper row selection failed at cursor=%s", current)
            break
        if not first:
            break
        candidate: Final = first[0]
        try:
            changed: Final = await _reconcile_one(prisma_client=prisma_client, candidate=candidate, now=now)
        except Exception:
            unreconciled += 1
            verbose_proxy_logger.exception(
                "Tokenin hold %s was not reconciled from spend log actual=None", candidate.request_id
            )
            cursor += 1
            continue
        if changed is not True:
            cursor += 1
            continue
        if candidate.state == "held":
            promoted += 1
        else:
            settled += 1
    return {"promoted": promoted, "settled": settled, "unreconciled": unreconciled}


async def account_hold_reaper(prisma_client: PrismaClient) -> None:
    from litellm.proxy.tokenin.accounts import account_spend_enabled

    if not account_spend_enabled():
        return
    try:
        await reconcile_account_holds(prisma_client=prisma_client)
    except Exception:
        verbose_proxy_logger.exception("Tokenin account hold reaper failed")
