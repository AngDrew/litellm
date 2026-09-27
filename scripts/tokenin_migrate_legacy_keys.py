"""One-time carry-over of legacy per-key budgets into non-expiring account PAYG grants.

Legacy keys cap a key; the account ledger caps the account. A legacy purchase has no
reliable paid anniversary, so the remaining value of every active, unexpired key is
carried over as a single non-expiring PAYG grant per account instead of a fabricated
paid period.

Amount math: one reconciled instant, ``sum(max(0, max_budget - spend))`` over keys that
are active and unexpired. Expired keys contribute nothing, no purchase anniversary is
inferred, and a key's reset window is never treated as fresh unused allowance - carry is
what the key shows as remaining right now.

Every account with active legacy keys is enrolled, even when it carries nothing. An
enrolled account is governed by the ledger, so leaving a spent-out key cap in place would
double-constrain a key whose owner has just bought credit on the account.

Idempotency: the grant id and payload hash are fixed, so a re-run inserts nothing and
reports the existing amount. A corrected amount needs a new idempotency key; never reuse
an idempotency key with a changed payload.

Dry-run by default. Nothing is written without --apply, the target database must come
from TOKENIN_MIGRATION_DATABASE_URL, and a non-local host needs both --force-prod and
TOKENIN_MIGRATION_ALLOW_PROD=true. Run it only inside the cutover window.

    TOKENIN_MIGRATION_DATABASE_URL=postgresql://... python scripts/tokenin_migrate_legacy_keys.py
    TOKENIN_MIGRATION_DATABASE_URL=postgresql://... python scripts/tokenin_migrate_legacy_keys.py \\
        --apply --clear-legacy-key-budgets --report /tmp/tokenin-carryover.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal
from typing import Final, Protocol
from urllib.parse import urlparse

NANO: Final = Decimal(1_000_000_000)
GRANT_KIND: Final = "payg"
GRANT_PLAN: Final = "payg"
# Fixed hash: the carried amount is derived, never caller-supplied, so a re-run with a
# different result is a reconciliation finding, not a payload change to accept.
PAYLOAD_HASH: Final = "legacy-carry:v1"
_LOCAL_HOSTS: Final = frozenset({"", "localhost", "127.0.0.1", "::1", "[::1]"})

_LEGACY_KEYS_QUERY: Final = (
    'SELECT "token", "user_id", "max_budget", "spend", "expires" FROM "LiteLLM_VerificationToken" '
    'WHERE "user_id" IS NOT NULL ORDER BY "user_id", "token"'
)
_CLEAR_LEGACY_KEY_BUDGETS_SQL: Final = (
    'UPDATE "LiteLLM_VerificationToken" SET "max_budget" = NULL, "budget_duration" = NULL, '
    '"budget_reset_at" = NULL, "budget_limits" = NULL, "expires" = NULL, "updated_at" = $2 '
    'WHERE "token" = ANY($1)'
)


class Client(Protocol):
    async def query_raw(self, sql: str, *values: object) -> list[Mapping[str, object]]: ...

    async def execute_raw(self, sql: str, *values: object) -> int: ...


@dataclass
class AccountPlan:
    user_id: str
    grant_id: str
    amount_nano: int
    keys: list[str] = field(default_factory=list)
    active_keys: list[str] = field(default_factory=list)
    has_active_keys: bool = False
    action: str = "would_create"
    detail: str = ""


def to_nano(value: Decimal) -> int:
    return int((value * NANO).to_integral_value(rounding=ROUND_CEILING))


def is_active(row: Mapping[str, object], now: datetime) -> bool:
    expires: Final = row.get("expires")
    return not isinstance(expires, datetime) or expires > now


def carry_over(rows: Sequence[Mapping[str, object]], now: datetime) -> tuple[Decimal, list[str], list[str]]:
    """Return (carry, counted keys, skipped keys) for one account's legacy keys."""
    carry: Final = Decimal(0)
    counted: Final[list[str]] = []
    skipped: Final[list[str]] = []
    for row in rows:
        token: Final = str(row["token"])
        budget: Final = row.get("max_budget")
        if not is_active(row=row, now=now):
            skipped.append(f"{token}:expired")
            continue
        if budget is None:
            skipped.append(f"{token}:no-key-budget")
            continue
        remaining: Final = Decimal(str(budget)) - Decimal(str(row.get("spend") or 0))
        if remaining <= 0:
            skipped.append(f"{token}:no-remaining-value")
            continue
        counted.append(token)
        carry += remaining
    return carry, counted, skipped


def plan_accounts(
    rows: Sequence[Mapping[str, object]], existing: Mapping[str, int], now: datetime
) -> list[AccountPlan]:
    grouped: Final[dict[str, list[Mapping[str, object]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["user_id"]), []).append(row)
    plans: Final[list[AccountPlan]] = []
    for user_id, account_rows in sorted(grouped.items()):
        active: Final = [str(row["token"]) for row in account_rows if is_active(row=row, now=now)]
        carry, counted, skipped = carry_over(rows=account_rows, now=now)
        grant_id: Final = f"legacy-carry:{user_id}"
        if carry <= 0:
            plans.append(
                AccountPlan(
                    user_id=user_id,
                    grant_id=grant_id,
                    amount_nano=0,
                    keys=counted,
                    active_keys=active,
                    has_active_keys=bool(active),
                    action="skipped",
                    detail="no remaining value" + (f"; {'; '.join(skipped)}" if skipped else ""),
                )
            )
            continue
        amount_nano: Final = to_nano(carry)
        recorded: Final = existing.get(grant_id)
        if recorded is None:
            action, detail = "would_create", f"carry ${carry}"
        elif recorded == amount_nano:
            action, detail = "exists", "already carried over"
        else:
            action, detail = "conflict", f"recorded {recorded} != recomputed {amount_nano}"
        plans.append(
            AccountPlan(
                user_id=user_id,
                grant_id=grant_id,
                amount_nano=amount_nano,
                keys=counted,
                active_keys=active,
                has_active_keys=bool(active),
                action=action,
                detail=detail + (f"; {'; '.join(skipped)}" if skipped else ""),
            )
        )
    return plans


def refuse_unsafe_target(database_url: str, force_prod: bool) -> str | None:
    host: Final = urlparse(database_url).hostname or ""
    if host in _LOCAL_HOSTS:
        return None
    if not force_prod:
        return f"Refusing non-local database host '{host}' without --force-prod"
    if os.environ.get("TOKENIN_MIGRATION_ALLOW_PROD") != "true":
        return "Refusing non-local database host: --force-prod also requires TOKENIN_MIGRATION_ALLOW_PROD=true"
    return None


async def migrate(
    client: Client,
    *,
    apply: bool,
    clear_legacy_key_budgets: bool,
    now: datetime | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, object]:
    moment: Final = now or datetime.now(timezone.utc).replace(tzinfo=None)
    rows: Final = await client.query_raw(_LEGACY_KEYS_QUERY)
    existing_rows: Final = await client.query_raw(
        'SELECT "idempotency_key", "amount_nano" FROM "LiteLLM_TokeninGrant" '
        "WHERE \"idempotency_key\" LIKE 'legacy-carry:%'"
    )
    existing: Final = {str(row["idempotency_key"]): int(row["amount_nano"]) for row in existing_rows}
    plans: Final = plan_accounts(rows=rows, existing=existing, now=moment)

    for plan in plans:
        log(
            f"{plan.action:11} {plan.user_id} grant={plan.grant_id} amount={plan.amount_nano} "
            f"active_keys={len(plan.active_keys)} credited_keys={len(plan.keys)} {plan.detail}"
        )

    if apply:
        for plan in plans:
            if plan.action == "conflict" or not plan.has_active_keys:
                continue
            await client.execute_raw(
                'INSERT INTO "LiteLLM_TokeninAccount" ("user_id") VALUES ($1) ON CONFLICT DO NOTHING', plan.user_id
            )
            if plan.amount_nano > 0:
                await client.execute_raw(
                    'INSERT INTO "LiteLLM_TokeninGrant" '
                    '("idempotency_key", "user_id", "payload_hash", "plan_id", "kind", "amount_nano") '
                    'VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT ("idempotency_key") DO NOTHING',
                    plan.grant_id,
                    plan.user_id,
                    PAYLOAD_HASH,
                    GRANT_PLAN,
                    GRANT_KIND,
                    plan.amount_nano,
                )
            if clear_legacy_key_budgets:
                await client.execute_raw(_CLEAR_LEGACY_KEY_BUDGETS_SQL, plan.active_keys, moment)

    summary: Final = {
        "generated_at": moment.isoformat(),
        "applied": apply,
        "cleared_legacy_key_budgets": bool(apply and clear_legacy_key_budgets),
        "accounts": len(plans),
        "keys": sum(len(plan.keys) for plan in plans),
        "active_keys": sum(len(plan.active_keys) for plan in plans),
        "carry_usd": str(sum(Decimal(plan.amount_nano) / NANO for plan in plans if plan.action != "conflict")),
        "created": sum(plan.action == "would_create" for plan in plans),
        "existing": sum(plan.action == "exists" for plan in plans),
        "skipped": sum(plan.action == "skipped" for plan in plans),
        "conflicts": [{"user_id": plan.user_id, "detail": plan.detail} for plan in plans if plan.action == "conflict"],
        "accounts_detail": [
            {
                "user_id": plan.user_id,
                "grant_id": plan.grant_id,
                "amount_nano": plan.amount_nano,
                "action": plan.action,
                "keys": plan.keys,
                "active_keys": plan.active_keys,
                "detail": plan.detail,
            }
            for plan in plans
        ],
    }
    if summary["conflicts"]:
        log(f"AUDIT conflicts={json.dumps(summary['conflicts'])}")
    log(f"AUDIT {json.dumps({key: summary[key] for key in summary if key != 'accounts_detail'})}")
    return summary


async def _run(arguments: argparse.Namespace) -> int:
    database_url: Final = os.environ.get("TOKENIN_MIGRATION_DATABASE_URL", "")
    if not database_url:
        print(  # noqa: T201 - the script reports to the operator's terminal
            "TOKENIN_MIGRATION_DATABASE_URL is required; refusing to guess a target database", file=sys.stderr
        )
        return 2
    refusal: Final = refuse_unsafe_target(database_url=database_url, force_prod=arguments.force_prod)
    if refusal:
        print(refusal, file=sys.stderr)  # noqa: T201 - operator-facing refusal
        return 2
    if not arguments.apply:
        print("DRY RUN: re-run with --apply to write carry-over grants")  # noqa: T201 - operator-facing notice

    os.environ["DATABASE_URL"] = database_url
    from prisma import Prisma

    db: Final = Prisma()
    await db.connect()
    try:
        summary: Final = await migrate(
            client=db,
            apply=bool(arguments.apply),
            clear_legacy_key_budgets=bool(arguments.clear_legacy_key_budgets),
        )
    finally:
        await db.disconnect()
    if arguments.report:
        with open(arguments.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    return 1 if summary["conflicts"] else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the carry-over grants (default: dry run)")
    parser.add_argument(
        "--clear-legacy-key-budgets",
        action="store_true",
        help="With --apply, also clear max_budget/budget_duration/budget_reset_at/budget_limits/expires",
    )
    parser.add_argument("--force-prod", action="store_true", help="Allow a non-local database host")
    parser.add_argument("--report", default=None, help="Write the JSON summary to this path")
    return parser


if __name__ == "__main__":
    sys.exit(asyncio.run(_run(arguments=_parser().parse_args())))
