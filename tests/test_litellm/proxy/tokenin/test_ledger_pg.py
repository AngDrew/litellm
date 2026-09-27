"""Optional isolated PostgreSQL check: TOKENIN_TEST_DATABASE_URL must target local tokenin_local."""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import psycopg
import pytest
from fastapi import HTTPException
from psycopg.rows import dict_row

from litellm.proxy.tokenin.ledger import reserve_account_request, settle_account_request


class PgClient:
    """Use existing psycopg to run the ledger's Prisma raw SQL against a real local database."""

    def __init__(self, conn: psycopg.AsyncConnection) -> None:
        self.conn = conn
        self.transaction = None

    def tx(self) -> PgClient:
        return self

    async def __aenter__(self) -> PgClient:
        self.transaction = self.conn.transaction()
        await self.transaction.__aenter__()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.transaction.__aexit__(exc_type, exc, traceback)

    async def query_raw(self, sql: str, *values: object) -> list[dict]:
        params = [values[int(match.group(1)) - 1] for match in re.finditer(r"\$(\d+)", sql)]
        query = re.sub(r"\$\d+", "%s", sql)
        async with self.conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, params)
            return await cursor.fetchall()

    async def execute_raw(self, sql: str, *values: object) -> int:
        params = [values[int(match.group(1)) - 1] for match in re.finditer(r"\$(\d+)", sql)]
        query = re.sub(r"\$\d+", "%s", sql)
        async with self.conn.cursor() as cursor:
            await cursor.execute(query, params)
            return cursor.rowcount


@pytest.mark.asyncio
async def test_real_postgres_fifo_locking_settlement_and_debt() -> None:
    url = os.getenv("TOKENIN_TEST_DATABASE_URL")
    if url is None:
        pytest.skip("Set TOKENIN_TEST_DATABASE_URL for an isolated local PostgreSQL database")
    parsed = urlparse(url)
    if parsed.hostname != "127.0.0.1" or parsed.path != "/tokenin_local" or os.getenv("DATABASE_URL") != url:
        pytest.fail("Refusing ledger integration test outside explicitly selected local tokenin_local database")
    first = PgClient(await psycopg.AsyncConnection.connect(url, autocommit=True))
    second = PgClient(await psycopg.AsyncConnection.connect(url, autocommit=True))
    try:
        for table in (
            "LiteLLM_TokeninAllocation",
            "LiteLLM_TokeninHold",
            "LiteLLM_TokeninPolicy",
            "LiteLLM_TokeninGrant",
            "LiteLLM_TokeninAccount",
        ):
            await first.execute_raw(f'DROP TABLE IF EXISTS "{table}" CASCADE')
        migration_root = Path(__file__).resolve().parents[4] / "litellm-proxy-extras/litellm_proxy_extras/migrations"
        for migration in ("20260927000000_add_tokenin_account_grants", "20260927010000_add_tokenin_account_holds"):
            sql = (migration_root / migration / "migration.sql").read_text()
            for statement in sql.split(";"):
                if statement.strip():
                    await first.execute_raw(statement)
        now = datetime.utcnow().replace(microsecond=0)
        old_start = now - timedelta(days=61)
        current_start = now - timedelta(days=30)
        current_end = now + timedelta(days=1)
        for user_id in ("shared", "debt"):
            await first.execute_raw('INSERT INTO "LiteLLM_TokeninAccount" ("user_id") VALUES ($1)', user_id)
            await first.execute_raw(
                'INSERT INTO "LiteLLM_TokeninPolicy" '
                '("idempotency_key", "user_id", "payload_hash", "plan_id", "models", "rpm_limit", '
                '"max_parallel_requests", "effective_at") '
                "VALUES ($1,$2,'trusted','medium',ARRAY['allowed'],10,2,$3)",
                f"{user_id}-policy",
                user_id,
                current_start,
            )
        for grant_id, amount, start, end, index, user_id in (
            ("old", 5_000_000_000, old_start, current_start, 0, "shared"),
            ("current", 5_000_000_000, current_start, current_end, 1, "shared"),
            ("debt-paid", 1_000_000_000, current_start, current_end, 1, "debt"),
        ):
            await first.execute_raw(
                'INSERT INTO "LiteLLM_TokeninGrant" '
                '("idempotency_key", "user_id", "payload_hash", "plan_id", "kind", '
                '"amount_nano", "subscription_id", "period_index", "period_start", "period_end") '
                "VALUES ($1,$2,'trusted','medium','fixed',$3,$4,$5,$6,$7)",
                grant_id,
                user_id,
                amount,
                user_id,
                index,
                start,
                end,
            )
        one, two = await asyncio.gather(
            reserve_account_request(SimpleNamespace(db=first), "shared", "one", "allowed", 4.0, key_hash="key-1"),
            reserve_account_request(SimpleNamespace(db=second), "shared", "two", "allowed", 4.0, key_hash="key-2"),
        )
        assert one == (("old", 4_000_000_000),)
        assert two == (("old", 1_000_000_000), ("current", 3_000_000_000))
        with pytest.raises(HTTPException) as full:
            await reserve_account_request(
                SimpleNamespace(db=second), "shared", "three", "allowed", 1.0, key_hash="key-3"
            )
        assert full.value.status_code == 429
        with pytest.raises(HTTPException) as replay:
            await reserve_account_request(
                SimpleNamespace(db=second), "shared", "one", "allowed", 4.0, key_hash="other-key"
            )
        assert replay.value.status_code == 409
        assert await settle_account_request(SimpleNamespace(db=first), "one", 2.0) == 2_000_000_000
        assert await settle_account_request(SimpleNamespace(db=second), "two", 4.0) == 4_000_000_000
        assert await reserve_account_request(
            SimpleNamespace(db=second), "shared", "three", "allowed", 1.0, key_hash="key-3"
        ) == (("old", 1_000_000_000),)
        await reserve_account_request(
            SimpleNamespace(db=first), "debt", "underestimated", "allowed", 1.0, key_hash="debt-key"
        )
        await settle_account_request(SimpleNamespace(db=second), "underestimated", 1.5)
        balance = await first.query_raw('SELECT "debt_nano" FROM "LiteLLM_TokeninAccount" WHERE "user_id" = $1', "debt")
        assert balance[0]["debt_nano"] == 500_000_000
        await first.execute_raw(
            'INSERT INTO "LiteLLM_TokeninGrant" '
            '("idempotency_key", "user_id", "payload_hash", "plan_id", "kind", "amount_nano") '
            "VALUES ('debt-topup','debt','trusted','payg','payg',1000000000)"
        )
        assert await reserve_account_request(
            SimpleNamespace(db=second), "debt", "after-topup", "allowed", 0.25, key_hash="debt-key"
        ) == (("debt-topup", 250_000_000),)
        balance = await first.query_raw('SELECT "debt_nano" FROM "LiteLLM_TokeninAccount" WHERE "user_id" = $1', "debt")
        assert balance[0]["debt_nano"] == 0
    finally:
        await second.conn.close()
        await first.conn.close()
