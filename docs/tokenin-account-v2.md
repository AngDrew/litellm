# Tokenin paid-period records (phase 1, dark)

This phase records verified account grants and future-dated policy revisions. It does **not** enforce account spend, consume grants, determine spendable balance, or issue managed keys. Keep `TOKENIN_ACCOUNT_V2_ENABLED` unset or `false` on the live proxy. Do not enable the platform's new purchase flow or deploy this as an account-budget fix

With the flag set to `true`, new `/tokenin/account/*` routes require `Authorization: Bearer <TOKENIN_ACCOUNT_SERVICE_TOKEN>`. The token must be at least 32 characters. It is a separate secret held by the trusted payment service, not a customer key or the proxy admin key. The payment service must verify payment, ownership, amount, and paid period before calling the proxy. Neither these routes nor their payloads are payment verification. The flag defaults to off, and the existing key and wallet routes retain their old behavior while it is off

`GET /tokenin/account/catalog` returns `{ "plans": [{ "id", "kind", "monthly_value_usd", "rpm_limit", "max_parallel_requests" }], "enforcement_active": false }`. Fixed `monthly_value_usd` is the proxy's resolved monthly credit; PAYG reports null. The payment service can snapshot this amount on invoice creation. There is no model alias catalog in the proxy's Tokenin plans, so paid-plan aliases must come from a separate trusted allowlist

`POST /tokenin/account/grants` accepts:

```json
{
  "user_id": "platform-users-id",
  "idempotency_key": "verified-purchase-id:0",
  "plan_id": "medium",
  "kind": "fixed",
  "subscription_id": "stable-subscription-id",
  "period_index": 0,
  "period_start": "2026-01-31T00:00:00Z",
  "period_end": "2026-02-28T00:00:00Z"
}
```

Fixed grants use the plan catalog's `monthly_value` when present. Older deployed catalogs store the monthly amount in `max_budget` and omit `monthly_value`; that fallback is allowed only when `budget_duration` is explicitly `30d` or `1mo`. A weekly catalog with no `monthly_value` is rejected. Audit the deployed catalog before allowing grants: in the checked-in proxy YAML, `max_budget` is weekly and `monthly_value` is monthly. `period_start` and `period_end` come from the verified paid anniversary, are timezone-aware, span 28 to 32 days, and must be contiguous with adjacent periods for the same subscription. The database rejects a second purchase id for the same `(user_id, subscription_id, period_index)` and the service rejects a repeated idempotency key with a different payload (409). Identical retries return the original credited amount with `duplicate: true`. The database creates no wallet credit or spendable balance in this phase

PAYG uses the same route with `kind: "payg"`, `plan_id: "payg"`, and a verified positive `topup_usd`; omit all subscription and period fields. The server stores money as integer nanodollars and rejects values with more than nine decimal places. The payment service must provide the original provider purchase ID as its stable idempotency key

`GET /tokenin/account/grants/{idempotency_key}?user_id=...` recovers the persisted credit amount after a timeout without creating a second grant. `GET /tokenin/account/summary?user_id=...` lists recorded grants and policy revisions. `available_usd: null` and `enforcement_active: false` mean no live spendable balance can be inferred from either endpoint

`POST /tokenin/account/policy` accepts `user_id`, `idempotency_key` (a stable policy event ID), `plan_id`, `models` (an explicit array of model aliases), `rpm_limit` (positive integer), `max_parallel_requests` (positive integer), and timezone-aware `effective_at`. Empty `models` explicitly denies all models; wildcards and `all-team-models` are rejected. An identical retry returns `duplicate: true`, and a changed payload for the same ID returns 409. Future policy revisions remain records only in phase 1, so an early upgrade cannot change running limits

When the flag is enabled and an account record exists, legacy `/tokenin/wallet/topup`, `/tokenin/key/update`, and `/tokenin/key/generate` refuse to mutate that account. They remain available for non-enrolled accounts. This is not a cutover mechanism by itself: any account with an unenrolled or unattributed key, an old wallet balance, or unresolved spend must be reconciled before phase 2 can enforce it

Phase 2 requires an admission guard for every account-attributed key, atomic shared holds with FIFO allocation, per-request cost settlement including streaming and cancellation, model/rate/concurrency enforcement across all keys and workers, account policy activation by effective paid period, and fail-closed pricing and reconciliation. Fixed balances may carry only into the *next paid* month; they stop at the end of the last paid month if there is no renewal. Preserve old key caps until staging proves zero-credit rejection with two keys, verified single credit shared across both, replay idempotency, expired rollover rejection, PAYG fallback, concurrent requests, and safe rotation. Coordinate the payment service deployment before any flag change. No schema migration should rewrite historic user or key spend; reconcile it from actual rows and verified purchases separately

## Phase 2 ledger (dark, no admission path yet)

The phase-2 migration adds `LiteLLM_TokeninHold`, `LiteLLM_TokeninAllocation`, `LiteLLM_TokeninAccount.debt_nano`, and a unique `LiteLLM_TokeninPolicy(user_id, effective_at)` index. `litellm/proxy/tokenin/ledger.py` implements reservation and settlement, but **no proxy route or auth path calls it**. Live behavior is unchanged and no account can spend from these rows yet

Eligibility and order: a fixed grant is spendable in its own paid period and in the immediately following period only when that next period was actually paid (a contiguous grant for the same `subscription_id` and `period_index + 1`); otherwise it stops at the end of its own period. PAYG is independent, never expires, needs no active subscription, and is spent after all eligible fixed credit, oldest period first

Admission invariants in the ledger: the active paid grant's `period_start` must have a trusted policy revision at exactly that timestamp before any of its credit is spendable, so a stale or future revision cannot unlock a new period. Later revisions to the same plan with `effective_at` inside the paid period take effect at that timestamp (latest `effective_at <= now` wins), and a policy revision alone never creates credit. The ledger also enforces `models` membership, an RPM window, and cross-worker concurrency counted from durable holds

Money and failure handling: estimated maximum cost is reserved under a `FOR UPDATE` lock on the account row; settlement charges the actual cost rounded up to nanodollars, returns unused reservation, and charges any overrun to still-eligible credit before recording account `debt_nano`. Debt is repaid from newly eligible credit on the next admission and blocks spend until it clears. A replayed request ID is bound to `(user_id, key_hash, model, estimated_nano)` and mismatches are `409`. Missing account, missing policy, unpriced or out-of-range cost, unknown hold state, and unpriceable requests fail closed

Cutover prerequisites: keep `apply_user_budget_to_team_keys` **false** in the running proxy while V2 is enabled. V2 grants must never be written to `LiteLLM_UserTable.max_budget`, and managed keys must be account-bound and team-scoped so the legacy personal-wallet check stays skipped; the ledger is the only authority for V2 spend. Preflight should reject enabling V2 enforcement when live `general_settings` turns the global user-budget guard on, because `ensure_wallet_user` seeds `max_budget = 0` and would deny every V2 key. Old per-key caps stay in place for legacy keys until staged verification passes

`tests/test_litellm/proxy/tokenin/test_ledger.py` exercises the pure rules and a transactional fake. `tests/test_litellm/proxy/tokenin/test_ledger_pg.py` runs the same path against an isolated local PostgreSQL when `TOKENIN_TEST_DATABASE_URL` points at a local `tokenin_local` database; it refuses any other host or database and is skipped without the variable
