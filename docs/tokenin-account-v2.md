# Tokenin paid-period records (phase 1, dark)

This phase records verified account grants and future-dated policy revisions. It does **not** enforce account spend, consume grants, determine spendable balance, or issue managed keys. Keep `TOKENIN_ACCOUNT_V2_ENABLED` unset or `false` on the live proxy. Do not enable the platform's new purchase flow or deploy this as an account-budget fix

With the flag set to `true`, new `/tokenin/account/*` routes require `Authorization: Bearer <TOKENIN_ACCOUNT_SERVICE_TOKEN>`. The token must be at least 32 characters. It is a separate secret held by the trusted payment service, not a customer key or the proxy admin key. The payment service must verify payment, ownership, amount, and paid period before calling the proxy. Neither these routes nor their payloads are payment verification. The flag defaults to off, and the existing key and wallet routes retain their old behavior while it is off

`GET /tokenin/account/status` is the platform's boot gate: `{ "v2_enabled", "spend_enabled", "enforcement_active", "supported_routes" }`, service token only, env reads only, no database access and no side effects. `enforcement_active` is the same global two-switch state this document calls the phase marker, and `supported_routes` lists the routes that currently admit managed accounts

`GET /tokenin/account/catalog` returns `{ "plans": [{ "id", "kind", "monthly_value_usd", "rpm_limit", "max_parallel_requests" }], "enforcement_active": false }` (`false` because spend is off; the field tracks the switch on every route). Fixed `monthly_value_usd` is the proxy's resolved monthly credit; PAYG reports null. The payment service can snapshot this amount on invoice creation. There is no model alias catalog in the proxy's Tokenin plans, so paid-plan aliases must come from a separate trusted allowlist

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

`GET /tokenin/account/grants/{idempotency_key}?user_id=...` recovers the persisted credit amount after a timeout without creating a second grant. `GET /tokenin/account/summary?user_id=...` lists recorded grants and policy revisions. `available_usd: null` means no live spendable balance can be inferred from either endpoint, and is not a spend authorization

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

## Hold states and operator remedy

A reservation is `held` from admission until the cost callback resolves it. `settled` records the charged amount (a later callback with a different amount is refused with 409). `cancelled` means the work never billed and the whole reservation was returned. `uncertain` means the outcome is unknown: every allocation stays reserved, so nothing is refunded for work the provider may already have billed

`uncertain` is deliberate, and it is the reason credit can look pinned. Admission, a missing-cost success, a callback exception, and a failure with no recovered usage all leave the reservation in place rather than guessing. A failure that recovered partial stream cost settles that partial amount instead

Operators clear pinned holds with the service-token-only reconciliation routes: `GET /tokenin/account/holds?user_id=...&states=held,uncertain&older_than_minutes=30` lists holds with their reserved and charged amounts, and `POST /tokenin/account/holds/{request_id}/resolve` with `{"action": "settle"|"cancel"|"uncertain", "cost_usd": ..., "note": "..."}` applies the decision. `settle` needs `cost_usd`, `cancel` refunds the whole reservation, and every resolution writes a warning-level proxy log line with the note as the audit record. Both also resolve an `uncertain` hold, and only this route can: the automatic callbacks call the ledger without `allow_uncertain`, so nothing but an operator decision with a note on record ever releases an unpriced reservation

Account debt blocks new admissions until it is repaid from newly eligible credit, which happens automatically on the next admission once a grant is available. Debt is never written off by hand

Spend is armed in two steps: `TOKENIN_ACCOUNT_V2_ENABLED=true` records grants and policies, and `TOKENIN_ACCOUNT_SPEND_ENABLED=true` lets enrolled accounts spend. With the second switch off, every enrolled request still fails closed with 503

The flat team is pre-existing provisioning the proxy never creates: `FLAT_TEAM_ID = "ac0b4e54-71a7-4e1f-bfaf-32fad13c09e9"` in `litellm/proxy/tokenin/plans.py`, and both the legacy `tokenin/key/generate` and `POST /tokenin/account/keys` bind every issued key to it. Without that row, a freshly issued key fails auth with `Team doesn't exist in db` while charging nothing. Create it once per environment with the master key: `POST /team/new {"team_id": "ac0b4e54-71a7-4e1f-bfaf-32fad13c09e9", "team_alias": "tokenin-flat", "models": []}` (an empty `models` allows every model, which is what `models=["all-team-models"]` on the key expects). Verify with `SELECT "team_id" FROM "LiteLLM_TeamTable" WHERE "team_id" = 'ac0b4e54-71a7-4e1f-bfaf-32fad13c09e9'`. Staging and production must both have it before any key is issued; the acceptance run seeds it by hand for the same reason

Keys for managed accounts are issued with `POST /tokenin/account/keys` (service token, body `{user_id, alias}`). It derives the account and team server-side, carries no plan, budget, rate, or expiry field, refuses unenrolled accounts, and moves no credit, so rotation and re-issuance are harmless. Legacy `/tokenin/key/generate` still refuses enrolled accounts

## Cutover options

Decision: one coordinated cutover. The live proxy still issues per-key budgets, and this branch issues account-wallet credit instead, so deploying this branch before the platform migrates would change money behavior for existing customers. Keep the new image undeployed until the platform is ready

The rejected alternative was a legacy-compat shim: keep `f8f647f5dc:litellm/proxy/tokenin/router.py` (191 lines, per-key `max_budget`/`budget_duration` with plan limits) as the flag-off path and select it by an env flag. Estimated 4 to 6 hours including both-path tests, low build risk because the source is an intact git blob, but it leaves two live key paths to keep in step until the platform migrates. Take that route only if a proxy bugfix must reach the host before the platform is ready. Neither option changes product behavior until the platform migrates

## Policy revisions

`POST /tokenin/account/policy` takes an optional `expected_policy_id`: null succeeds only when the account has no policy yet, otherwise it must equal the account's newest-created policy, or the call returns 409 with `{"error": "stale_policy", "current_policy_id": ...}` and writes nothing. Identical retries of an already-applied policy still return `duplicate: true`. `GET /tokenin/account/summary` lists policies newest-created first with `policy_id` and `created_at`, and reports `latest_policy_id` for CAS. A policy revision that is not yet effective is recorded but inert, and `models` is validated against `GET /tokenin/account/models`

## Acceptance run and staging checks

`scripts/tokenin_e2e_acceptance.sh` is the gate: it starts an ephemeral Postgres, applies the proxy-extras migrations, and runs `tests/test_litellm/proxy/tokenin/test_account_spend_e2e.py` against the real ASGI proxy with both switches armed and a mock priced model. It passes (8 tests) on macOS arm64 and on Linux. On macOS arm64 `prisma py fetch` installs only the node CLI, so fetch the standalone query engine for the same engine hash into the path the generated client expects (`~/.cache/prisma-python/binaries/<version>/<hash>/node_modules/prisma/query-engine-<platform>`); without it the proxy's prisma client cannot connect and every test fails with a null client

The acceptance test holds a module-scoped app, which fights two per-test isolation fixtures: `tests/test_litellm/proxy/conftest.py` restores `proxy_server` globals and `tests/test_litellm/conftest.py` empties `litellm`'s callback lists after each test. Both must be re-stamped before each test, or the app loses `_ProxyDBLogger` from the second test on and **no cost callback ever runs**, which looks exactly like a settlement bug. An empty acceptance database also needs the flat team row (`FLAT_TEAM_ID`) seeded: `tokenin/key/generate` already assumes that team exists in production, and the proxy never creates it

Settlement is asynchronous relative to the HTTP response, so tests poll account state until no hold is `held` before asserting on charges

Timestamps cross the prisma boundary as text: raw query results are JSON strings with a UTC offset and bound parameters reach Postgres untyped. Every `TIMESTAMP(3)` bind carries an explicit `::timestamp` cast and a UTC ISO string, and every value read from a raw row goes through `ledger.as_naive_utc`. Comparisons and inserts that skip this fail with `column ... is of type timestamp without time zone but expression is of type text`, and a skipped read silently degrades a paid period to "no period", which would let fixed credit outlive its paid month

Two checks need a real provider on staging. First, streaming logs the cost callback per chunk and only the final chunk carries a cost, so an in-flight chunk must leave the hold `held`; the guard lives in `litellm/proxy/hooks/proxy_track_cost_callback.py`. Second, a client that disconnects mid-stream must leave `settled`/`uncertain` (never a silent refund) and reconciliation must clear the residual

`scripts/tokenin_e2e_disconnect.sh` runs the disconnect check locally against the same ephemeral Postgres: uvicorn serves the proxy on an ephemeral port, a real client hangs up mid-stream, and a stub upstream stays mid-flight so the interruption is genuine. Three tests pass. The first delivers one chunk then hangs up and requires the hold to settle at the partial cost, which is also the proof of the in-flight stream guard, because a per-chunk callback that marked the live hold `uncertain` would make that settle fail. The second destroys the TCP connection before the provider produced anything and takes the raw-socket route deliberately, since a client cannot hang up before the response headers: the proxy defers the response start until the provider's first chunk, so the whole-reservation refund branch in `common_request_processing.py` is unreachable over HTTP and the reachable outcome is `settled` or `uncertain`. The third disables streaming logging, which forces the deterministic `uncertain` case, and clears it through `POST /tokenin/account/holds/{request_id}/resolve`. Staging still owes the same check against a real provider

`enforcement_active` is not one flag. The records-only routes (`/tokenin/account/catalog`, `/tokenin/account/models`, policy and summary responses) report a literal `false`, meaning no live spendable balance can be inferred from them. `/tokenin/account/keys` and the holds listing report `account_spend_enabled()`, the global two-switch state. Neither is per-account spend authorization: that is decided per request from enrollment, policy, credit, and the switches

Revoking an account key needs no raw key value: `POST /key/delete` accepts `{"keys": ["<token_id>"]}` (the response's `token_id` is the stored hash, and the route hashes only raw `sk-` values) or `{"key_aliases": ["<alias>"]}`, and either never moves credit because grants and holds hang off `user_id`. An orphan whose `token_id` was lost to a crash is therefore still removable by alias
