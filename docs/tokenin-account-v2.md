# Tokenin paid-period records (phase 1, dark)

This phase records verified account grants and future-dated policy revisions. It does **not** enforce account spend, consume grants, determine spendable balance, or issue managed keys. Keep `TOKENIN_ACCOUNT_V2_ENABLED` unset or `false` on the live proxy. Do not enable the platform's new purchase flow or deploy this as an account-budget fix

With the flag set to `true`, new `/tokenin/account/*` routes require `Authorization: Bearer <TOKENIN_ACCOUNT_SERVICE_TOKEN>`. The token must be at least 32 characters. It is a separate secret held by the trusted payment service, not a customer key or the proxy admin key. The payment service must verify payment, ownership, amount, and paid period before calling the proxy. Neither these routes nor their payloads are payment verification. The flag defaults to off, and the existing key and wallet routes retain their old behavior while it is off

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
