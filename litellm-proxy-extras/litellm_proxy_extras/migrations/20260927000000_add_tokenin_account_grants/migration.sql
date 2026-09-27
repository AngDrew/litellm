CREATE TABLE "LiteLLM_TokeninAccount" (
    "user_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_TokeninAccount_pkey" PRIMARY KEY ("user_id")
);

CREATE TABLE "LiteLLM_TokeninGrant" (
    "idempotency_key" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "payload_hash" TEXT NOT NULL,
    "plan_id" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "amount_nano" BIGINT NOT NULL,
    "subscription_id" TEXT,
    "period_index" INTEGER,
    "period_start" TIMESTAMP(3),
    "period_end" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_TokeninGrant_pkey" PRIMARY KEY ("idempotency_key"),
    CONSTRAINT "LiteLLM_TokeninGrant_amount_check" CHECK ("amount_nano" > 0),
    CONSTRAINT "LiteLLM_TokeninGrant_period_check" CHECK (
        ("kind" = 'fixed' AND "subscription_id" IS NOT NULL AND "period_index" IS NOT NULL
         AND "period_start" IS NOT NULL AND "period_end" IS NOT NULL AND "period_start" < "period_end")
        OR ("kind" = 'payg' AND "subscription_id" IS NULL AND "period_index" IS NULL
            AND "period_start" IS NULL AND "period_end" IS NULL)
    )
);

CREATE TABLE "LiteLLM_TokeninPolicy" (
    "idempotency_key" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "payload_hash" TEXT NOT NULL,
    "plan_id" TEXT NOT NULL,
    "models" TEXT[] NOT NULL,
    "rpm_limit" INTEGER NOT NULL,
    "max_parallel_requests" INTEGER NOT NULL,
    "effective_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_TokeninPolicy_pkey" PRIMARY KEY ("idempotency_key"),
    CONSTRAINT "LiteLLM_TokeninPolicy_rpm_check" CHECK ("rpm_limit" > 0),
    CONSTRAINT "LiteLLM_TokeninPolicy_parallel_check" CHECK ("max_parallel_requests" > 0)
);

CREATE UNIQUE INDEX "LiteLLM_TokeninGrant_user_id_subscription_id_period_index_key"
    ON "LiteLLM_TokeninGrant"("user_id", "subscription_id", "period_index");
CREATE INDEX "LiteLLM_TokeninGrant_user_id_period_start_idx"
    ON "LiteLLM_TokeninGrant"("user_id", "period_start");
CREATE INDEX "LiteLLM_TokeninPolicy_user_id_effective_at_idx"
    ON "LiteLLM_TokeninPolicy"("user_id", "effective_at");
ALTER TABLE "LiteLLM_TokeninGrant"
    ADD CONSTRAINT "LiteLLM_TokeninGrant_user_id_fkey" FOREIGN KEY ("user_id")
    REFERENCES "LiteLLM_TokeninAccount"("user_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_TokeninPolicy"
    ADD CONSTRAINT "LiteLLM_TokeninPolicy_user_id_fkey" FOREIGN KEY ("user_id")
    REFERENCES "LiteLLM_TokeninAccount"("user_id") ON DELETE RESTRICT ON UPDATE CASCADE;
