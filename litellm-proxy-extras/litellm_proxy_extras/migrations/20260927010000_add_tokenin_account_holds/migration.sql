ALTER TABLE "LiteLLM_TokeninAccount"
    ADD COLUMN "debt_nano" BIGINT NOT NULL DEFAULT 0;

DROP INDEX "LiteLLM_TokeninPolicy_user_id_effective_at_idx";
CREATE UNIQUE INDEX "LiteLLM_TokeninPolicy_user_id_effective_at_key"
    ON "LiteLLM_TokeninPolicy"("user_id", "effective_at");

CREATE TABLE "LiteLLM_TokeninHold" (
    "request_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "key_hash" TEXT NOT NULL,
    "model" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "estimated_nano" BIGINT NOT NULL,
    "charged_nano" BIGINT,
    "admitted_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "settled_at" TIMESTAMP(3),
    CONSTRAINT "LiteLLM_TokeninHold_pkey" PRIMARY KEY ("request_id"),
    CONSTRAINT "LiteLLM_TokeninHold_state_check" CHECK ("state" IN ('held', 'settled', 'uncertain', 'cancelled')),
    CONSTRAINT "LiteLLM_TokeninHold_estimate_check" CHECK ("estimated_nano" > 0),
    CONSTRAINT "LiteLLM_TokeninHold_charged_check" CHECK ("charged_nano" IS NULL OR "charged_nano" >= 0),
    CONSTRAINT "LiteLLM_TokeninHold_user_id_fkey" FOREIGN KEY ("user_id")
        REFERENCES "LiteLLM_TokeninAccount"("user_id") ON DELETE RESTRICT ON UPDATE CASCADE
);

CREATE TABLE "LiteLLM_TokeninAllocation" (
    "request_id" TEXT NOT NULL,
    "grant_id" TEXT NOT NULL,
    "amount_nano" BIGINT NOT NULL,
    CONSTRAINT "LiteLLM_TokeninAllocation_pkey" PRIMARY KEY ("request_id", "grant_id"),
    CONSTRAINT "LiteLLM_TokeninAllocation_amount_check" CHECK ("amount_nano" >= 0),
    CONSTRAINT "LiteLLM_TokeninAllocation_request_id_fkey" FOREIGN KEY ("request_id")
        REFERENCES "LiteLLM_TokeninHold"("request_id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "LiteLLM_TokeninAllocation_grant_id_fkey" FOREIGN KEY ("grant_id")
        REFERENCES "LiteLLM_TokeninGrant"("idempotency_key") ON DELETE RESTRICT ON UPDATE CASCADE
);

CREATE INDEX "LiteLLM_TokeninHold_user_id_admitted_at_idx"
    ON "LiteLLM_TokeninHold"("user_id", "admitted_at");
CREATE INDEX "LiteLLM_TokeninHold_user_id_state_idx"
    ON "LiteLLM_TokeninHold"("user_id", "state");
CREATE INDEX "LiteLLM_TokeninAllocation_grant_id_idx"
    ON "LiteLLM_TokeninAllocation"("grant_id");
