# Upgrade to LiteLLM v1.104.0

Tracking doc for merging upstream `v1.104.0` (`79645770fe`) into the fork branch `tokenin-managed-read-models` (`563d44be62`, on v1.100.0)

Work happens on branch `upgrade-v1.104.0` in the worktree `../litellm-upgrade-v1.104.0`, so the main checkout is untouched until the branch is reviewed

## Baseline

| | |
|---|---|
| Merge base | `10631eb834` (`v1.100.0-rc.1`) |
| Fork commits since base | 50 non-merge commits (116 files, mostly Tokenin) |
| Upstream commits since base | 7,480 (7,690 files) |
| Conflicted files | 33 |

The v1.100.0 release tag (`e4f2526570`) sits on an upstream release branch that v1.104.0 does not contain. Its 2 extra commits (Wolfi glibc pins in the Dockerfiles) show up as "ours" in the merge, but they are upstream work and v1.104.0 supersedes them

## Resolution rules

When a conflict mixes upstream refactors with fork behavior, keep upstream's structure and re-apply the fork behavior on top of it. Fork behavior that upstream already ships in another form is dropped in favor of upstream, and that is written down below. Nothing fork-specific is removed silently

## Progress

### Phase 1: merge and easy conflicts

- [x] Merge started on `upgrade-v1.104.0`
- [x] `.gitignore`
- [x] Dockerfiles (`Dockerfile`, `backend/`, `gateway/`, `migrations/`, `docker/Dockerfile.database`, `docker/Dockerfile.non_root`)
- [x] `ui/litellm-dashboard/src/components/templates/key_info_view.tsx`
- [x] Test files moved by upstream (`tests/test_litellm/*` to `tests/unit/*`)
  - [x] `tests/unit/test_router/test_router.py`
  - [x] `tests/unit/router_utils/pre_call_checks/test_deployment_affinity_check.py` (upstream, see router decision)
  - [x] `test_simple_shuffle_capacity.py` dropped (see router decision)
  - [x] `tests/unit/proxy/test_proxy_config_unit_test.py` (config path fixed for the new depth)
  - [x] `tests/test_litellm/proxy/{hooks/test_proxy_track_cost_callback,management_endpoints/test_key_management_endpoints,proxy_server/test_routes_models,test_proxy_utils}.py`: rebuilt as the v1.104.0 file plus every test the fork added. The fork's other edits in these files were formatting only (checked by comparing ASTs), so nothing else is carried

### Phase 2: proxy and core

- [x] `litellm/proxy/_types.py`
- [x] `litellm/types/proxy/model_listing.py`
- [x] `litellm/proxy/litellm_pre_call_utils.py`
- [x] `litellm/proxy/common_request_processing.py`
- [x] `litellm/proxy/hooks/proxy_track_cost_callback.py`
- [x] `litellm/proxy/management_endpoints/key_management_endpoints.py`
- [x] `litellm/proxy/proxy_server.py`
- [x] `litellm/proxy/utils.py`
- [x] `litellm/main.py`
- [x] `litellm/litellm_core_utils/exception_mapping_utils.py`
- [x] `litellm/litellm_core_utils/litellm_logging.py`
- [x] `litellm/litellm_core_utils/streaming_handler.py`

### Phase 3: router concurrency

- [x] `litellm/types/router.py`
- [x] `litellm/router_utils/client_initalization_utils.py`
- [x] `litellm/router_utils/pre_call_checks/deployment_affinity_check.py`
- [x] `litellm/proxy/hooks/parallel_request_limiter_v3.py`
- [x] `litellm/router.py`

### Phase 4: dependencies and database

- [x] `uv.lock` and `pyproject.toml` merged without conflicts and the fork adds no dependency, so the lock is v1.104.0's and `uv sync --frozen --extra proxy --group proxy-dev` installs it as is
- [x] Prisma schema validates, the 3 `schema.prisma` copies are identical, and the 2 Tokenin migrations sort after upstream's latest (`20260923000000_add_agent_kill_switch`)

### Phase 5: verification

- [x] No unmerged paths or conflict markers left, every changed Python file parses
- [x] Fork modules import cleanly against v1.104.0 (proxy server, router, Tokenin, concurrency queue, NeuralWatt, Jev callbacks)
- [x] Router concurrency and affinity tests pass
- [x] Tokenin, cost callback, key management and model listing tests pass
- [x] NeuralWatt and DeepSeek tests pass
- [x] Merged tree compared with a clean v1.104.0 checkout on the same suites (10,400 tests across router, streaming, cost, logging, exception mapping, proxy hooks, spend tracking, model listing, public endpoints, DeepSeek, DeepInfra, OpenRouter, xAI): the same 33 failures and 2 errors on both sides, none only in the merge
- [x] Fork-only suites pass (4,773 tests: cost margin, Tokenin, NeuralWatt, Jev, Dockerfile callbacks, concurrency queue, management endpoints, pre-call utils). The only errors are `test_prompt_caching_requests.py`, which needs a database and errors on v1.104.0 too
- [x] Dashboard: the fork's 3 UI source files typecheck and the touched component tests pass (392)
- [ ] Integrations hit by major bumps (`mcp` 2.x, `langfuse` 4.x, OpenTelemetry 1.33): untouched by the fork, covered only by the import check
- [ ] Docker image builds (blocked, see open issues)
- [ ] Tokenin end-to-end scripts pass against a live proxy (blocked on the image)

## Decisions

### Dockerfiles

`backend/`, `gateway/`, `migrations/`, `docker/Dockerfile.database` and `docker/Dockerfile.non_root` only differed by the v1.100.0 release-branch base image pins, so they take the v1.104.0 version as is

`Dockerfile` takes v1.104.0 (new base digest, pgbouncer stage, public Wolfi repo, `bedrock-realtime` extra) and re-applies the two fork changes: `nodejs-26` with `!openssl-4.0-libcrypto` / `!openssl-4.0-libssl` in both the builder and runtime stages, and `COPY --from=builder /app/callbacks /app/callbacks`. The image build in phase 5 has to confirm the OpenSSL pin still resolves on the new base digest

### Router concurrency: upstream wins (decided by the owner)

The fork's own `max_parallel_requests` work is dropped in favor of upstream. v1.104.0 already ships the core idea the fork added: `MaxParallelRequestsLimit` in `client_initalization_utils.py` refuses a request with a 429 (`RateLimitError`, `CONCURRENT_REQUESTS`) when a deployment's slots are full instead of blocking, and an `AsyncExitStack` deployment slot is held until a stream closes

What goes away from the fork:

- `provider_max_parallel_requests` (one cap shared by every deployment with the same `api_base` + `api_key`). No config in this repo or in `tokenin-platform` uses it. If a config still sets it, the key is silently ignored after the upgrade
- `litellm.MaxParallelRequestsError`. Upstream raises a plain `RateLimitError` with `rate_limit_type=CONCURRENT_REQUESTS`, so existing fallback handling still applies
- Routing strategies skipping saturated deployments before selection (`_filter_mpr_available_deployments`, the simple-shuffle capacity gate)
- Deployment affinity soft-pin at `max_parallel_requests` (`deployment_affinity_check.py` takes upstream as is)
- Tests for the above: `test_simple_shuffle_capacity.py`, the fork additions in `test_router_max_parallel_requests.py`, `test_deployment_affinity_check.py`, `test_streaming_connection_cleanup.py` and the `on_close` assertions in `test_router.py`. `litellm/__init__.py`, `litellm/exceptions.py` and `litellm/utils.py` are back to exactly v1.104.0

What is kept, because it is not deployment concurrency:

- `router.py`: `_aggregate_declared_flag` and the per-group `supports_audio_input` / `supports_documents` / `supports_video_input` / `supports_structured_output` / `default_reasoning_effort` aggregation, plus `_mask_fallback_config_for_errors` so fallback errors never print configured model names. Everything else in `router.py` is exactly v1.104.0
- `types/router.py`: the new `ModelGroupInfo` fields, next to upstream's new `supports_fast_mode`
- `parallel_request_limiter_v3.py`: the Tokenin concurrency queue still owns key-level `max_parallel_requests` when it holds a slot for the request. Upstream moved descriptor building into `_build_request_rate_limit_descriptors`, so the `skip_api_key_max_parallel_requests` flag is threaded through it. `request_capacity` (non-generation calls) does not pass it, same as before
- Test additions for capability flags and fallback masking in `tests/unit/test_router/test_router.py`

### Proxy and core

Each item below keeps v1.104.0 and puts the fork behavior back on top

- `proxy/_types.py`: the Tokenin hold fields sit next to upstream's new budget snapshot fields
- `litellm_pre_call_utils.py`: `user_api_key_tokenin_account_hold_id` stays in the client-stripped metadata set, next to upstream's new reserved keys
- `key_management_endpoints.py`: `/key/info` still runs `_add_reset_timing_to_key_info` (the `reset_in` fields the dashboard's Budget Windows card reads) on upstream's renamed `key_info_dict`
- `common_request_processing.py`: provider cost header is still popped from response headers, next to upstream's new call id and timing values. The Tokenin hold-on-disconnect call landed in upstream's refactored `_finalize_streaming_generator_cleanup` in the same spot as before
- `proxy_track_cost_callback.py`: `response_cost` still falls back to `kwargs["response_cost"]` when the standard logging payload has none, written as upstream's `Final`. The Tokenin hold is marked uncertain in upstream's new "no tracking but reservation exists" branch too, so a hold is never silently left open there, and in upstream's rewritten exception handler
- `proxy_server.py`: managed-account policy filtering is applied before upstream's new undiscoverable and listing-callback filters in `/v1/models` and `/v1/models/{id}`
- `proxy/utils.py` (`create_model_info_response`): upstream rewrote the lookup to resolve aliases and every deployment behind a name. The fork's `supports_*` pass-through now reads those same candidates (first deployment, best source wins) and config `model_info` still overrides the cost map
- `exception_mapping_utils.py`: NeuralWatt provider-name scrubbing now also covers upstream's new bug-report notice. Upstream dropped the `Any` import that `_scrub_provider_name` used, which would have been a `NameError` at import time, so the helper is now typed with a `TypeVar`
- `litellm_logging.py`: the sync streaming path still skips recomputing `response_cost` when one is already set, and also calls upstream's new `_surface_response_headers_from_result`
- `package-lock.json` (root): back to upstream. The fork only had the package `name` changed by a local `npm install`

### Streaming cost (needs a careful look in review)

Upstream added `_stamp_streaming_usage_cost` (LIT-6427): the SDK stream builder now puts the final cost on `usage.cost` by default, without `include_cost_in_streaming_usage`. It also limits copying `usage.cost` into the provider cost header to OpenRouter (`_USAGE_COST_HEADER_PROVIDERS`), and xAI's reported cost is priced by the calculator

The fork's rule (`fix(cost): apply margins to provider-reported spend`) is that the client only ever sees the billed amount (margin included), never the provider's raw number

A first resolution that ran upstream's stamp after the fork's steps billed a token-priced OpenRouter stream twice (cost x 1.3 x 1.3): the stamp wrote the billed amount into `usage.cost` after the final cost was already decided, and the stream wrapper then copied it into the provider cost header. The fork's end-to-end cost tests caught it. The final shape replaces the fork's flag-gated billing, the fork's withholding and upstream's stamp with one step, run before the final cost is recorded:

1. `_attach_streamed_provider_cost` hands any provider-reported cost to the calculator (unchanged)
2. `_stamp_streaming_usage_cost` puts the billed amount on `usage.cost`. A raw provider number is replaced, and dropped if it cannot be billed
3. `_set_stream_builder_response_cost` records that billed amount as the final cost, so `_propagate_usage_cost_to_hidden_params` (OpenRouter only, and skipped once a cost is set) never copies it back

Effects to know about:

- Streams now show the billed cost even with `include_cost_in_streaming_usage` off. Before the merge they showed nothing in that case. The proxy config turns the flag on, so production output does not change
- A provider outside OpenRouter, DeepInfra and xAI that reports `usage.cost` is shown and billed the calculator's price, as the fork already did with the flag on. Upstream would show and bill the provider's number. `test_stream_chunk_builder_keeps_provider_reported_usage_cost` was rewritten to the fork rule
- xAI is left as upstream has it (raw `usage.cost` shown, billed once by the calculator) because its calculator reads `usage.cost` itself, so writing the billed amount there would mark it up again
- `ResponseMetadata._get_value_from_hidden_params` was removed upstream but the fork's `_show_billed_cost` (auto-merged) still called it, which broke every non-streaming call with an `AttributeError` (110 test failures). It is back as a small typed getter

### Fallback errors

Upstream added a "Fallbacks are configured for: <groups>" sentence to the no-fallback error. That prints configured model group names, which the fork hides, so it now gets the masked names (`some**************`). `test_router_exception_redaction.py` asserts the masked form

### Model listing capabilities

The fork's declared `supports_*` override in `create_model_info_response` fetched the deployment a second time. It now reads `declared_capabilities`, a new field on upstream's `DeploymentModelListingInfo` filled in `Router.get_model_listing_info` from the first deployment's `model_info`, and `mode` comes from upstream's `get_configured_mode`. Upstream's `/v1/models` test asserts an exact entry, so it now checks the non-capability fields exactly and the capability flags separately

### NeuralWatt

- Errors: the fork's own NeuralWatt status branch only knew 400, 401, 429, 500 and 503 and turned everything else (403, 404, 408, 422) into `APIError`. Upstream's generic status mapping covers all of them, so the fork branch is removed. The provider-name scrubbing stays and a test now checks every mapped status never names the provider
- Provider registry: upstream's test wants every provider in the Add Model dropdown or in a frozen unlisted set. `/public/providers/fields` is a public endpoint, so NeuralWatt goes in the unlisted set instead of advertising the provider

### Spend log ids (owner decision needed)

The fork stores spend logs under `<litellm_call_id>_<response id>` (`fix(spend): prevent duplicate log IDs`) because some providers reuse response ids and `skip_duplicates` then silently dropped successful spend rows. Upstream's new tests (LIT-6302, LIT-6806) require `request_id` to be exactly the id the client received so `GET /spend/logs?request_id=...` finds it. The fork behavior is kept and those 6 tests assert the prefixed form. The cost is that lookups by the bare response id stop matching; upstream's separate `litellm_call_id` column still works

### Jev classifier

Upstream added a `NON_REASONING` tier to `_CLASSIFICATION_TIER_CRITERIA`, which the Jev classifier copied into its choices. A router without that pool treats the answer as a failed classification and falls back to the heuristic, so Jev is limited to the 4 standard tiers again

### Fork tests adjusted for upstream API changes

- `test_accounts.py`, `test_proxy_utils.py`: fake routers now provide `model_group_alias`, `get_model_listing_info` and `get_configured_mode` instead of the removed `get_configured_token_limits`
- `test_common_daily_activity.py`: fake rows carry upstream's new `total_response_time_ms` and `timed_requests`
- `test_provider_cost_margin.py`: `_propagate_usage_cost_to_hidden_params` takes the provider. The test that expected no streamed cost with the flag off now expects the billed cost, and a new test checks a token-priced stream is shown and billed once
- `test_key_management_endpoints.py`: upstream's budget-limits test now allows the fork's `reset_in` fields
- `test_streaming_handler.py`: the OpenRouter propagation test starts from an unpriced response, which is the order the stream wrapper uses

## Open issues

- Docker image build is blocked: the Docker daemon cannot pull from Docker Hub (`docker pull docker/dockerfile:1.7` times out while `registry-1.docker.io` answers over HTTPS). The build, the OpenSSL pin check and the Tokenin end-to-end scripts still have to run
- Spend log id format needs an owner decision (see above)
- Lint ratchets measured against v1.104.0 report fork-owned code over upstream's stricter budgets (ruff `B006`, `B008`, `B010`, `DTZ001`, `PERF401`, `UP028`; basedpyright `reportOptionalOperand`, `reportOperatorIssue` and others), mainly in `litellm/llms/neuralwatt`, `litellm/proxy/tokenin` and `concurrency_queue.py`. These files are unchanged by the merge. Fixing them is follow-up work, not part of the upgrade
- Pre-existing upstream failures on both sides, for reference: `test_max_parallel_requests_rpm_rate_limiting`, `test_async_log_success_event_hands_the_sidecar_a_compact_event_and_skips_the_pipeline`, the order-dependent `test_logging_result_for_bridge_calls`, and the 33 shared failures in the comparison suites
- `TestMemberAutoRouterInference::test_cached_roster_revocation_blocks_classifier_and_session_rebinding` fails now and then under xdist load (classifier timeout) and passes on its own
- The merge is committed on `upgrade-v1.104.0` before the Docker build and end-to-end run, which still have to pass before the branch is merged
