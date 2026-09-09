# Gates: LiteLLM v1.100.0 fork upgrade

OWNS: Dockerfile, GATES.md, litellm/**, model_prices_and_context_window.json, package-lock.json, tests/**, ui/litellm-dashboard/**, uv.lock

Scope: Adopt upstream v1.100.0 while retaining and reconciling the fork-owned provider, routing, proxy accounting, pricing, and runtime behavior

- [x] G0: this ledger states runnable outcomes
  CHECK: node /Users/andrewanggada/.selesai/agent/skills/unlazy/scripts/gate-lint.mjs GATES.md
  EXPECT: LINT OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/andrewanggada/Documents/workdir/js_proj/litellm; path=2610fb58ca00/52 entries; EXPECT=matched; output-sha256=48630b7361dd44ee870917b12c3d19b9d7bdea738aaca16bb04d4cab83b772d2; output-bytes=8

- [x] G1: v1.100.0 is an ancestor and the merged source has no whitespace conflicts
  CHECK: git merge-base --is-ancestor e4f25265704e2b2c6cf6e81be2e4c5cffff896f4 HEAD && git diff --check e4f25265704e2b2c6cf6e81be2e4c5cffff896f4..HEAD && printf 'upstream baseline merged\n'
  EXPECT: upstream baseline merged
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/andrewanggada/Documents/workdir/js_proj/litellm; path=2610fb58ca00/52 entries; EXPECT=matched; output-sha256=3a8eb53b092e01bfc7670986ce0bd6d9fdd18e680dda5e7ba154ffffa9c7aab3; output-bytes=25

- [x] G2: router saturation, streaming release, affinity, and authorized fallbacks pass their regression tests
  CHECK: uv run --no-sync pytest tests/local_testing/test_router_max_parallel_requests.py tests/test_litellm/test_router.py tests/test_litellm/test_streaming_connection_cleanup.py tests/test_litellm/router_utils/pre_call_checks/test_deployment_affinity_check.py tests/test_litellm/proxy/auth/test_fallback_model_access.py -q && printf 'router regressions verified\n'
  EXPECT: router regressions verified
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/andrewanggada/Documents/workdir/js_proj/litellm; path=2610fb58ca00/52 entries; EXPECT=matched; output-sha256=9afe70d2a0edb924fa6e2e9a17399fafe025844ccd2402343caf682397bb90ec; output-bytes=1632

- [x] G3: Neuralwatt, Tokenin, spend IDs, tier budgets, and provider masking pass their regression tests
  CHECK: uv run --no-sync pytest tests/test_litellm/llms/neuralwatt/test_neuralwatt_chat.py tests/proxy_unit_tests/test_tokenin_concurrency_queue.py tests/test_litellm/proxy/spend_tracking/test_spend_tracking_utils.py tests/test_litellm/proxy/hooks/test_proxy_track_cost_callback.py tests/test_litellm/proxy/management_endpoints/test_key_management_endpoints.py tests/test_litellm/proxy/management_endpoints/test_common_daily_activity.py tests/test_litellm/proxy/auth/test_model_access_group_budgets.py -q && printf 'fork proxy regressions verified\n'
  EXPECT: fork proxy regressions verified
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/andrewanggada/Documents/workdir/js_proj/litellm; path=2610fb58ca00/52 entries; EXPECT=matched; output-sha256=7112b39f248d1c3f42118149f835e885010dd59cf1bfd2569d3d3db190da8233; output-bytes=1736

- [x] G4: the additive Prisma schema and migrations validate and generate a client
  CHECK: DATABASE_URL=postgresql://litellm:litellm@localhost:5432/litellm uv run --no-sync prisma validate --schema litellm/proxy/schema.prisma && DATABASE_URL=postgresql://litellm:litellm@localhost:5432/litellm uv run --no-sync prisma generate --schema litellm/proxy/schema.prisma && uv run --no-sync pytest tests/proxy_migration_tests -q && printf 'prisma schema verified\n'
  EXPECT: prisma schema verified
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/andrewanggada/Documents/workdir/js_proj/litellm; path=2610fb58ca00/52 entries; EXPECT=matched; output-sha256=a53a702b8f6dac6da817068f0c2bd44895ef54d3a2937a7122f2f080422dcf14; output-bytes=1160

- [x] G5: the reconciled runtime image builds
  CHECK: docker build -t litellm:v1.100.0-upgrade . && printf 'runtime image verified\n'
  EXPECT: runtime image verified
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/andrewanggada/Documents/workdir/js_proj/litellm; path=2610fb58ca00/52 entries; EXPECT=matched; output-sha256=a95426317fefe8bea24c27e424fe3e21a161f1e9fa678b3d51dac65a1a941e03; output-bytes=11709

- [ ] G6: repository static checks pass
  CHECK: log=$(mktemp) && make check >"$log" 2>&1; status=$?; tail -n 200 "$log"; rm -f "$log"; [ "$status" -eq 0 ] && printf 'repository checks verified\n'
  EXPECT: repository checks verified
  EVIDENCE: pending
