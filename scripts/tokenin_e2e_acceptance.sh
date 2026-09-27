#!/usr/bin/env bash
# One-command local acceptance run for managed-account spend.
#
# Starts an ephemeral Postgres container, applies the proxy extras migrations, then runs
# the ASGI acceptance test with V2 + account spend armed and litellm's mock provider
# standing in for the upstream (no external calls). The container is removed on exit.
#
#   scripts/tokenin_e2e_acceptance.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="tokenin-e2e-pg-$$"

cd "$REPO_ROOT"

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "starting ephemeral postgres on a free local port"
docker run -d --rm --name "$CONTAINER" \
  -e POSTGRES_PASSWORD=e2e -e POSTGRES_USER=e2e -e POSTGRES_DB=tokenin_e2e \
  -p "127.0.0.1::5432" postgres:16.9-alpine >/dev/null

until docker exec "$CONTAINER" pg_isready -U e2e -d tokenin_e2e >/dev/null 2>&1; do sleep 1; done
PORT="$(docker port "$CONTAINER" 5432/tcp | head -1 | sed 's/.*://')"
DATABASE="postgresql://e2e:e2e@127.0.0.1:${PORT}/tokenin_e2e"
echo "postgres ready at 127.0.0.1:${PORT}"

if [ ! -e "$HOME/.cache/prisma-python/binaries" ]; then
  echo "fetching the prisma python query engine (one-time)"
  ./.venv/bin/prisma py fetch >/dev/null
fi

echo "applying proxy extras migrations (includes the Tokenin account migrations)"
DATABASE_URL="$DATABASE" ./.venv/bin/prisma migrate deploy \
  --schema litellm-proxy-extras/litellm_proxy_extras/schema.prisma >/dev/null

echo "running acceptance test"
DATABASE_URL="$DATABASE" TOKENIN_E2E_DATABASE_URL="$DATABASE" DD_TRACE_ENABLED=false \
  .venv/bin/python -m pytest -q -o addopts='' tests/test_litellm/proxy/tokenin/test_account_spend_e2e.py
