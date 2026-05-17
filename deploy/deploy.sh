#!/usr/bin/env bash
#
# One-shot deploy script for the production VM.
#
# Images are built locally on the self-hosted runner (no registry).
#
# Usage:
#   SKIP_PULL=1 ./deploy.sh                  # restart `latest` (local image)
#   SKIP_PULL=1 IMAGE_TAG=<tag> ./deploy.sh  # restart a specific local image
#   ./deploy.sh                              # legacy: `compose pull` first
#
# Assumes:
#   - docker + docker compose v2 installed
#   - the image `omniasr-server:${IMAGE_TAG:-latest}` exists locally
#   - .env file lives next to docker-compose.prod.yml
#
# This script is idempotent: running it twice with the same IMAGE_TAG is a no-op.

set -euo pipefail

cd "$(dirname "$0")/.."

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
COMPOSE_CMD=(docker compose -f "$COMPOSE_FILE")

if [ "${SKIP_PULL:-0}" = "1" ]; then
    echo "==> SKIP_PULL=1 — using local image (tag: ${IMAGE_TAG:-latest})."
else
    echo "==> Pulling image (tag: ${IMAGE_TAG:-latest})..."
    "${COMPOSE_CMD[@]}" pull
fi

echo "==> Restarting service..."
"${COMPOSE_CMD[@]}" up -d

echo "==> Waiting for /readyz..."
ATTEMPTS=0
MAX_ATTEMPTS=60   # 60 * 5s = 5 minutes
PORT="${OMNILINGUAL_PORT:-8081}"
ROOT_PATH="${OMNILINGUAL_ROOT_PATH:-}"
until curl -fsS "http://127.0.0.1:${PORT}${ROOT_PATH}/readyz" >/dev/null 2>&1; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if (( ATTEMPTS >= MAX_ATTEMPTS )); then
        echo "!! /readyz never came up in $((MAX_ATTEMPTS * 5))s — dumping logs and bailing." >&2
        "${COMPOSE_CMD[@]}" logs --tail=200 omniasr
        exit 1
    fi
    sleep 5
done

echo "==> Deployed and ready."
"${COMPOSE_CMD[@]}" ps
