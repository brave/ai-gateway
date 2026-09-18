#!/bin/bash
# Public runnability smoke test: reproduces the exact open-source onboarding
# path (fresh clone -> make setup && make start, only .env.example) and
# POSTs model:"automatic" to /v1/chat/completions - the request shape that
# triggers BLOCKER-HIGH H (bare model_triaging[...] lookup crashing on the
# public default MODEL_TRIAGING="{}").
#
# A tiny fake OpenAI-compatible backend stands in for Ollama (see
# scripts/smoke-test/fake_backend.py) - H's crash happens during model
# *selection*, before any backend call, so no real inference is needed to
# catch it.
set -euo pipefail
cd "$(dirname "$0")/.."

FAKE_BACKEND_PORT=11434
SERVER_URL="http://127.0.0.1:8000"
FAKE_BACKEND_PID=""
ENV_BACKUP=""

cleanup() {
  echo "--- Cleaning up ---"
  docker stop ai-gateway-server >/dev/null 2>&1 || true
  if [ -n "$FAKE_BACKEND_PID" ]; then
    kill "$FAKE_BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "$ENV_BACKUP" ]; then
    mv -f "$ENV_BACKUP" .env
  elif [ "$CREATED_ENV" = "1" ]; then
    rm -f .env
  fi
}
trap cleanup EXIT

# Use .env.example verbatim, regardless of any existing .env, so this
# actually tests the public onboarding path - not a developer's local config.
CREATED_ENV=0
if [ -f .env ]; then
  ENV_BACKUP="$(mktemp)"
  cp .env "$ENV_BACKUP"
else
  CREATED_ENV=1
fi
cp .env.example .env

echo "--- make setup ---"
make setup

echo "--- Starting fake backend on :$FAKE_BACKEND_PORT ---"
python3 scripts/smoke-test/fake_backend.py "$FAKE_BACKEND_PORT" &
FAKE_BACKEND_PID=$!

for _ in $(seq 1 20); do
  if curl -sf "http://127.0.0.1:$FAKE_BACKEND_PORT/" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

echo "--- make start ---"
make start

echo "--- Waiting for /health/ready ---"
READY=0
for _ in $(seq 1 60); do
  if curl -sf "$SERVER_URL/health/ready" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [ "$READY" != "1" ]; then
  echo "FAIL: server never became ready"
  docker logs ai-gateway-server || true
  exit 1
fi

echo "--- POST /v1/chat/completions (model=automatic) ---"
RESPONSE_FILE="$(mktemp)"
STATUS=$(curl -s -o "$RESPONSE_FILE" -w '%{http_code}' \
  -X POST "$SERVER_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model": "automatic", "messages": [{"role": "user", "content": "ping"}]}')

BODY="$(cat "$RESPONSE_FILE")"
rm -f "$RESPONSE_FILE"

if [ "$STATUS" != "200" ]; then
  echo "FAIL: expected HTTP 200, got $STATUS"
  echo "Response body: $BODY"
  docker logs ai-gateway-server || true
  exit 1
fi

if ! echo "$BODY" | python3 -c '
import json, sys
data = json.load(sys.stdin)
content = data["choices"][0]["message"]["content"]
assert content, "empty message content"
' 2>/dev/null; then
  echo "FAIL: response was 200 but did not contain a chat completion message"
  echo "Response body: $BODY"
  docker logs ai-gateway-server || true
  exit 1
fi

echo "PASS: model=automatic request round-tripped successfully"
