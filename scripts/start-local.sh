#!/bin/bash

# These are forwarded into the container even when unset; empty strings would
# fail JSON/bool parsing for these fields (unlike plain string settings, which
# tolerate ""), so default them here the same way start-local-native.sh does.
# (Defaults are kept in separate variables - embedding a literal "{}" inside
# a ${VAR:-...} expansion is escaping-fragile across shells.)
MODEL_TRIAGING_DEFAULT='{}'
MCP_SERVERS_DEFAULT='[]'
DYNAMIC_LEO_CONFIG_DEFAULT='{}'
MODEL_TRIAGING="${MODEL_TRIAGING:-$MODEL_TRIAGING_DEFAULT}"
MCP_SERVERS="${MCP_SERVERS:-$MCP_SERVERS_DEFAULT}"
DYNAMIC_LEO_CONFIG="${DYNAMIC_LEO_CONFIG:-$DYNAMIC_LEO_CONFIG_DEFAULT}"
DEEP_RESEARCH_ENABLED="${DEEP_RESEARCH_ENABLED:-false}"
SHARE_MAX_CIPHERTEXT_BYTES="${SHARE_MAX_CIPHERTEXT_BYTES:-1048576}"

# Attach a TTY for interactive local use (Ctrl+C stops the container), but
# fall back to detached mode when run from a script/CI where stdin isn't a
# terminal - `-it` would otherwise fail with "the input device is not a TTY".
if [ -t 0 ]; then
  RUN_MODE_FLAGS="-it"
else
  RUN_MODE_FLAGS="-d"
fi

# shellcheck disable=SC2046  # process substitution is a single word
docker run $RUN_MODE_FLAGS --rm \
  --name ai-gateway-server \
  -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)/aichat:/app/aichat" \
  --env-file <(env | grep '^INFERENCE_PROFILE_') \
  -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  -e AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" \
  -e AWS_DEFAULT_REGION="us-west-2" \
  -e AWS_REGION="us-west-2" \
  -e REDIS_HOST="$REDIS_HOST" \
  -e NEAR_API_KEY="$NEAR_API_KEY" \
  -e MODELS="${MODELS}" \
  -e MODEL_TRIAGING="${MODEL_TRIAGING}" \
  -e RATE_LIMITING_ENABLED="0" \
  -e ENV="local" \
  -e AI_CHAT_PREMIUM_HOST="127.0.0.1:8000" \
  -e LOG_LEVEL="INFO" \
  -e CONVERSATION_API="1" \
  -e SEARCH_ENABLED="1" \
  -e ALLOWED_LANGUAGES='["en","fr","de","it","es"]' \
  -e BRAVE_SEARCH_API_KEY="${BRAVE_SEARCH_API_KEY}" \
  -e BRAVE_SEARCH_RH_API_KEY="${BRAVE_SEARCH_RH_API_KEY}" \
  -e VLLM_API_KEY="${VLLM_API_KEY}" \
  -e FUNCTION_CALLING_ENABLED="${FUNCTION_CALLING_ENABLED}" \
  -e USE_VLLM_CHAT_API="${USE_VLLM_CHAT_API}" \
  -e CONVERSATION_TITLE_ENABLED="${CONVERSATION_TITLE_ENABLED}" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e USE_CONTENT_SEARCH="1" \
  -e TITLE_MODEL="${TITLE_MODEL}" \
  -e ANDROCLES_MODEL_ADDRESS="${ANDROCLES_MODEL_ADDRESS}" \
  -e ANALYTICS_MODEL_ADDRESS="${ANALYTICS_MODEL_ADDRESS}" \
  -e MCP_SERVERS="$MCP_SERVERS" \
  -e SEARCH_API_ALLOWED_CORS_ORIGINS="$SEARCH_API_ALLOWED_CORS_ORIGINS" \
  -e SHARE_S3_BUCKET="$SHARE_S3_BUCKET" \
  -e SHARE_VIEWER_ORIGIN="$SHARE_VIEWER_ORIGIN" \
  -e SHARE_MAX_CIPHERTEXT_BYTES="$SHARE_MAX_CIPHERTEXT_BYTES" \
  -e ENABLE_BRAVE_SUMMARY="$ENABLE_BRAVE_SUMMARY" \
  -e DEEP_RESEARCH_URL="$DEEP_RESEARCH_URL" \
  -e DEEP_RESEARCH_ENABLED="$DEEP_RESEARCH_ENABLED" \
  -e ENABLE_DYNAMIC_LEO="true" \
  -e DYNAMIC_LEO_CONFIG="$DYNAMIC_LEO_CONFIG" \
  -e DYNAMIC_LEO_LAST_N_USER_TURNS="5" \
  -e DYNAMIC_LEO_EMBEDDING_SIMILARITY_THRESHOLD="0.72" \
  -e DYNAMIC_LEO_EMBEDDING_MODEL="$DYNAMIC_LEO_EMBEDDING_MODEL" \
  -e INTERNAL_BASE_URL="http://host.docker.internal:8000" \
  ai-gateway uvicorn aichat.serve.api_server:app --host 0.0.0.0 --port 8000
