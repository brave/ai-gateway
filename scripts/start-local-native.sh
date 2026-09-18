#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

native_host() {
  case "${1:-}" in
    host.docker.internal | "") echo "localhost" ;;
    *) echo "$1" ;;
  esac
}

normalize_docker_host_urls() {
  local value="$1"
  value="${value//host.docker.internal/localhost}"
  echo "$value"
}

MODELS_DEFAULT='{}'
MODEL_TRIAGING_DEFAULT='{}'
MCP_SERVERS_DEFAULT='[]'

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
export AWS_SESSION_TOKEN="${AWS_SESSION_TOKEN:-}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export AWS_REGION="${AWS_REGION:-us-west-2}"
export NEAR_API_KEY="${NEAR_API_KEY:-}"
export MODELS="${MODELS:-${MODELS_DEFAULT}}"
export MODEL_TRIAGING="${MODEL_TRIAGING:-${MODEL_TRIAGING_DEFAULT}}"
export RATE_LIMITING_ENABLED="${RATE_LIMITING_ENABLED:-0}"
export ENV="${ENV:-local}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export CONVERSATION_API="${CONVERSATION_API:-1}"
export SEARCH_ENABLED="${SEARCH_ENABLED:-1}"
export ALLOWED_LANGUAGES="${ALLOWED_LANGUAGES:-[\"en\",\"fr\",\"de\",\"it\",\"es\"]}"
export BRAVE_SEARCH_API_KEY="${BRAVE_SEARCH_API_KEY:-}"
export BRAVE_SEARCH_RH_API_KEY="${BRAVE_SEARCH_RH_API_KEY:-}"
export VLLM_API_KEY="${VLLM_API_KEY:-}"
export FUNCTION_CALLING_ENABLED="${FUNCTION_CALLING_ENABLED:-1}"
export USE_VLLM_CHAT_API="${USE_VLLM_CHAT_API:-1}"
export CONVERSATION_TITLE_ENABLED="${CONVERSATION_TITLE_ENABLED:-1}"
export HF_TOKEN="${HF_TOKEN:-}"
export USE_CONTENT_SEARCH="${USE_CONTENT_SEARCH:-1}"
export TITLE_MODEL="${TITLE_MODEL:-}"
export ANDROCLES_MODEL_ADDRESS="${ANDROCLES_MODEL_ADDRESS:-}"
export ANALYTICS_MODEL_ADDRESS="${ANALYTICS_MODEL_ADDRESS:-}"
export MCP_SERVERS="${MCP_SERVERS:-${MCP_SERVERS_DEFAULT}}"
export SEARCH_API_ALLOWED_CORS_ORIGINS="${SEARCH_API_ALLOWED_CORS_ORIGINS:-}"
export ENABLE_BRAVE_SUMMARY="${ENABLE_BRAVE_SUMMARY:-false}"
_deep_research_url="$(normalize_docker_host_urls "${DEEP_RESEARCH_URL:-http://localhost:8080}")"
export DEEP_RESEARCH_URL="$_deep_research_url"
export DEEP_RESEARCH_ENABLED="${DEEP_RESEARCH_ENABLED:-false}"
export INTERNAL_BASE_URL="http://localhost:8000"

exec poetry run uvicorn aichat.serve.api_server:app --host 0.0.0.0 --port 8000
