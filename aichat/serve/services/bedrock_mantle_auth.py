"""Short-lived Bedrock Mantle bearer tokens from IAM credentials."""

import logging
import os
import time

from aws_bedrock_token_generator import provide_token

logger = logging.getLogger(__name__)

_TOKEN_CACHE_TTL_SECONDS = 45 * 60

_cached_token: str | None = None
_cached_at: float = 0.0


def get_bedrock_mantle_bearer_token() -> str:
    """
    Return a Bedrock API bearer token for bedrock-mantle OpenAI-compatible calls.

    Uses BEDROCK_MANTLE_API_KEY when set (local dev / long-term key). Otherwise
    generates a short-term token via aws-bedrock-token-generator from the AWS
    credential chain (same as existing bedrock-runtime IAM).
    """
    global _cached_token, _cached_at

    env_key = os.getenv("BEDROCK_MANTLE_API_KEY") or os.getenv(
        "AWS_BEARER_TOKEN_BEDROCK"
    )
    if env_key:
        return env_key

    now = time.monotonic()
    if _cached_token and (now - _cached_at) < _TOKEN_CACHE_TTL_SECONDS:
        return _cached_token

    token = provide_token()
    if not token:
        raise RuntimeError("Failed to generate Bedrock Mantle bearer token")

    _cached_token = token
    _cached_at = now
    logger.debug("Refreshed Bedrock Mantle bearer token cache")
    return token


def clear_bedrock_mantle_token_cache() -> None:
    """Clear cached token (for tests)."""
    global _cached_token, _cached_at
    _cached_token = None
    _cached_at = 0.0
