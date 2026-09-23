"""HTTP client for the aichat-media service (sandboxed PDF/STT parsing).

Preserves the same call signature/exceptions the in-process media sandbox
pool used to raise, so callers in pdf.py and stt/generator.py don't need to
change their error handling.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from aichat.llm.metrics import MEDIA_REQUEST_RETRY_TOTAL
from aichat.serve.media_settings import media_settings

logger = logging.getLogger(__name__)

# Both media ops are pure functions of their input bytes, so retrying a
# transient connection error is always safe (no idempotency concerns).
_MAX_ATTEMPTS = 2
_RETRYABLE_ERRORS = (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError)


class SandboxWorkerError(RuntimeError):
    pass


class SandboxOpError(SandboxWorkerError):
    pass


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=media_settings.media_base_url)
    return _client


async def shutdown_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _handle_retryable_error(path: str, e: httpx.HTTPError, attempt: int) -> bool:
    """Returns True if the caller should retry, False if it should give up."""
    if attempt < _MAX_ATTEMPTS:
        logger.debug(
            "aichat-media %s %s on attempt %d/%d, retrying: %s",
            path,
            type(e).__name__,
            attempt,
            _MAX_ATTEMPTS,
            e,
        )
        MEDIA_REQUEST_RETRY_TOTAL.labels(path=path, outcome="attempted").inc()
        return True

    MEDIA_REQUEST_RETRY_TOTAL.labels(path=path, outcome="failed").inc()
    return False


async def _call(path: str, body: dict[str, Any]) -> dict[str, Any]:
    client = _get_client()
    last_error: httpx.HTTPError | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = await client.post(
                path, json=body, timeout=media_settings.media_request_timeout_seconds
            )
        except _RETRYABLE_ERRORS as e:
            last_error = e
            if _handle_retryable_error(path, e, attempt):
                continue
            raise SandboxWorkerError(
                f"aichat-media unreachable for {path} after {attempt} attempts: {e}"
            ) from e
        except httpx.HTTPError as e:
            raise SandboxWorkerError(f"aichat-media unreachable for {path}: {e}") from e

        if response.status_code == 422:
            raise SandboxOpError(response.json().get("error", "media op failed"))
        if response.status_code != 200:
            raise SandboxWorkerError(
                f"aichat-media {path} returned {response.status_code}: {response.text}"
            )
        if attempt > 1:
            MEDIA_REQUEST_RETRY_TOTAL.labels(path=path, outcome="succeeded").inc()
        return response.json()

    # Unreachable: the loop above always returns or raises.
    raise SandboxWorkerError(f"aichat-media unreachable for {path}: {last_error}")


async def call_pdf_analyze(args: dict[str, Any]) -> dict[str, Any]:
    return await _call("/v1/pdf/analyze", args)


async def call_stt_decode(args: dict[str, Any]) -> dict[str, Any]:
    return await _call("/v1/stt/decode", args)
