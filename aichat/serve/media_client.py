import json
import logging
from typing import Any

import httpx

from aichat.serve.media_settings import media_settings
from aichat.serve.metrics import MEDIA_REQUEST_RETRY_TOTAL

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 2
_MAX_REQUEST_BYTES = 64 * 1024 * 1024
_MAX_RESPONSE_BYTES = 128 * 1024 * 1024
_MAX_ERROR_CHARS = 200

_client: httpx.AsyncClient | None = None


class SandboxWorkerError(RuntimeError):
    """The media sandbox service is unreachable or failed unexpectedly."""


class SandboxOpError(SandboxWorkerError):
    """The media sandbox service rejected the request as malformed input."""


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


def _encode_request(path: str, body: dict[str, Any]) -> bytes:
    for value in body.values():
        if isinstance(value, str) and len(value.encode("utf-8")) > _MAX_REQUEST_BYTES:
            raise SandboxWorkerError(
                f"ai-gateway-media-processor {path} request exceeds {_MAX_REQUEST_BYTES} bytes"
            )
    try:
        data = json.dumps(body, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as e:
        raise SandboxWorkerError(
            f"ai-gateway-media-processor {path} request could not be encoded"
        ) from e
    if len(data) > _MAX_REQUEST_BYTES:
        raise SandboxWorkerError(
            f"ai-gateway-media-processor {path} request exceeds {_MAX_REQUEST_BYTES} bytes"
        )
    return data


async def _read_response(path: str, response: httpx.Response) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            declared = None
        if declared is not None and declared > _MAX_RESPONSE_BYTES:
            raise SandboxWorkerError(
                f"ai-gateway-media-processor {path} response exceeds {_MAX_RESPONSE_BYTES} bytes"
            )
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > _MAX_RESPONSE_BYTES:
            raise SandboxWorkerError(
                f"ai-gateway-media-processor {path} response exceeds {_MAX_RESPONSE_BYTES} bytes"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _json_object(path: str, status_code: int, raw: bytes) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SandboxWorkerError(
            f"ai-gateway-media-processor {path} returned {status_code} with a non-JSON body"
        ) from e
    if not isinstance(data, dict):
        raise SandboxWorkerError(
            f"ai-gateway-media-processor {path} returned {status_code} with a non-object JSON body"
        )
    return data


def _handle_retryable_error(path: str, e: httpx.HTTPError, attempt: int) -> bool:
    """Returns True if the caller should retry, False if it should give up."""
    if attempt < _MAX_ATTEMPTS:
        logger.debug(
            "ai-gateway-media-processor %s %s on attempt %d/%d, retrying: %s",
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
    payload = _encode_request(path, body)
    client = _get_client()
    last_error: httpx.HTTPError | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            async with client.stream(
                "POST",
                path,
                content=payload,
                headers={"Content-Type": "application/json"},
                timeout=media_settings.media_request_timeout_seconds,
            ) as response:
                raw = await _read_response(path, response)
                status_code = response.status_code
        except httpx.TimeoutException as e:
            raise SandboxWorkerError(
                f"ai-gateway-media-processor timed out for {path} ({type(e).__name__})"
            ) from e
        except httpx.ConnectError as e:
            last_error = e
            if _handle_retryable_error(path, e, attempt):
                continue
            raise SandboxWorkerError(
                f"ai-gateway-media-processor unreachable for {path} after {attempt} attempts ({type(e).__name__})"
            ) from e
        except httpx.HTTPError as e:
            raise SandboxWorkerError(
                f"ai-gateway-media-processor unreachable for {path} ({type(e).__name__})"
            ) from e

        if status_code == 422:
            data = _json_object(path, status_code, raw)
            error = data.get("error")
            if not isinstance(error, str) or not error:
                error = "media op failed"
            elif len(error) > _MAX_ERROR_CHARS:
                error = error[:_MAX_ERROR_CHARS]
            raise SandboxOpError(error)
        if status_code != 200:
            raise SandboxWorkerError(
                f"ai-gateway-media-processor {path} returned {status_code}"
            )
        if attempt > 1:
            MEDIA_REQUEST_RETRY_TOTAL.labels(path=path, outcome="succeeded").inc()
        return _json_object(path, status_code, raw)
    detail = type(last_error).__name__ if last_error is not None else "unknown"
    raise SandboxWorkerError(
        f"ai-gateway-media-processor unreachable for {path} ({detail})"
    )


async def call_pdf_analyze(args: dict[str, Any]) -> dict[str, Any]:
    return await _call("/v1/pdf/analyze", args)


async def call_stt_decode(args: dict[str, Any]) -> dict[str, Any]:
    return await _call("/v1/stt/decode", args)
