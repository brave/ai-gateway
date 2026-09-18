import logging
from typing import Any

import httpx
import sentry_sdk

from aichat.llm.metrics import INTERNAL_REQUEST_RETRY_TOTAL
from aichat.serve.internal_settings import internal_settings

logger = logging.getLogger(__name__)


_MAX_ATTEMPTS = 2


def _report_unavailable(path: str, reason: str, detail: str) -> None:
    logger.error("aichat-internal %s unavailable (%s): %s", path, reason, detail)
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("aichat_internal_unavailable", "true")
        scope.set_tag("aichat_internal_path", path)
        scope.set_tag("aichat_internal_reason", reason)
        scope.fingerprint = ["aichat-internal-unavailable", path, reason]
        sentry_sdk.capture_message(
            f"aichat-internal {path} unavailable ({reason})",
            level="error",
        )


async def auth_verify(
    client: httpx.AsyncClient,
    authorization: str | None,
    digest: str | None,
    body_sha256_b64: str,
    x_forwarded_host: str | None,
    x_brave_key: str | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return await _post(
        client,
        "/1/auth",
        {
            "authorization": authorization,
            "digest": digest,
            "body_sha256_b64": body_sha256_b64,
            "x_forwarded_host": x_forwarded_host,
            "x_brave_key": x_brave_key,
            "metadata": metadata,
        },
        idempotent=True,
    )


async def sku_verify(
    client: httpx.AsyncClient,
    sku_credential: str | None,
    service_key_id: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any] | None:
    return await _post(
        client,
        "/1/sku_verification",
        {
            "sku_credential": sku_credential,
            "service_key_id": service_key_id,
            "idempotency_key": idempotency_key,
        },
        idempotent=True,
    )


async def rate_limit_salts(client: httpx.AsyncClient) -> dict[str, Any] | None:
    return await _get(client, "/1/rate_limit_salts", idempotent=True)


async def rate_limit_check(
    client: httpx.AsyncClient,
    *,
    model: str,
    hashed_ip_current: str,
    hashed_ip_previous: str | None,
    is_premium_host: bool,
    is_content_agent_request: bool,
    is_free_model: bool,
    maximum_requests: int,
    interval_in_seconds: int,
    force_error_rate_limit: bool,
    is_premium_request: bool = False,
) -> dict[str, Any] | None:
    return await _post(
        client,
        "/1/rate_limit",
        {
            "model": model,
            "hashed_ip_current": hashed_ip_current,
            "hashed_ip_previous": hashed_ip_previous,
            "is_premium_host": is_premium_host,
            "is_content_agent_request": is_content_agent_request,
            "is_free_model": is_free_model,
            "maximum_requests": maximum_requests,
            "interval_in_seconds": interval_in_seconds,
            "force_error_rate_limit": force_error_rate_limit,
            "is_premium_request": is_premium_request,
        },
    )


async def rate_limit_content_agent(
    client: httpx.AsyncClient,
    *,
    hashed_ip_current: str,
    hashed_ip_previous: str | None,
) -> dict[str, Any] | None:
    return await _post(
        client,
        "/1/rate_limit/content_agent",
        {
            "hashed_ip_current": hashed_ip_current,
            "hashed_ip_previous": hashed_ip_previous,
        },
    )


async def rate_limit_route(
    client: httpx.AsyncClient,
    *,
    route: str,
    hashed_ip_current: str,
    hashed_ip_previous: str | None,
    daily_limit: int | None = None,
    config_key: str | None = None,
) -> dict[str, Any] | None:
    return await _post(
        client,
        "/1/rate_limit/route",
        {
            "route": route,
            "hashed_ip_current": hashed_ip_current,
            "hashed_ip_previous": hashed_ip_previous,
            "daily_limit": daily_limit,
            "config_key": config_key,
        },
    )


async def rate_limit_automatic_mode_peek(
    client: httpx.AsyncClient,
    *,
    hashed_ip_current: str,
    hashed_ip_previous: str | None,
) -> dict[str, Any] | None:
    return await _post(
        client,
        "/1/rate_limit/automatic_mode/peek",
        {
            "hashed_ip_current": hashed_ip_current,
            "hashed_ip_previous": hashed_ip_previous,
        },
    )


async def rate_limit_automatic_mode_check(
    client: httpx.AsyncClient,
    *,
    hashed_ip_current: str,
    hashed_ip_previous: str | None,
) -> dict[str, Any] | None:
    return await _post(
        client,
        "/1/rate_limit/automatic_mode",
        {
            "hashed_ip_current": hashed_ip_current,
            "hashed_ip_previous": hashed_ip_previous,
        },
    )


async def _post(
    client: httpx.AsyncClient,
    path: str,
    body: dict[str, Any],
    *,
    idempotent: bool = False,
) -> dict[str, Any] | None:
    url = internal_settings.internal_base_url.rstrip("/") + path
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = await client.post(
                url,
                json=body,
                timeout=internal_settings.internal_request_timeout_seconds,
            )
        except httpx.RemoteProtocolError as e:
            if not _handle_retryable_error(path, e, attempt):
                return None
            continue
        except httpx.ReadError as e:
            if not idempotent:
                _report_unavailable(path, type(e).__name__, str(e))
                return None
            if not _handle_retryable_error(path, e, attempt):
                return None
            continue
        except httpx.HTTPError as e:
            _report_unavailable(path, type(e).__name__, str(e))
            return None

        if response.status_code != 200:
            _report_unavailable(
                path, f"http_{response.status_code}", response.text[:200]
            )
            return None
        if attempt > 1:
            INTERNAL_REQUEST_RETRY_TOTAL.labels(path=path, outcome="succeeded").inc()
        return response.json()
    return None


async def _get(
    client: httpx.AsyncClient, path: str, *, idempotent: bool = False
) -> dict[str, Any] | None:
    url = internal_settings.internal_base_url.rstrip("/") + path
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = await client.get(
                url, timeout=internal_settings.internal_request_timeout_seconds
            )
        except httpx.RemoteProtocolError as e:
            if not _handle_retryable_error(path, e, attempt):
                return None
            continue
        except httpx.ReadError as e:
            if not idempotent:
                _report_unavailable(path, type(e).__name__, str(e))
                return None
            if not _handle_retryable_error(path, e, attempt):
                return None
            continue
        except httpx.HTTPError as e:
            _report_unavailable(path, type(e).__name__, str(e))
            return None

        if response.status_code != 200:
            _report_unavailable(
                path, f"http_{response.status_code}", response.text[:200]
            )
            return None
        if attempt > 1:
            INTERNAL_REQUEST_RETRY_TOTAL.labels(path=path, outcome="succeeded").inc()
        return response.json()
    return None


def _handle_retryable_error(path: str, e: httpx.HTTPError, attempt: int) -> bool:
    """Returns True if the caller should retry, False if it should give up."""
    if attempt < _MAX_ATTEMPTS:
        logger.debug(
            "aichat-internal %s %s on attempt %d/%d, retrying: %s",
            path,
            type(e).__name__,
            attempt,
            _MAX_ATTEMPTS,
            e,
        )
        INTERNAL_REQUEST_RETRY_TOTAL.labels(path=path, outcome="attempted").inc()
        return True

    INTERNAL_REQUEST_RETRY_TOTAL.labels(path=path, outcome="failed").inc()
    _report_unavailable(path, type(e).__name__, str(e))
    return False
