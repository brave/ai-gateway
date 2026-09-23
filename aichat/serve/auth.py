import base64
import json
import logging
import secrets
from hashlib import sha256

import blake3
import fastapi
from fastapi import Request

from aichat.serve import internal_client
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.internal_settings import internal_settings
from aichat.serve.server_settings import server_settings

logger = logging.getLogger(__name__)


async def _request_auth_verdict(
    raw_request: Request,
    authorization: str | None = fastapi.Header(None),
    digest: str | None = fastapi.Header(None),
    x_forwarded_host: str | None = fastapi.Header(None),
    x_brave_key: str | None = fastapi.Header(None),
) -> dict | None:
    """Single source of truth for the aichat-internal /1/auth verdict.

    FastAPI memoizes Depends(callable) per request by callable identity,
    Returns None on failure or when internal_api_enabled is False.
    """
    if not internal_settings.internal_api_enabled:
        return None
    body_bytes = await raw_request.body()
    body_sha256_b64 = base64.b64encode(sha256(body_bytes).digest()).decode("utf-8")
    metadata = None
    try:
        body = json.loads(body_bytes)
        # Relay non-message fields only; never forward conversation content.
        metadata = {k: v for k, v in body.items() if k != "messages"}
    except (ValueError, AttributeError, TypeError):
        pass
    return await internal_client.auth_verify(
        raw_request.state.httpx_client,
        authorization=authorization,
        digest=digest,
        body_sha256_b64=body_sha256_b64,
        x_forwarded_host=x_forwarded_host,
        x_brave_key=x_brave_key,
        metadata=metadata,
    )


def check_premium_host(
    host: str = fastapi.Header(None), x_forwarded_host: str = fastapi.Header(None)
) -> bool:
    premium_host = external_service_settings.ai_chat_premium_host
    return host == premium_host or x_forwarded_host == premium_host


async def check_sku_credential(
    raw_request: Request,
    sku_credential: str = fastapi.Cookie(None, alias="__Secure-sku#brave-leo-premium"),
    auth_verdict: dict | None = fastapi.Depends(_request_auth_verdict),
) -> bool:

    if server_settings.env == "local":
        return True
    if not internal_settings.internal_api_enabled:
        # Self-host mode: no premium tier without aichat-internal.
        return False
    if not sku_credential:
        return False

    try:
        body = json.loads(await raw_request.body())
        messages = body.get("messages")
        model = body.get("model")
    except (ValueError, AttributeError):
        messages = None
        model = None
    idempotency_key = create_idempotency_key(messages, model)

    service_key_id = (auth_verdict or {}).get("service_key_id") or "unknown"
    verdict = await internal_client.sku_verify(
        raw_request.state.httpx_client,
        sku_credential=sku_credential,
        service_key_id=service_key_id,
        idempotency_key=idempotency_key,
    )
    if verdict is None:
        logger.error("aichat-internal unavailable for SKU check")
        raise fastapi.HTTPException(
            status_code=503, detail="premium backend unavailable"
        )
    return bool(verdict.get("allowed", False))


async def check_x_brave_key(
    verdict: dict | None = fastapi.Depends(_request_auth_verdict),
):
    if not internal_settings.internal_api_enabled:
        return True
    if verdict is None:
        raise fastapi.HTTPException(status_code=500, detail="backend unavailable")
    return bool(verdict.get("brave_key_allowed", False))


async def check_brave_services_key_v2(
    raw_request: Request,
    verdict: dict | None = fastapi.Depends(_request_auth_verdict),
):
    if not internal_settings.internal_api_enabled:
        # Self-host mode: no service-key gating.
        return True
    if verdict is None:
        logger.error("aichat-internal unavailable for service-key check")
        raise fastapi.HTTPException(status_code=500, detail="backend unavailable")

    if verdict.get("service_key_id"):
        raw_request.state.service_key_id = verdict["service_key_id"]
    raw_request.state.model_override = bool(verdict.get("model_override", False))
    raw_request.state.model_override_fallback = (
        verdict.get("fallback_model") or "automatic"
    )
    raw_request.state.model_override_premium_fallback = verdict.get(
        "premium_fallback_model"
    )
    # Default True when older aichat-internal omits the field.
    raw_request.state.request_allowed = bool(verdict.get("request_allowed", True))
    return bool(verdict.get("service_key_allowed", False))


async def verify_service_key(raw_request: Request) -> bool:
    """Wraps check_brave_services_key_v2 for callers invoked outside FastAPI's
    dependency injection (e.g. api_key_chat_api, called directly from
    middleware rather than as a routed endpoint), so its Depends(...) default
    and HTTPException-on-unavailable don't need FastAPI's routing/exception
    machinery to work correctly.

    Reads the service-key signature from x-services-authorization rather than
    Authorization, since on this path Authorization already carries the
    caller's API key.
    """
    headers = raw_request.headers
    verdict = await _request_auth_verdict(
        raw_request,
        authorization=headers.get("x-services-authorization"),
        digest=headers.get("digest"),
        x_forwarded_host=headers.get("x-forwarded-host"),
        x_brave_key=headers.get("x-brave-key"),
    )
    try:
        return await check_brave_services_key_v2(raw_request, verdict)
    except fastapi.HTTPException:
        return False


def create_idempotency_key(
    messages: list[dict] | None,
    model: str | None,
) -> str:
    if messages is None and model is None:
        # Primarily for OHTTP where the request body is non-deterministic.
        return f"rand_{secrets.token_hex(32)}"

    payload = json.dumps(
        {"messages": messages, "model": model},
        sort_keys=True,
        separators=(",", ":"),
    )
    return blake3.blake3(payload.encode("utf-8")).hexdigest()
