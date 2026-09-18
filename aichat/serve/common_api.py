import logging
import secrets

import fastapi
from fastapi import Request
from fastapi.responses import JSONResponse

from aichat.llm.metrics import (
    AUTH_REQUESTS_BY_KEY,
    MODEL_OVERRIDE_TOTAL,
    PREMIUM_MODEL_DOWNGRADE_TOTAL,
)
from aichat.protocol.anthropic_api_protocol import (
    ErrorMessage as AnthropicErrorMessage,
)
from aichat.protocol.anthropic_api_protocol import (
    ErrorResponse as AnthropicErrorResponse,
)
from aichat.protocol.leo_api_protocol import Capability
from aichat.protocol.open_ai_protocol import ErrorCode, has_capability
from aichat.protocol.open_ai_protocol import Request as OpenAIRequest
from aichat.serve.api_version import api_version_from_path
from aichat.serve.auth import *
from aichat.serve.rate_limiting import *
from aichat.serve.rate_limiting_settings import rate_limiting_settings
from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.models import get_model_config
from aichat.serve.services.security_settings import security_settings
from aichat.serve.utils import (
    get_real_ip,
    parse_free_model_names,
)

logger = logging.getLogger(__name__)


def _peek_under_limit(peek: dict) -> bool:
    if peek.get("exceeded"):
        return False
    return int(peek.get("count", 0)) < int(peek.get("limit", 0)) + 1


def create_error_response(
    code: int, message: str, api_version: int | None = None
) -> JSONResponse:
    logger.warning(
        "Encountered error: %s on api version %s with code %d",
        message,
        api_version,
        code,
    )
    return JSONResponse(
        status_code=int(code / 100),
        content=AnthropicErrorResponse(
            type="error", error=AnthropicErrorMessage(message=message, type=code)
        ).dict(),
    )


def extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credentials:
        return None
    return credentials


def is_valid_internal_models_api_key(authorization: str | None) -> bool:
    expected = security_settings.internal_models_api_key
    if not expected:
        return True
    token = extract_bearer_token(authorization)
    if token is None:
        return False
    return secrets.compare_digest(token, expected)


def require_internal_models_api_key(
    authorization: str | None,
) -> JSONResponse | None:
    if not security_settings.internal_models_api_key:
        return None
    if is_valid_internal_models_api_key(authorization):
        return None
    return create_error_response(ErrorCode.INVALID_AUTH_KEY, "Invalid API key")


def resolve_model(
    request: OpenAIRequest,
    is_premium_host: bool = fastapi.Depends(check_premium_host),
) -> str:
    """
    Returns the model name to use for the request, downgrading premium model
    requests on the non-premium endpoint to "automatic".

    The UI sometimes sends requests for premium models to the non-premium
    endpoint without the SKU credential, so we can't tell if the user is
    actually premium. Rather than rejecting the request as INVALID_MODEL,
    downgrade to "automatic" so model selection picks an appropriate free
    model. Explicit free-model selections are preserved.
    """
    model = request.model
    if (
        not is_premium_host
        and model not in parse_free_model_names(model_settings.models)
        and not model.startswith("automatic")
    ):
        logger.warning(
            f'Premium model requested on non-premium endpoint: "{model}"  '
            "downgrading to automatic"
        )
        PREMIUM_MODEL_DOWNGRADE_TOTAL.labels(model=model).inc()
        return "automatic"
    return model


async def create_base_common_params(
    raw_request: Request,
    is_premium_host: bool = fastapi.Depends(check_premium_host),
    has_valid_premium_credential: bool = fastapi.Depends(check_sku_credential),
    x_forwarded_host: str | None = fastapi.Header(None),
    x_forwarded_for: str | None = fastapi.Header(None),
    pragma: str | None = fastapi.Header(None),
    x_amzn_waf_incorrect_header_order: str | None = fastapi.Header(None),
    is_valid_brave_services_key=fastapi.Depends(check_x_brave_key),
    is_valid_brave_services_key_v2=fastapi.Depends(check_brave_services_key_v2),
) -> dict:

    api_version = api_version_from_path(raw_request.url.path)
    raw_request.state.api_version = api_version

    return {
        "is_premium_host": is_premium_host,
        "has_valid_premium_credential": has_valid_premium_credential,
        "x_forwarded_host": x_forwarded_host,
        "x_forwarded_for": x_forwarded_for,
        "pragma": pragma,
        "x_amzn_waf_incorrect_header_order": x_amzn_waf_incorrect_header_order,
        "is_valid_brave_services_key": is_valid_brave_services_key,
        "is_valid_brave_services_key_v2": is_valid_brave_services_key_v2,
        # Set by check_brave_services_key_v2; True when internal API is off.
        "request_allowed": getattr(raw_request.state, "request_allowed", True),
        "api_version": api_version,
    }


async def create_completion_common_params(
    raw_request: Request,
    request: OpenAIRequest,
    model: str = fastapi.Depends(resolve_model),
    common: dict = fastapi.Depends(create_base_common_params),
):
    raw_request.state.model = request.model
    raw_request.state.events = getattr(request, "events", None)
    raw_request.state.messages = getattr(request, "messages", None)

    is_premium = common["is_premium_host"] and common["has_valid_premium_credential"]
    model = apply_model_override(raw_request, model, is_premium)

    common["model"] = model
    common["is_automatic_model_request"] = model == "automatic"
    common["premium_model_mismatch"] = model != request.model
    return common


def apply_model_override(raw_request: Request, model: str, is_premium: bool) -> str:
    if not getattr(raw_request.state, "model_override", False):
        return model

    premium_fallback = getattr(
        raw_request.state, "model_override_premium_fallback", None
    )
    if is_premium:
        if not premium_fallback:
            return model
        fallback = premium_fallback
    else:
        fallback = getattr(raw_request.state, "model_override_fallback", "automatic")
    key_id = str(getattr(raw_request.state, "service_key_id", "unknown"))
    MODEL_OVERRIDE_TOTAL.labels(key_id=key_id, requested_model=model).inc()
    return fallback


async def check_requests_common(
    raw_request: Request,
    common: dict,
) -> JSONResponse | None:
    """
    Performs validation on requests that applies to all completions
    """

    model = common["model"]
    real_ip = get_real_ip(common["x_forwarded_for"])
    httpx_client = getattr(raw_request.state, "httpx_client", None)

    # If the request is to the premium endpoint and the SKU credential is missing
    # or invalid, we fail the request because otherwise users could get around the
    # free tier rate limits.
    if common["is_premium_host"] and not common["has_valid_premium_credential"]:
        logger.warning("Invalid or missing SKU credential.")
        return create_error_response(
            ErrorCode.INVALID_SKU_CREDENTIAL, "Invalid or missing SKU credential"
        )

    if rate_limiting_settings.rate_limiting_enabled and raw_request:
        is_content_agent_request = has_capability(
            getattr(raw_request.state, "capability", None), Capability.content_agent
        )
        rate_limit_verdict = await check_rate_limit(
            raw_request,
            model,
            common["is_premium_host"],
            real_ip,
            is_content_agent_request,
            has_valid_premium_credential=common["has_valid_premium_credential"],
        )
        if not rate_limit_verdict.allowed:
            return create_error_response(
                ErrorCode.RATE_LIMIT,
                f"Exceeded the rate limit for model {model}",
            )
        if rate_limit_verdict.fallback_model:
            model = rate_limit_verdict.fallback_model
            common["model"] = model

    if not common["is_valid_brave_services_key_v2"]:
        return create_error_response(ErrorCode.INVALID_AUTH_KEY, "Invalid services key")

    if not common.get("request_allowed", True):
        # Generic 400 — the specific reason is intentionally not surfaced.
        return create_error_response(ErrorCode.INVALID_REQUEST, "Invalid request")

    if (not model.startswith("automatic")) and model not in model_settings.models:
        return create_error_response(
            ErrorCode.MODEL_NOT_FOUND,
            f"model is not supported - {model}",
        )

    # Require valid premium credential for premium models
    capability = getattr(raw_request.state, "capability", None) if raw_request else None
    if (
        model not in parse_free_model_names(model_settings.models)
        and not common["has_valid_premium_credential"]
        and not (
            has_capability(capability, Capability.content_agent)
            and model in model_settings.models
            and get_model_config(model) is not None
            and get_model_config(model).content_agent_support
        )
        and not (
            common.get("is_automatic_model_request")
            and _peek_under_limit(
                await check_automatic_mode_daily_limit(
                    real_ip, httpx_client=httpx_client
                )
            )
        )
    ):
        logger.warning(
            f'Model "{model}" is only supported for premium users. X-Forwarded-Host: {common["x_forwarded_host"]}'
        )
        return create_error_response(
            ErrorCode.INVALID_MODEL, "model is only supported for premium users"
        )

    # Check content agent specific rate limits for free users
    if (
        rate_limiting_settings.rate_limiting_enabled
        and has_capability(capability, Capability.content_agent)
        and not (common["is_premium_host"] and common["has_valid_premium_credential"])
    ) and not await check_content_agent_rate_limits(real_ip, httpx_client=httpx_client):
        return create_error_response(
            ErrorCode.RATE_LIMIT,
            "Content agent request limit exceeded",
        )

    AUTH_REQUESTS_BY_KEY.labels(
        key_id=str(getattr(raw_request.state, "service_key_id", "unknown")),
        model=model,
        is_premium=common["is_premium_host"] and common["has_valid_premium_credential"],
        api_version=common.get("api_version", "unknown"),
    ).inc()
