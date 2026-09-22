import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from aichat.protocol.open_ai_protocol import ErrorCode
from aichat.serve.common_api import (
    create_error_response,
    require_internal_models_api_key,
)
from aichat.serve.rate_limiting import rate_limit_route
from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.system_one.generator import run_system_one

logger = logging.getLogger(__name__)

v1_router = APIRouter()


@v1_router.post("/systemone")
@rate_limit_route(
    config_key="system_one",
    route_path="/systemone",
    error_message="Daily rate limit exceeded for systemone endpoint",
)
async def v1_systemone(raw_request: Request) -> JSONResponse:
    """System One typed decisions via a configured Triton backend."""
    if auth_err := require_internal_models_api_key(
        raw_request.headers.get("Authorization")
    ):
        return auth_err
    try:
        body = await raw_request.json()
    except Exception as e:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            f"Invalid JSON: {e!s}",
        )

    if not isinstance(body, dict):
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            "request body must be a JSON object",
        )

    model = body.get("model")
    if not model:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            "model parameter is required for systemone",
        )
    if model not in model_settings.models:
        return create_error_response(
            ErrorCode.MODEL_NOT_FOUND,
            f"model is not supported - {model}",
        )
    if model_settings.models.get(model, {}).get("type", "") != "system_one":
        return create_error_response(
            ErrorCode.MODEL_NOT_FOUND,
            f"model does not support systemone - {model}",
        )

    if "state" not in body:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            "state parameter is required",
        )
    if "questions" not in body:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            "questions parameter is required",
        )

    model_override = None
    if "model_override" in body:
        raw_override = body["model_override"]
        if not isinstance(raw_override, str) or not raw_override.strip():
            return create_error_response(
                ErrorCode.BAD_REQUEST_ERROR,
                "model_override must be a non-empty string",
            )
        model_override = raw_override.strip()

    if "route_only" in body and not isinstance(body["route_only"], bool):
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            "route_only must be a boolean",
        )
    if "include_routing" in body and not isinstance(body["include_routing"], bool):
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            "include_routing must be a boolean",
        )

    cfg = model_settings.models[model]
    response_model = str(cfg.get("response_model") or model)

    try:
        response = await run_system_one(
            model_id=model,
            state=body["state"],
            questions=body["questions"],
            model_override=model_override,
            route_only=bool(body.get("route_only")),
            response_model=response_model,
            include_routing=bool(body.get("include_routing")),
        )
        return JSONResponse(content=response)
    except ValueError as e:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            str(e),
        )
    except Exception as e:
        logger.exception("System One inference failed")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"System One inference failed: {e!s}",
        )
