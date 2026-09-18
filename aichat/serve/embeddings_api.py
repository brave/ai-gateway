import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from aichat.protocol.open_ai_protocol import ErrorCode
from aichat.serve.common_api import (
    create_error_response,
    require_internal_models_api_key,
)
from aichat.serve.rate_limiting import rate_limit_route
from aichat.serve.services.embeddings.generator import generate_embeddings
from aichat.serve.services.model_settings import model_settings

logger = logging.getLogger(__name__)

v1_router = APIRouter()


@v1_router.post("/embeddings")
@rate_limit_route(
    config_key="text_embedding",
    route_path="/embeddings",
    error_message="Daily rate limit exceeded for embeddings endpoint",
)
async def v1_embeddings(
    raw_request: Request,
) -> JSONResponse:
    """OpenAI-compatible embeddings passthrough endpoint."""
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

    model = body.get("model")
    if not model:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            "model parameter is required for embeddings",
        )

    input_data = body.get("input")
    if not input_data:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            "input parameter is required for embeddings",
        )

    if model not in model_settings.models:
        return create_error_response(
            ErrorCode.MODEL_NOT_FOUND,
            f"model is not supported - {model}",
        )
    elif model_settings.models.get(model).get("type", "") != "embedding":
        return create_error_response(
            ErrorCode.MODEL_NOT_FOUND,
            f"model does not support embeddings - {model}",
        )

    try:
        # Only pass model and input, ignore other fields from body
        response = await generate_embeddings(model=model, input=input_data)
        return JSONResponse(content=response)
    except Exception as e:
        logger.exception("Embeddings generation failed")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Embeddings generation failed: {e!s}",
        )
