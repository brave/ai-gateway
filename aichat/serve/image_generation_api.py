import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from litellm import ImageResponse

from aichat.protocol.open_ai_protocol import ErrorCode
from aichat.serve.common_api import (
    create_error_response,
    require_internal_models_api_key,
)
from aichat.serve.rate_limiting import rate_limit_route
from aichat.serve.services.image_gen.generator import generate_image
from aichat.serve.services.model_settings import model_settings

logger = logging.getLogger(__name__)

v1_router = APIRouter()


@v1_router.post("/images/generations", response_model=ImageResponse)
@rate_limit_route(
    config_key="image_generation",
    route_path="/images/generations",
    error_message="Daily rate limit exceeded for image generation endpoint",
)
async def v1_images_generations(
    raw_request: Request,
) -> ImageResponse | JSONResponse:
    """OpenAI-compatible image generation passthrough endpoint."""
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
            "model parameter is required for image generation",
        )

    if model not in model_settings.models:
        return create_error_response(
            ErrorCode.MODEL_NOT_FOUND,
            f"model is not supported - {model}",
        )

    try:
        response = await generate_image(**body)
        return response
    except Exception as e:
        logger.exception("Image generation failed")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Image generation failed: {e!s}",
        )
