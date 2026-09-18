import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from aichat.protocol.open_ai_protocol import ErrorCode
from aichat.serve.common_api import (
    create_error_response,
    require_internal_models_api_key,
)
from aichat.serve.rate_limiting import rate_limit_route
from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.tts.generator import generate_speech, stream_speech

logger = logging.getLogger(__name__)

v1_router = APIRouter()

CONTENT_TYPE_MAP = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


@v1_router.post("/audio/speech", response_model=None)
@rate_limit_route(
    config_key="tts",
    route_path="/audio/speech",
    error_message="Daily rate limit exceeded for TTS endpoint",
)
async def v1_audio_speech(
    raw_request: Request,
) -> Response | JSONResponse | StreamingResponse:
    """OpenAI-compatible TTS passthrough endpoint."""
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
            "model parameter is required for speech generation",
        )
    elif model not in model_settings.models:
        return create_error_response(
            ErrorCode.MODEL_NOT_FOUND,
            f"model is not supported - {model}",
        )
    elif model_settings.models.get(model).get("type", "") != "tts":
        return create_error_response(
            ErrorCode.MODEL_NOT_FOUND,
            f"model does not support TTS - {model}",
        )

    content_type = CONTENT_TYPE_MAP.get(
        body.get("response_format", "mp3"), "audio/mpeg"
    )

    try:
        if body.get("stream"):
            return StreamingResponse(
                await stream_speech(model, body),
                media_type=content_type,
            )

        response = await generate_speech(**body)
        if not hasattr(response, "content"):
            return create_error_response(
                ErrorCode.INTERNAL_ERROR,
                "Invalid response format from speech generation",
            )

        return Response(content=response.content, media_type=content_type)
    except Exception as e:
        logger.exception("Speech generation failed")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Speech generation failed: {e!s}",
        )
