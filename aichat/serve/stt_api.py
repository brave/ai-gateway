import logging

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from aichat.protocol.open_ai_protocol import ErrorCode
from aichat.serve.common_api import (
    create_error_response,
    require_internal_models_api_key,
)
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.rate_limiting import rate_limit_route
from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.stt.generator import transcribe_upload
from aichat.serve.services.stt.validation import (
    read_stt_upload_limited,
    sanitize_stt_filename,
    validate_pcm_wav_upload,
)

logger = logging.getLogger(__name__)

v1_router = APIRouter()


def _ms_to_srt_timestamp(total_ms: float) -> str:
    ms = round(total_ms)
    h, r = divmod(ms, 3_600_000)
    m, r = divmod(r, 60_000)
    s, frac = divmod(r, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{frac:03d}"


def _ms_to_vtt_timestamp(total_ms: float) -> str:
    ms = round(total_ms)
    h, r = divmod(ms, 3_600_000)
    m, r = divmod(r, 60_000)
    s, frac = divmod(r, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{frac:03d}"


@v1_router.post("/audio/transcriptions", response_model=None)
@rate_limit_route(
    config_key="stt",
    route_path="/audio/transcriptions",
    error_message="Daily rate limit exceeded for transcriptions endpoint",
)
async def v1_audio_transcriptions(
    raw_request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str | None = Form("json"),
    temperature: float | None = Form(0.0),
) -> Response | JSONResponse:
    """Parakeet via Triton (passthrough infer, same pattern as embeddings). Upload must be PCM WAV; other formats are rejected. language, prompt, temperature are accepted but ignored."""
    del language, prompt, temperature

    if auth_err := require_internal_models_api_key(
        raw_request.headers.get("Authorization")
    ):
        return auth_err

    if not model:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            "model parameter is required for transcription",
        )

    if model not in model_settings.models:
        return create_error_response(
            ErrorCode.MODEL_NOT_FOUND,
            f"model is not supported - {model}",
        )
    if model_settings.models.get(model, {}).get("type", "") != "speech_to_text":
        return create_error_response(
            ErrorCode.MODEL_NOT_FOUND,
            f"model does not support speech-to-text - {model}",
        )

    max_bytes = external_service_settings.max_stt_upload_size_mb * 1024 * 1024
    try:
        safe_filename = sanitize_stt_filename(file.filename)
        audio_bytes = await read_stt_upload_limited(file, max_bytes)
        validate_pcm_wav_upload(
            audio_bytes,
            max_bytes=max_bytes,
        )
    except ValueError as e:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            str(e),
        )
    except Exception as e:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            f"Failed to read upload: {e}",
        )

    fmt = response_format or "json"
    if fmt not in ("json", "text", "verbose_json", "srt", "vtt"):
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            f"unsupported response_format: {fmt}",
        )

    try:
        transcript, duration_ms = await transcribe_upload(
            model=model,
            audio_bytes=audio_bytes,
            filename=safe_filename,
        )
    except ValueError as e:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR,
            str(e),
        )
    except Exception as e:
        logger.exception("Transcription failed")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Transcription failed: {e}",
        )

    if fmt == "text":
        return Response(content=transcript, media_type="text/plain")
    if fmt == "verbose_json":
        return JSONResponse(
            content={
                "task": "transcribe",
                "language": "en",
                "duration": duration_ms / 1000.0,
                "text": transcript,
                "words": [],
            }
        )
    if fmt == "srt":
        end = _ms_to_srt_timestamp(duration_ms)
        srt_content = f"1\n00:00:00,000 --> {end}\n{transcript}\n"
        return Response(content=srt_content, media_type="text/plain")
    if fmt == "vtt":
        end = _ms_to_vtt_timestamp(duration_ms)
        vtt_content = f"WEBVTT\n\n00:00:00.000 --> {end}\n{transcript}\n"
        return Response(content=vtt_content, media_type="text/vtt")

    return JSONResponse(content={"text": transcript})
