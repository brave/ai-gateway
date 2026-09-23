import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aichat_media.media_sandbox import SandboxOpError, SandboxWorkerError, get_pool

logger = logging.getLogger(__name__)

router = APIRouter()


class SttDecodeRequest(BaseModel):
    audio_b64: str
    max_duration_seconds: float


@router.post("/v1/stt/decode")
async def stt_decode(req: SttDecodeRequest):
    try:
        result = await get_pool().call(
            "stt_decode",
            {
                "audio_b64": req.audio_b64,
                "max_duration_seconds": req.max_duration_seconds,
            },
        )
    except SandboxOpError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    except SandboxWorkerError as e:
        logger.error("Media sandbox worker failed hard: %s", e)
        return JSONResponse(status_code=503, content={"error": str(e)})
    return JSONResponse(status_code=200, content=result)
