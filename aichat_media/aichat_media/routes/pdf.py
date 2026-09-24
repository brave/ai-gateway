import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aichat_media.media_sandbox import SandboxOpError, SandboxWorkerError, get_pool

logger = logging.getLogger(__name__)

router = APIRouter()


class PdfAnalyzeRequest(BaseModel):
    pdf_b64: str
    encoding_name: str
    max_extraction_tokens: int
    max_allowed_pages: int


@router.post("/v1/pdf/analyze")
async def pdf_analyze(req: PdfAnalyzeRequest):
    try:
        result = await get_pool().call(
            "pdf_analyze",
            {
                "pdf_b64": req.pdf_b64,
                "encoding_name": req.encoding_name,
                "max_extraction_tokens": req.max_extraction_tokens,
                "max_allowed_pages": req.max_allowed_pages,
            },
        )
    except SandboxOpError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    except SandboxWorkerError as e:
        logger.error("Media sandbox worker failed hard: %s", e)
        return JSONResponse(status_code=503, content={"error": str(e)})
    return JSONResponse(status_code=200, content=result)
