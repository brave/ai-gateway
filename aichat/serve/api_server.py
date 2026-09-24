"""
A server that provides Anthropic-compatible and custom conversation RESTful APIs. It supports:
- Completions. (Reference: https://console.anthropic.com/docs/api/reference#-v1-complete)
- Conversations.

Usage:
python3 api_server.py
"""

import argparse
import asyncio
import json
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import httpx
import sentry_sdk
import sentry_sdk.scrubber
import uvicorn

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module="pydantic",
)

from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aichat.protocol.open_ai_protocol import ATTACHMENT_URL_ERROR, ErrorCode
from aichat.serve import api_key_chat_api
from aichat.serve.brave_search_api import router as brave_search_router
from aichat.serve.common_api import create_error_response
from aichat.serve.constants import PRODUCTION
from aichat.serve.conversation_api import (
    router as conversation_router,
)
from aichat.serve.embeddings_api import v1_router as embeddings_router_v1
from aichat.serve.http_client_metrics import InstrumentedHTTPTransport
from aichat.serve.image_generation_api import v1_router as image_generation_router_v1
from aichat.serve.mcp_integration import (
    set_shared_mcp_client,
    shutdown_shared_mcp_client,
    warmup_shared_mcp_client,
)
from aichat.serve.media_client import shutdown_client as shutdown_media_client
from aichat.serve.metrics import instrument
from aichat.serve.models_api import v1_router as models_router_v1
from aichat.serve.ohttp import v1_router as ohttp_router_v1
from aichat.serve.open_ai_api import v1_router as open_ai_router_v1
from aichat.serve.passthrough_api import v1_router as passthrough_router_v1
from aichat.serve.server_settings import server_settings
from aichat.serve.services.backend import initialize_backends
from aichat.serve.share_api import v1_router as share_router_v1
from aichat.serve.stt_api import v1_router as stt_router_v1
from aichat.serve.system_one_api import v1_router as system_one_router_v1
from aichat.serve.tts_api import v1_router as tts_router_v1
from aichat.serve.utils import get_malloc_trim, periodic_malloc_trim

logging.basicConfig(
    level=getattr(logging, server_settings.log_level, logging.WARNING),
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
)
logger = logging.getLogger(__name__)


VERSION = os.getenv("AICHAT_GIT_SHA", "unknown")

sentry_denylist = sentry_sdk.scrubber.DEFAULT_DENYLIST + ["prompt", "content"]
sentry_sdk.init(
    send_default_pii=False,
    event_scrubber=sentry_sdk.scrubber.EventScrubber(
        denylist=sentry_denylist, recursive=True
    ),
    include_local_variables=False,
)


_malloc_trim = get_malloc_trim()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False

    asyncio.get_event_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=server_settings.litellm_executor_max_workers)
    )

    # Initialize all backends at startup
    initialize_backends()

    trim_task = None
    if _malloc_trim:
        trim_task = asyncio.create_task(periodic_malloc_trim(120))
        logger.info("Periodic malloc_trim enabled (every 120s)")

    # Defaults are pretty low at 100, 20 so bump them up *3
    limits = httpx.Limits(max_connections=300, max_keepalive_connections=60)
    transport = InstrumentedHTTPTransport(limits=limits)
    mcp_warmed = None
    async with httpx.AsyncClient(transport=transport) as client:
        app.state.ready = True
        mcp_warmed = await warmup_shared_mcp_client()
        if mcp_warmed is not None:
            set_shared_mcp_client(mcp_warmed)
        try:
            yield {"httpx_client": client}
        finally:
            await shutdown_shared_mcp_client(mcp_warmed)
            await shutdown_media_client()

    if trim_task:
        trim_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(OverflowError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    logger.exception("Validation error: %s", exc)
    sentry_sdk.capture_exception(exc)
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request, exc
):  # pylint: disable=unused-argument
    sentry_sdk.capture_exception(exc)
    errors = exc.errors()
    error = next((e for e in errors if e["type"] == ATTACHMENT_URL_ERROR), errors[0])
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, error["msg"])


@app.exception_handler(Exception)
async def exception_handler(request, exc):  # pylint: disable=unused-argument
    sentry_sdk.capture_exception(exc)
    message = "Internal Server Error" if server_settings.env == PRODUCTION else str(exc)
    return create_error_response(ErrorCode.INTERNAL_ERROR, message)


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.exception("ValueError: %s", exc)
    sentry_sdk.capture_exception(exc)
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))


@app.get("/health/ready")
async def health_ready():
    """
    Readiness endpoint that returns 200 OK only when the app is fully ready (backends are initialised)
    else returns a 503.
    """
    if getattr(app.state, "ready", False):
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready"},
        )


@app.get("/info")
async def info():
    return JSONResponse(status_code=status.HTTP_200_OK, content={"version": VERSION})


@app.middleware("http")
async def api_key_dispatch(request, call_next):
    if (
        server_settings.api_key_chat_enabled
        and request.url.path == "/v1/chat/completions"
    ):
        api_key = request.headers.get("x-api-key")
        if api_key is not None:
            if not api_key_chat_api.is_valid_api_key(api_key):
                return api_key_chat_api.openai_error_response(
                    401, "Invalid API key.", "invalid_api_key"
                )
            return await api_key_chat_api.handle_chat_completions(request)
    return await call_next(request)


app.include_router(conversation_router, prefix="/v1")
app.include_router(models_router_v1, prefix="/v1")
app.include_router(open_ai_router_v1, prefix="/v1")
app.include_router(image_generation_router_v1, prefix="/v1")
app.include_router(tts_router_v1, prefix="/v1")
app.include_router(embeddings_router_v1, prefix="/v1")
app.include_router(stt_router_v1, prefix="/v1")
app.include_router(system_one_router_v1, prefix="/v1")
app.include_router(brave_search_router, prefix="/search-api")
app.include_router(ohttp_router_v1, prefix="/v1")
app.include_router(passthrough_router_v1, prefix="/v1")
app.include_router(share_router_v1, prefix="/v1")

instrument(app, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="aichat RESTful API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins",
        type=json.loads,
        default=["*"],
        help="allowed origins",
    )
    parser.add_argument(
        "--allowed-methods",
        type=json.loads,
        default=["*"],
        help="allowed methods",
    )
    parser.add_argument(
        "--allowed-headers",
        type=json.loads,
        default=["*"],
        help="allowed headers",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
