"""
aichat-media: sandboxed PDF/STT parsing service, called over HTTP by ai-gateway.

Usage:
python -m uvicorn aichat_media.api_server:app
"""

import argparse
import json
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from aichat_media.media_sandbox import shutdown_pool
from aichat_media.routes.pdf import router as pdf_router
from aichat_media.routes.stt import router as stt_router

logging.basicConfig(
    level=getattr(
        logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
    ),
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
)
logger = logging.getLogger(__name__)

VERSION = os.getenv("AICHAT_GIT_SHA", "unknown")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = True
    try:
        yield
    finally:
        await shutdown_pool()


app = FastAPI(lifespan=lifespan)


@app.get("/health/ready")
async def health_ready():
    if getattr(app.state, "ready", False):
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "not ready"},
    )


@app.get("/info")
async def info():
    return JSONResponse(status_code=status.HTTP_200_OK, content={"version": VERSION})


app.include_router(pdf_router)
app.include_router(stt_router)

Instrumentator(
    excluded_handlers=["/metrics", "/health/ready", ".*docs.*", "/openapi.json"]
).instrument(app).expose(app)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="aichat-media RESTful API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
