import logging
from urllib.parse import urlparse

import fastapi
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from aichat.protocol.open_ai_protocol import ErrorCode
from aichat.serve.auth import check_x_brave_key
from aichat.serve.common_api import (
    check_requests_common,
    create_base_common_params,
    create_error_response,
)
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.services import near
from aichat.serve.services.model_settings import model_settings

logger = logging.getLogger(__name__)


v1_router = APIRouter()


def _get_model_config(model_name: str) -> dict | None:
    """Return the model config if the model exists and supports e2ee, else None."""
    model_cfg = model_settings.models.get(model_name)
    if not model_cfg or not model_cfg.get("e2ee_support"):
        return None
    return model_cfg


def _build_gateway_url(address: str) -> str:
    parsed = urlparse(address)
    return f"https://{parsed.netloc}/ohttp"


@v1_router.get("/models/{model_name}/ohttp_config")
async def get_model_ohttp_config(
    model_name: str,
    raw_request: Request,
    is_valid_x_brave_key: bool = fastapi.Depends(check_x_brave_key),
):
    raw_request.state.model = model_name

    if not is_valid_x_brave_key:
        return create_error_response(ErrorCode.INVALID_AUTH_KEY, "Invalid services key")

    model_cfg = _get_model_config(model_name)
    if model_cfg is None:
        return create_error_response(ErrorCode.MODEL_NOT_FOUND, "Not Found")

    config = await near.verify_and_get_ohttp_config(model_name)
    if config is None:
        return create_error_response(ErrorCode.INTERNAL_ERROR, "Internal server error")

    return config


async def create_ohttp_common_params(
    model_name: str,
    raw_request: Request,
    common: dict = fastapi.Depends(create_base_common_params),
) -> dict:
    common["model"] = model_name
    raw_request.state.model = model_name
    return common


@v1_router.post("/models/{model_name}/relay")
async def relay_model_ohttp(
    model_name: str,
    raw_request: Request,
    common: dict = fastapi.Depends(create_ohttp_common_params),
):
    error = await check_requests_common(raw_request, common)
    if error:
        return error

    model_cfg = _get_model_config(model_name)
    if model_cfg is None:
        return create_error_response(ErrorCode.MODEL_NOT_FOUND, "Not Found")

    if await near.verify_and_get_ohttp_config(model_name) is None:
        return create_error_response(ErrorCode.INTERNAL_ERROR, "Internal server error")

    near_api_key = external_service_settings.near_api_key
    if not near_api_key:
        logger.warning("near_api_key is not configured; refusing relay request.")
        return create_error_response(ErrorCode.INTERNAL_ERROR, "Internal server error")

    gateway_url = _build_gateway_url(model_cfg["address"])

    headers = {
        "Authorization": f"Bearer {near_api_key}",
        "content-type": raw_request.headers.get("content-type", "message/ohttp-req"),
    }

    body = await raw_request.body()

    try:
        client = httpx.AsyncClient(timeout=300)
        upstream = await client.send(
            client.build_request("POST", gateway_url, headers=headers, content=body),
            stream=True,
        )
    except httpx.HTTPError as exc:
        logger.warning(
            f"Relay upstream request failed for {model_name}: {type(exc).__name__}: {exc}"
        )
        return create_error_response(ErrorCode.BAD_GATEWAY, "Bad gateway")

    upstream_content_type = upstream.headers.get("content-type", "message/ohttp-res")

    async def stream_and_close():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_and_close(),
        status_code=upstream.status_code,
        media_type=upstream_content_type,
    )
