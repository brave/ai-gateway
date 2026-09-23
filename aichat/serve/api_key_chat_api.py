import logging
import re
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types.chat import CompletionCreateParams
from pydantic import TypeAdapter
from starlette.requests import Request

from aichat.serve.backend.litellm import apply_claude_upstream_sampling_params
from aichat.serve.common_api import extract_bearer_token
from aichat.serve.services.backend import get_backend

logger = logging.getLogger(__name__)

v1_router = APIRouter()

API_KEY_PREFIX = "brv_live_"
_API_KEY_PATTERN = re.compile(rf"^{re.escape(API_KEY_PREFIX)}[A-Za-z0-9]{{32,64}}$")

_request_adapter = TypeAdapter(CompletionCreateParams)
_PASSTHROUGH_EXCLUDE = {"model", "messages", "stream"}


def openai_error_response(
    status_code: int, message: str, error_type: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": error_type}},
    )


def is_valid_api_key(api_key: str) -> bool:
    return bool(_API_KEY_PATTERN.match(api_key))


def api_key_from_authorization(authorization: str | None) -> str | None:
    token = extract_bearer_token(authorization)
    if token is None or not token.startswith(API_KEY_PREFIX):
        return None
    return token


async def _stream_chunks(response: AsyncIterator) -> AsyncIterator[str]:
    async for chunk in response:
        try:
            yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as e:
            logger.warning(f"Failed to serialize chunk: {e}")
    yield "data: [DONE]\n\n"


@v1_router.post("/chat/completions", response_model=None)
async def handle_chat_completions(request: Request):
    logger.debug("api_key_chat_api.py /chat/completions")
    try:
        body = await request.json()
        chat_request = _request_adapter.validate_python(body)
    except Exception as e:
        return openai_error_response(400, str(e), "invalid_request_error")

    enable_prompt_caching = body.get("prompt_caching")
    if enable_prompt_caching is not None and not isinstance(
        enable_prompt_caching, bool
    ):
        return openai_error_response(
            400, "prompt_caching must be a boolean", "invalid_request_error"
        )

    try:
        model = str(chat_request["model"])
        messages = list(chat_request["messages"])
        stream = bool(chat_request.get("stream"))
        extra_params = {
            key: value
            for key, value in chat_request.items()
            if key not in _PASSTHROUGH_EXCLUDE and value is not None
        }
        tools = list(extra_params.pop("tools", None) or [])

        backend = get_backend(model)

        params = backend.build_params(
            stream=stream,
            tools=tools,
            messages=messages,
            enable_prompt_caching=enable_prompt_caching,
        )
        for key in ("temperature", "top_p", "stop"):
            if key not in extra_params:
                params.pop(key, None)
        params.update(extra_params)
        apply_claude_upstream_sampling_params(backend.config.upstream_model, params)

        response = await backend.converse(messages, stream=stream, params=params)
    except Exception as e:
        logger.exception("API-key chat completion failed")
        return openai_error_response(500, str(e), "internal_error")

    if isinstance(response, dict) and response.get("type") == "error":
        return openai_error_response(
            int(response.get("code", 50000) / 100),
            response.get("content"),
            "internal_error",
        )

    if stream:
        return StreamingResponse(
            _stream_chunks(response), media_type="text/event-stream"
        )
    return JSONResponse(content=response.model_dump())
