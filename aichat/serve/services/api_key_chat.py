import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing

from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import Request

from aichat.serve.internal_settings import internal_settings
from aichat.serve.open_ai_api import detect_media_content, get_last_user_message_content
from aichat.serve.services import api_key_usage
from aichat.serve.services.backend import get_backend
from aichat.serve.services.chat_request import (
    build_backend_params,
    parse_chat_request,
    parse_messages,
    prepare_messages,
)
from aichat.serve.services.dynamic_leo.signals import run_dynamic_leo
from aichat.serve.services.model_selection import select_model_for_request
from aichat.serve.utils import get_real_ip

logger = logging.getLogger(__name__)


def openai_error_response(
    status_code: int, message: str, error_type: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": error_type}},
    )


# Streams the completion chunks of the response to the client.
# Usage is settled in the background after the response is delivered.
async def _stream_completion_chunks(
    response: AsyncIterator,
    model: str,
    usage_ctx: api_key_usage.UsageContext | None = None,
    include_usage: bool = True,
) -> AsyncIterator[str]:
    async with aclosing(api_key_usage.settled_stream(response, usage_ctx)) as chunks:
        async for chunk in chunks:
            if getattr(chunk, "usage", None) is not None and not include_usage:
                if not chunk.choices:
                    continue
                chunk.usage = None
            try:
                chunk.model = model
                yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as e:
                logger.warning(f"Failed to serialize chunk: {e}")
    yield "data: [DONE]\n\n"


async def handle_chat_completions(
    request: Request, api_key: str | None = None, served_model: str | None = None
):
    logger.debug("api_key_chat /chat/completions")
    try:
        chat_request = parse_chat_request(await request.json())
    except Exception as e:
        return openai_error_response(400, str(e), "invalid_request_error")

    try:
        protocol_messages = parse_messages(chat_request)
        messages = await prepare_messages(protocol_messages)
        dynamic_leo_prefetch = await run_dynamic_leo(protocol_messages)
        model = await select_model_for_request(
            model=served_model or chat_request.model,
            messages=protocol_messages,
            is_premium=False,
            last_user_message_content=get_last_user_message_content(protocol_messages),
            media_type=detect_media_content(protocol_messages),
            rate_key=get_real_ip(request.headers.get("x-forwarded-for")),
            httpx_client=getattr(request.state, "httpx_client", None),
            androcles_prefetch=dynamic_leo_prefetch,
        )

        backend = get_backend(model)
        params = build_backend_params(backend, chat_request, messages)
        include_usage = bool(
            (chat_request.extra_params.get("stream_options") or {}).get("include_usage")
        )
        if chat_request.stream:
            params["stream_options"] = {
                **(params.get("stream_options") or {}),
                "include_usage": True,
            }

        usage_ctx = None
        if api_key is not None and internal_settings.internal_api_enabled:
            usage_ctx = api_key_usage.new_usage_context(
                request.state.httpx_client,
                api_key,
                served_model or model,
                backend.config,
            )

        converse = backend.converse(messages, stream=chat_request.stream, params=params)
        if chat_request.stream:
            response = await converse
        else:
            response = await asyncio.shield(
                api_key_usage.spawn(
                    api_key_usage.converse_and_settle(converse, usage_ctx)
                )
            )
    except Exception:
        logger.exception("API-key chat completion failed")
        return openai_error_response(500, "Internal server error.", "internal_error")

    if isinstance(response, dict) and response.get("type") == "error":
        return openai_error_response(
            int(response.get("code", 50000) / 100),
            response.get("content"),
            "internal_error",
        )

    if chat_request.stream:
        return StreamingResponse(
            _stream_completion_chunks(
                response, chat_request.model, usage_ctx, include_usage
            ),
            media_type="text/event-stream",
        )
    response.model = chat_request.model
    return JSONResponse(content=response.model_dump())
