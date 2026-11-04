import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import replace
from enum import StrEnum
from functools import partial
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import Request

from aichat.serve.internal_settings import internal_settings
from aichat.serve.open_ai_api import detect_media_content, get_last_user_message_content
from aichat.serve.services import api_key_usage
from aichat.serve.services.backend import get_backend
from aichat.serve.services.chat_request import (
    ChatRequest,
    build_backend_params,
    parse_chat_request,
    parse_messages,
    prepare_messages,
)
from aichat.serve.services.dynamic_leo.signals import run_dynamic_leo
from aichat.serve.services.model_selection import select_model_for_request
from aichat.serve.services.search_stream import load_search_tools, stream_with_search
from aichat.serve.utils import get_real_ip

logger = logging.getLogger(__name__)


class ApiKeyCapability(StrEnum):
    chat = "chat"
    web_search = "web_search"
    deep_research = "deep_research"


DEFAULT_API_KEY_CAPABILITIES: frozenset[str] = frozenset({ApiKeyCapability.chat})


def openai_error_response(
    status_code: int, message: str, error_type: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": error_type}},
    )


def _client_chunk(chunk: Any, model: str, include_usage: bool) -> Any | None:
    if getattr(chunk, "usage", None) is not None and not include_usage:
        if not chunk.choices:
            return None
        chunk.usage = None
    chunk.model = model
    return chunk


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
            try:
                chunk = _client_chunk(chunk, model, include_usage)
                if chunk is None:
                    continue
                yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as e:
                logger.warning(f"Failed to serialize chunk: {e}")
    yield "data: [DONE]\n\n"


def _build_params(backend, chat_request: ChatRequest, messages: list[dict]) -> dict:
    params = build_backend_params(backend, chat_request, messages)
    if chat_request.stream:
        params["stream_options"] = {
            **(params.get("stream_options") or {}),
            "include_usage": True,
        }
    return params


def _usage_context(
    request: Request, api_key: str | None, model: str, backend
) -> api_key_usage.UsageContext | None:
    if api_key is None or not internal_settings.internal_api_enabled:
        return None
    return api_key_usage.new_usage_context(
        request.state.httpx_client, api_key, model, backend.config
    )


def _search_stream(
    request: Request,
    api_key: str | None,
    response,
    usage_ctx,
    *,
    model: str,
    usage_model: str,
    response_model: str,
    backend,
    chat_request: ChatRequest,
    messages: list[dict],
    search_tools: list[dict],
    executor,
    include_usage: bool,
) -> AsyncIterator[str]:
    async def next_round(followup_messages: list[dict]):
        params = _build_params(backend, chat_request, followup_messages)
        followup = await backend.converse(followup_messages, stream=True, params=params)
        if isinstance(followup, dict):
            logger.warning("Search follow-up failed: %s", followup.get("content"))
            return None
        round_ctx = _usage_context(request, api_key, usage_model, backend)
        return api_key_usage.settled_stream(followup, round_ctx)

    return stream_with_search(
        api_key_usage.settled_stream(response, usage_ctx),
        model=model,
        messages=messages,
        inline_search=chat_request.inline_search,
        search_tools=search_tools,
        mcp_executor=executor,
        next_round=next_round,
        on_chunk=partial(
            _client_chunk, model=response_model, include_usage=include_usage
        ),
    )


async def handle_chat_completions(
    request: Request,
    api_key: str | None = None,
    served_model: str | None = None,
    key_capabilities: frozenset[str] = DEFAULT_API_KEY_CAPABILITIES,
):
    logger.debug("api_key_chat /chat/completions")
    try:
        chat_request = parse_chat_request(await request.json())
    except Exception as e:
        return openai_error_response(400, str(e), "invalid_request_error")

    if (
        chat_request.wants_search
        and ApiKeyCapability.web_search not in key_capabilities
    ):
        return openai_error_response(
            403,
            "This API key does not have the web_search capability.",
            "permission_denied",
        )

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
        search_tools, executor = [], None
        if chat_request.search_tools:
            search_tools, executor = await load_search_tools(
                backend.config, chat_request.tools
            )
            chat_request = replace(
                chat_request, tools=chat_request.tools + search_tools
            )

        params = _build_params(backend, chat_request, messages)
        include_usage = bool(
            (chat_request.extra_params.get("stream_options") or {}).get("include_usage")
        )
        usage_ctx = _usage_context(request, api_key, served_model or model, backend)

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

    if chat_request.wants_search:
        return StreamingResponse(
            _search_stream(
                request,
                api_key,
                response,
                usage_ctx,
                model=model,
                usage_model=served_model or model,
                response_model=chat_request.model,
                backend=backend,
                chat_request=chat_request,
                messages=messages,
                search_tools=search_tools,
                executor=executor,
                include_usage=include_usage,
            ),
            media_type="text/event-stream",
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
