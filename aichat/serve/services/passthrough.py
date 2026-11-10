import logging
from collections.abc import AsyncIterator
from dataclasses import replace

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.requests import Request

from aichat.serve.services.backend import get_backend
from aichat.serve.services.chat_request import (
    ChatRequest,
    build_backend_params,
    parse_chat_request,
    parse_messages,
    prepare_messages,
)
from aichat.serve.services.search_stream import (
    load_search_tools,
    relax_followup_tool_choice,
    stream_with_search,
)

logger = logging.getLogger(__name__)


async def _stream_chunks(response: AsyncIterator) -> AsyncIterator[str]:
    """Yield SSE-formatted chunks from the litellm streaming response."""
    async for chunk in response:
        try:
            yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as e:
            logger.warning(f"Failed to serialize chunk: {e}")
    yield "data: [DONE]\n\n"


def _raise_for_error(response) -> None:
    if isinstance(response, dict) and response.get("type") == "error":
        raise HTTPException(
            status_code=int(response.get("code", 50000) / 100),
            detail=response.get("content"),
        )


def _search_stream(
    response, backend, chat_request: ChatRequest, messages, search_tools, executor
) -> AsyncIterator[str]:
    async def next_round(followup_messages: list[dict]):
        params = build_backend_params(backend, chat_request, followup_messages)
        relax_followup_tool_choice(params, chat_request.tools, search_tools)
        followup = await backend.converse(followup_messages, stream=True, params=params)
        if isinstance(followup, dict):
            logger.warning("Search follow-up failed: %s", followup.get("content"))
        return followup

    return stream_with_search(
        response,
        model=chat_request.model,
        messages=messages,
        inline_search=chat_request.inline_search,
        search_tools=search_tools,
        mcp_executor=executor,
        next_round=next_round,
    )


async def handle_passthrough(request: Request):
    try:
        chat_request = parse_chat_request(await request.json())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    messages = await prepare_messages(parse_messages(chat_request))

    try:
        backend = get_backend(chat_request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    search_tools, executor = [], None
    if chat_request.search_tools:
        search_tools, executor = await load_search_tools(
            backend.config, chat_request.tools
        )
        chat_request = replace(chat_request, tools=chat_request.tools + search_tools)

    params = build_backend_params(backend, chat_request, messages)
    response = await backend.converse(
        messages, stream=chat_request.stream, params=params
    )
    _raise_for_error(response)

    if chat_request.search_enabled:
        return StreamingResponse(
            _search_stream(
                response, backend, chat_request, messages, search_tools, executor
            ),
            media_type="text/event-stream",
        )
    if chat_request.stream:
        return StreamingResponse(
            _stream_chunks(response),
            media_type="text/event-stream",
        )
    return response
