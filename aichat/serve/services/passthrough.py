import logging
from collections.abc import AsyncIterator

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.requests import Request

from aichat.serve.services.backend import get_backend
from aichat.serve.services.chat_request import (
    build_backend_params,
    parse_chat_request,
    parse_messages,
    prepare_messages,
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

    params = build_backend_params(backend, chat_request, messages)
    response = await backend.converse(
        messages, stream=chat_request.stream, params=params
    )

    if isinstance(response, dict) and response.get("type") == "error":
        raise HTTPException(
            status_code=int(response.get("code", 50000) / 100),
            detail=response.get("content"),
        )

    if chat_request.stream:
        return StreamingResponse(
            _stream_chunks(response),
            media_type="text/event-stream",
        )
    return response
