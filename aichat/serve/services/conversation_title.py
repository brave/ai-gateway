from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from aichat.llm.metrics import CONVERSATION_TITLE_INVALID_REQUEST_TOTAL
from aichat.prompts.conversation_title import (
    TYPE as BRAVE_CONVERSATION_TITLE_PART_TYPE,
)
from aichat.prompts.conversation_title import (
    conversation_title as conversation_title_prompt,
)
from aichat.protocol.open_ai_protocol import ErrorCode
from aichat.protocol.open_ai_protocol import Request as OpenAIRequest
from aichat.serve.common_api import create_error_response
from aichat.serve.services.backend import get_backend
from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.models import get_model_config

_INVALID_TITLE_REQUEST_MESSAGE = (
    "Conversation title request is missing required text content."
)


def last_message_includes_conversation_title(messages: list[Any]) -> bool:
    """True if the last message is user and includes a brave-conversation-title part."""
    for message in messages:
        if message.role != "user" or not isinstance(message.content, list):
            return False
        for part in message.content:
            ptype = (
                part.get("type")
                if isinstance(part, dict)
                else getattr(part, "type", None)
            )
            if ptype == BRAVE_CONVERSATION_TITLE_PART_TYPE:
                return True
    return False


def _title_part_has_no_text(messages: list[Any]) -> bool:
    """True when a brave-conversation-title part is present but has no usable text."""
    for message in messages:
        if message.role != "user" or not isinstance(message.content, list):
            continue
        for part in message.content:
            ptype = (
                part.get("type")
                if isinstance(part, dict)
                else getattr(part, "type", None)
            )
            if ptype != BRAVE_CONVERSATION_TITLE_PART_TYPE:
                continue
            text = (
                part.get("text")
                if isinstance(part, dict)
                else getattr(part, "text", None)
            )
            return not text
    return False


def _invalid_title_request_response() -> JSONResponse:
    CONVERSATION_TITLE_INVALID_REQUEST_TOTAL.inc()
    return create_error_response(
        ErrorCode.BAD_REQUEST_ERROR,
        _INVALID_TITLE_REQUEST_MESSAGE,
    )


async def complete_conversation_title_chat(
    *,
    request: OpenAIRequest,
    prompts: Any,
    process_streaming_response: Any,
) -> StreamingResponse | JSONResponse:
    """Title path: title-only augment; reuse main streaming stack (no tools / MCP / injection task)."""
    if _title_part_has_no_text(request.messages):
        return _invalid_title_request_response()

    conversation_title_model = model_settings.model_triaging.get("conversation_title")
    model_config = get_model_config(conversation_title_model)
    messages = conversation_title_prompt.augment(
        [m.model_dump(exclude_none=True) for m in request.messages],
        tools=[],
    )

    tools = []
    backend = get_backend(conversation_title_model)
    params = backend.build_params(request.stream, tools, messages)
    response = await backend.converse(messages, request.stream, params)

    if request.stream:
        return StreamingResponse(
            process_streaming_response(
                response,
                request,
                tools,
                None,
                backend,
                model_config,
                prompts,
                None,
                injection_scan_task=None,
                trimmed_tokens=0,
            ),
            media_type="text/event-stream",
        )

    if isinstance(response, dict) and response.get("type") == "error":
        raise HTTPException(
            status_code=int(response.get("code", 50000) / 100),
            detail=response.get("content"),
        )

    return response
