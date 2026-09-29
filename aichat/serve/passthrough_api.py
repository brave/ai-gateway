import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai.types.chat import CompletionCreateParams
from pydantic import TypeAdapter
from starlette.requests import Request

from aichat.protocol.open_ai_protocol import MessageUnion
from aichat.serve.backend.litellm import apply_claude_upstream_sampling_params
from aichat.serve.common_api import require_internal_models_api_key
from aichat.serve.services.backend import get_backend
from aichat.serve.services.data_url_gate import remove_unprocessed_data_url_parts
from aichat.serve.services.pdf import process_messages_for_pdf_limits

logger = logging.getLogger(__name__)

v1_router = APIRouter()

_request_adapter = TypeAdapter(CompletionCreateParams)
_message_adapter = TypeAdapter(MessageUnion)
_PASSTHROUGH_EXCLUDE = {"model", "messages", "stream"}


async def _stream_chunks(response: AsyncIterator) -> AsyncIterator[str]:
    """Yield SSE-formatted chunks from the litellm streaming response."""
    async for chunk in response:
        try:
            yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as e:
            logger.warning(f"Failed to serialize chunk: {e}")
    yield "data: [DONE]\n\n"


@v1_router.post("/passthrough", response_model=None)
async def v1_passthrough(request: Request):
    """
    Passthrough endpoint that forwards the request directly to the litellm backend
    and streams the raw OpenAI-compatible chunks back to the caller.
    Supports both streaming and non-streaming responses.
    """
    if auth_err := require_internal_models_api_key(
        request.headers.get("Authorization")
    ):
        return auth_err
    try:
        body = await request.json()
        chat_request = _request_adapter.validate_python(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    enable_prompt_caching = body.get("prompt_caching")
    if enable_prompt_caching is not None and not isinstance(
        enable_prompt_caching, bool
    ):
        raise HTTPException(status_code=400, detail="prompt_caching must be a boolean")

    model = str(chat_request["model"])
    # Validate the RAW body messages once: outer CompletionCreateParams
    # validation exhausts the lazy content iterators, so messages taken
    # from chat_request would re-validate/dump to empty content.
    messages = [
        _message_adapter.validate_python(m).model_dump(exclude_none=True)
        for m in body["messages"]
    ]
    messages = await remove_unprocessed_data_url_parts(messages)
    messages = await process_messages_for_pdf_limits(messages, None)
    stream = bool(chat_request.get("stream"))
    extra_params = {
        key: value
        for key, value in chat_request.items()
        if key not in _PASSTHROUGH_EXCLUDE and value is not None
    }
    tools = list(extra_params.pop("tools", None) or [])

    try:
        backend = get_backend(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

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

    if isinstance(response, dict) and response.get("type") == "error":
        raise HTTPException(
            status_code=int(response.get("code", 50000) / 100),
            detail=response.get("content"),
        )

    if stream:
        return StreamingResponse(
            _stream_chunks(response),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming: return JSON response directly
        return response
