from dataclasses import dataclass
from typing import Any

from openai.types.chat import CompletionCreateParams
from pydantic import TypeAdapter

from aichat.protocol.open_ai_protocol import MessageUnion
from aichat.serve.backend.litellm import apply_claude_upstream_sampling_params
from aichat.serve.services.data_url_gate import remove_unprocessed_data_url_parts
from aichat.serve.services.pdf import process_messages_for_pdf_limits

_request_adapter = TypeAdapter(CompletionCreateParams)
_message_adapter = TypeAdapter(MessageUnion)
_PASSTHROUGH_EXCLUDE = {"model", "messages", "stream"}
_PRUNED_DEFAULT_PARAMS = ("temperature", "top_p", "stop")


@dataclass(frozen=True)
class ChatRequest:
    model: str
    raw_messages: list[dict]
    stream: bool
    tools: list
    extra_params: dict[str, Any]
    enable_prompt_caching: bool | None


def parse_chat_request(body: Any) -> ChatRequest:
    chat_request = _request_adapter.validate_python(body)

    enable_prompt_caching = body.get("prompt_caching")
    if enable_prompt_caching is not None and not isinstance(
        enable_prompt_caching, bool
    ):
        raise ValueError("prompt_caching must be a boolean")

    extra_params = {
        key: value
        for key, value in chat_request.items()
        if key not in _PASSTHROUGH_EXCLUDE and value is not None
    }
    tools = list(extra_params.pop("tools", None) or [])
    return ChatRequest(
        model=str(chat_request["model"]),
        raw_messages=body["messages"],
        stream=bool(chat_request.get("stream")),
        tools=tools,
        extra_params=extra_params,
        enable_prompt_caching=enable_prompt_caching,
    )


def parse_messages(chat_request: ChatRequest) -> list[MessageUnion]:
    # Validate the RAW body messages once: outer CompletionCreateParams
    # validation exhausts the lazy content iterators, so messages taken
    # from the validated request would re-validate/dump to empty content.
    # Raw dicts validate cleanly and model_dump strips extra keys
    # (file_id/format).
    return [_message_adapter.validate_python(m) for m in chat_request.raw_messages]


async def prepare_messages(protocol_messages: list[MessageUnion]) -> list[dict]:
    messages = [m.model_dump(exclude_none=True) for m in protocol_messages]
    messages = await remove_unprocessed_data_url_parts(messages)
    return await process_messages_for_pdf_limits(messages, None)


def build_backend_params(
    backend, chat_request: ChatRequest, messages: list[dict]
) -> dict[str, Any]:
    params = backend.build_params(
        stream=chat_request.stream,
        tools=chat_request.tools,
        messages=messages,
        enable_prompt_caching=chat_request.enable_prompt_caching,
    )
    for key in _PRUNED_DEFAULT_PARAMS:
        if key not in chat_request.extra_params:
            params.pop(key, None)
    params.update(chat_request.extra_params)
    apply_claude_upstream_sampling_params(backend.config.upstream_model, params)
    return params
