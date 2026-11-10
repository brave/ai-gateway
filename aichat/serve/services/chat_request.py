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
    inline_search: bool = False
    search_tools: bool = False

    @property
    def search_enabled(self) -> bool:
        return self.inline_search or self.search_tools


def _optional_bool(body: Any, key: str) -> bool | None:
    value = body.get(key)
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def parse_chat_request(body: Any) -> ChatRequest:
    chat_request = _request_adapter.validate_python(body)

    enable_prompt_caching = _optional_bool(body, "prompt_caching")
    inline_search = bool(_optional_bool(body, "brave_inline_search"))
    search_tools = bool(_optional_bool(body, "brave_search_tools"))
    stream = bool(chat_request.get("stream"))
    if (inline_search or search_tools) and not stream:
        raise ValueError(
            "brave_inline_search and brave_search_tools require stream to be true"
        )
    if (inline_search or search_tools) and chat_request.get("n") not in (None, 1):
        raise ValueError("brave_inline_search and brave_search_tools require n to be 1")

    extra_params = {
        key: value
        for key, value in chat_request.items()
        if key not in _PASSTHROUGH_EXCLUDE and value is not None
    }
    tools = list(extra_params.pop("tools", None) or [])
    return ChatRequest(
        model=str(chat_request["model"]),
        raw_messages=body["messages"],
        stream=stream,
        tools=tools,
        extra_params=extra_params,
        enable_prompt_caching=enable_prompt_caching,
        inline_search=inline_search,
        search_tools=search_tools,
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
