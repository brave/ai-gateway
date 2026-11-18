import logging
from dataclasses import dataclass
from typing import Any

from openai.types.chat import CompletionCreateParams
from pydantic import TypeAdapter

from aichat.protocol.open_ai_protocol import MessageUnion
from aichat.serve.backend.litellm import apply_claude_upstream_sampling_params
from aichat.serve.services.bedrock import MAX_CACHE_POINTS
from aichat.serve.services.data_url_gate import remove_unprocessed_data_url_parts
from aichat.serve.services.pdf import process_messages_for_pdf_limits

logger = logging.getLogger(__name__)

_request_adapter = TypeAdapter(CompletionCreateParams)
_message_adapter = TypeAdapter(MessageUnion)
_PASSTHROUGH_EXCLUDE = {"model", "messages", "stream"}
_PRUNED_DEFAULT_PARAMS = ("temperature", "top_p", "stop")
_EPHEMERAL = {"type": "ephemeral"}


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
    cached_message_indices: frozenset[int] = frozenset()
    cached_tool_indices: frozenset[int] = frozenset()

    @property
    def search_enabled(self) -> bool:
        return self.inline_search or self.search_tools

    @property
    def has_cache_marks(self) -> bool:
        return bool(self.cached_message_indices or self.cached_tool_indices)


def _optional_bool(body: Any, key: str) -> bool | None:
    value = body.get(key)
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _is_cache_mark(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") == "ephemeral"


def _message_has_cache_mark(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    if _is_cache_mark(message.get("cache_control")):
        return True
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(part, dict) and _is_cache_mark(part.get("cache_control"))
        for part in content
    )


def _tool_has_cache_mark(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    function = tool.get("function")
    return _is_cache_mark(tool.get("cache_control")) or (
        isinstance(function, dict) and _is_cache_mark(function.get("cache_control"))
    )


def _marked_indices(items: Any, has_mark) -> frozenset[int]:
    if not isinstance(items, list):
        return frozenset()
    return frozenset(i for i, item in enumerate(items) if has_mark(item))


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
        cached_message_indices=_marked_indices(
            body["messages"], _message_has_cache_mark
        ),
        cached_tool_indices=_marked_indices(body.get("tools"), _tool_has_cache_mark),
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


def _mark_last_part(message: dict) -> None:
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return
        content = [{"type": "text", "text": content}]
        message["content"] = content
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1]["cache_control"] = dict(_EPHEMERAL)


def _apply_client_cache_marks(
    chat_request: ChatRequest, messages: list[dict]
) -> tuple[list, bool]:
    marked = False
    tools = list(chat_request.tools)
    tool_indices = sorted(chat_request.cached_tool_indices)[:MAX_CACHE_POINTS]
    for index in tool_indices:
        if index < len(tools) and isinstance(tools[index], dict):
            tools[index] = {**tools[index], "cache_control": dict(_EPHEMERAL)}
            marked = True
    message_budget = MAX_CACHE_POINTS - len(tool_indices)
    message_indices = sorted(chat_request.cached_message_indices)
    if message_indices and message_budget > 0:
        if len(messages) == len(chat_request.raw_messages):
            for index in message_indices[-message_budget:]:
                _mark_last_part(messages[index])
            marked = True
        else:
            logger.warning("Message count changed; client cache_control ignored")
    return tools, marked


def build_backend_params(
    backend, chat_request: ChatRequest, messages: list[dict]
) -> dict[str, Any]:
    tools = chat_request.tools
    cache_kwargs = {}
    if chat_request.has_cache_marks and backend.uses_prompt_caching(
        chat_request.enable_prompt_caching
    ):
        tools, marked = _apply_client_cache_marks(chat_request, messages)
        cache_kwargs["client_cache_marks"] = marked
    params = backend.build_params(
        stream=chat_request.stream,
        tools=tools,
        messages=messages,
        enable_prompt_caching=chat_request.enable_prompt_caching,
        **cache_kwargs,
    )
    for key in _PRUNED_DEFAULT_PARAMS:
        if key not in chat_request.extra_params:
            params.pop(key, None)
    params.update(chat_request.extra_params)
    apply_claude_upstream_sampling_params(backend.config.upstream_model, params)
    return params
