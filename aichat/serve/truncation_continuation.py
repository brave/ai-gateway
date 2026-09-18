"""Detect early stop (finish_reason=stop) and recover via retry or continuation."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from aichat.protocol.open_ai_protocol import Request, Tool
    from aichat.serve.services.models import ModelConfig

logger = logging.getLogger(__name__)

_TERMINAL_TAIL = frozenset(".!?…")
_CHAR_RUN_IGNORE = frozenset(" \n\t\r")
DEFAULT_DEGENERATE_REPETITION_MIN_RUN = 15

TruncationAction = Literal["none", "continue", "retry"]


def longest_char_run(text: str) -> int:
    if not text:
        return 0
    best = 0
    cur = 0
    prev: str | None = None
    for c in text:
        if c in _CHAR_RUN_IGNORE:
            prev = None
            cur = 0
            continue
        if c == prev:
            cur += 1
        else:
            prev = c
            cur = 1
        best = max(best, cur)
    return best


def has_degenerate_repetition(
    text: str, min_run: int = DEFAULT_DEGENERATE_REPETITION_MIN_RUN
) -> bool:
    """True when the same non-whitespace character repeats many times in a row."""
    return longest_char_run(text) >= min_run


def should_continue_truncated_reply(
    *,
    finish_reason: str | None,
    text: str,
    min_chars: int = 40,
) -> bool:
    """True when a stop looks like an accidental early end, not a complete short reply."""
    if finish_reason != "stop":
        return False
    trimmed = text.rstrip()
    if len(trimmed) < min_chars:
        return False
    return trimmed[-1] not in _TERMINAL_TAIL


def resolve_truncation_action(
    *,
    finish_reason: str | None,
    text: str,
    degenerate_min_run: int = DEFAULT_DEGENERATE_REPETITION_MIN_RUN,
) -> TruncationAction:
    if finish_reason != "stop" or not text.strip():
        return "none"
    if has_degenerate_repetition(text, degenerate_min_run):
        return "retry"
    if should_continue_truncated_reply(finish_reason=finish_reason, text=text):
        return "continue"
    return "none"


def truncation_fixup_enabled(model_config: ModelConfig | None) -> bool:
    if model_config is None:
        return False
    return getattr(model_config, "truncation_continuation", False) is True


TRUNCATION_CONTINUATION_USER_TEXT = (
    "Your previous reply was cut off. Continue exactly where you stopped. "
    "Output only the continuation—do not repeat what you already wrote."
)


def build_truncation_continuation_messages(
    messages: list,
    partial_assistant_text: str,
) -> list[dict]:
    new_messages: list[dict] = []
    for message in messages:
        if hasattr(message, "model_dump"):
            new_messages.append(message.model_dump(exclude_none=True))
        else:
            new_messages.append(dict(message))
    new_messages.append({"role": "assistant", "content": partial_assistant_text})
    new_messages.append({"role": "user", "content": TRUNCATION_CONTINUATION_USER_TEXT})
    return new_messages


def _response_has_tool_calls(response: Any) -> bool:
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if not choices:
            return False
        message = choices[0].get("message") or {}
        return bool(message.get("tool_calls"))
    if not getattr(response, "choices", None):
        return False
    message = getattr(response.choices[0], "message", None)
    return bool(getattr(message, "tool_calls", None))


def _completion_content_and_finish_reason(
    response: Any,
) -> tuple[str | None, str | None]:
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if not choices:
            return None, None
        choice = choices[0]
        message = choice.get("message") or {}
        return message.get("content"), choice.get("finish_reason")
    if not getattr(response, "choices", None):
        return None, None
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message else None
    return content, getattr(choice, "finish_reason", None)


def _set_completion_content(response: Any, content: str) -> None:
    if isinstance(response, dict):
        response["choices"][0]["message"]["content"] = content
        return
    response.choices[0].message.content = content


async def augment_continuation_messages(
    request: Request,
    partial_assistant_text: str,
    *,
    model_config: ModelConfig,
    client_tools: list[str | Tool],
) -> list[dict]:
    from aichat.prompts.prompts import prompts
    from aichat.serve.mcp_tool_execution import simplify_messages_for_llm

    new_messages = build_truncation_continuation_messages(
        request.messages, partial_assistant_text
    )
    for prompt in prompts.prompts:
        new_messages = prompt.augment(
            new_messages,
            model_config=model_config,
            tools=client_tools,
        )
    return simplify_messages_for_llm(new_messages)


async def maybe_non_stream_truncation_recovery(
    *,
    response: Any,
    request: Request,
    llm_messages: list[dict],
    tools: list[str | Tool],
    params: dict,
    backend: Any,
    model_config: ModelConfig,
    depth: int = 0,
) -> Any:
    from aichat.llm.metrics import TRUNCATION_RECOVERY_TOTAL
    from aichat.serve.server_settings import server_settings

    if not truncation_fixup_enabled(model_config):
        return response
    if depth >= server_settings.truncation_continuation_max_depth:
        return response
    if isinstance(response, dict) and response.get("type") == "error":
        return response

    if _response_has_tool_calls(response):
        return response

    content, finish_reason = _completion_content_and_finish_reason(response)
    if not content:
        return response

    action = resolve_truncation_action(finish_reason=finish_reason, text=content)
    if action == "none":
        return response

    if action == "retry":
        TRUNCATION_RECOVERY_TOTAL.labels(request.model, "retry").inc()
        retry_response = await backend.converse(llm_messages, False, params)
        return await maybe_non_stream_truncation_recovery(
            response=retry_response,
            request=request,
            llm_messages=llm_messages,
            tools=tools,
            params=params,
            backend=backend,
            model_config=model_config,
            depth=depth + 1,
        )

    TRUNCATION_RECOVERY_TOTAL.labels(request.model, "continue").inc()
    continuation_messages = await augment_continuation_messages(
        request,
        content,
        model_config=model_config,
        client_tools=tools,
    )
    followup = await backend.converse(continuation_messages, False, params)
    if isinstance(followup, dict) and followup.get("type") == "error":
        return response

    extra, _ = _completion_content_and_finish_reason(followup)
    if extra:
        _set_completion_content(response, content + extra)
    return response


async def stream_truncation_continuation(
    *,
    request: Request,
    partial_assistant_text: str,
    client_tools: list[str | Tool],
    mcp_executor: Any,
    backend: Any,
    model_config: ModelConfig,
    prompts_obj: Any,
    requested_model: str | None,
    metrics_state: dict,
    process_streaming_response: Callable[..., AsyncIterator[str]],
) -> AsyncIterator[str]:
    from aichat.protocol.open_ai_protocol import Request as OpenAIRequest
    from aichat.serve.external_service_settings import external_service_settings

    new_messages = await augment_continuation_messages(
        request,
        partial_assistant_text,
        model_config=model_config,
        client_tools=client_tools,
    )
    params = backend.build_params(request.stream, client_tools, new_messages)
    if request.model.startswith("near-"):
        params["api_key"] = external_service_settings.near_api_key
    followup_response = await backend.converse(new_messages, request.stream, params)
    followup_request = OpenAIRequest.model_validate(
        request.model_dump(exclude_none=True) | {"messages": new_messages}
    )
    followup_metrics = {
        **metrics_state,
        "truncation_continuation_depth": metrics_state.get(
            "truncation_continuation_depth", 0
        )
        + 1,
    }
    async for chunk_str in process_streaming_response(
        followup_response,
        followup_request,
        client_tools,
        mcp_executor,
        backend,
        model_config,
        prompts_obj,
        None,
        requested_model=requested_model,
        metrics_state=followup_metrics,
    ):
        yield chunk_str


def plan_stream_truncation_recovery(
    *,
    assistant_text: str,
    tool_calls_in_response: bool,
    last_finish_reason: str | None,
    model_config: ModelConfig | None,
    backend: Any,
    metrics_state: dict,
) -> TruncationAction:
    from aichat.serve.server_settings import server_settings

    if (
        not assistant_text
        or tool_calls_in_response
        or not truncation_fixup_enabled(model_config)
        or not backend
    ):
        return "none"
    depth = metrics_state.get("truncation_continuation_depth", 0)
    if depth >= server_settings.truncation_continuation_max_depth:
        return "none"
    return resolve_truncation_action(
        finish_reason=last_finish_reason,
        text=assistant_text,
    )


def record_stream_truncation_recovery(
    *, model: str, action: TruncationAction, depth: int
) -> None:
    from aichat.llm.metrics import TRUNCATION_RECOVERY_TOTAL

    if action == "retry":
        TRUNCATION_RECOVERY_TOTAL.labels(model, "skipped_degenerate_stream").inc()
        logger.info(
            "Degenerate repetition in stream; continuation skipped (depth=%s)",
            depth,
        )
    elif action == "continue":
        TRUNCATION_RECOVERY_TOTAL.labels(model, "continue").inc()
        logger.info(
            "Truncation continuation for cut-off assistant text (depth=%s)",
            depth,
        )
