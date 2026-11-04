import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from aichat.protocol.open_ai_protocol import ToolCall, ToolCallFunction
from aichat.serve.mcp_integration import initialize_mcp_for_request
from aichat.serve.mcp_tool_execution import (
    build_tool_followup_messages,
    run_mcp_tool_round,
)
from aichat.serve.services.mcp.mcp_settings import mcp_settings
from aichat.serve.services.models import ModelConfig
from aichat.serve.services.search import InlineSearchStream
from aichat.serve.tool_parser import ToolCallAccumulator, parse_streaming_tool_calls

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 5
SEARCH_TOOL_NAMES = frozenset({"brave_web_search"})

NextRound = Callable[[list[dict]], Awaitable[AsyncIterator | None]]
OnChunk = Callable[[Any], Any | None]


def _tool_name(tool: Any) -> str | None:
    function = tool.get("function") if isinstance(tool, dict) else None
    return function.get("name") if isinstance(function, dict) else None


async def load_search_tools(
    model_config: ModelConfig, client_tools: list
) -> tuple[list[dict], Any]:
    if not mcp_settings.mcp_enabled or not model_config.tool_support:
        return [], None
    try:
        mcp_tools, mcp_executor = await initialize_mcp_for_request(
            model_config=model_config, dynamic_leo_enabled=False
        )
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError):
        logger.exception("Failed to load search tools")
        return [], None

    client_tool_names = {_tool_name(tool) for tool in client_tools}
    search_tools = [
        tool.model_dump(exclude_none=True)
        for tool in mcp_tools
        if tool.function.name in SEARCH_TOOL_NAMES
        and tool.function.name not in client_tool_names
    ]
    return search_tools, mcp_executor


@dataclass
class _Round:
    accumulator: ToolCallAccumulator = field(default_factory=ToolCallAccumulator)
    tool_calls: list = field(default_factory=list)
    hidden_indexes: set[int] = field(default_factory=set)
    response_id: str | None = None


def _hide_search_tool_calls(chunk: Any, state: _Round, search_names: set[str]) -> bool:
    """Strip search tool call deltas from the chunk. Returns False to drop it."""
    choice = chunk.choices[0]
    delta = choice.delta
    tool_calls = getattr(delta, "tool_calls", None)
    stripped = False
    if tool_calls:
        for tool_call in tool_calls:
            function = tool_call.function
            if function is not None and function.name in search_names:
                state.hidden_indexes.add(tool_call.index or 0)
        visible = [
            tool_call
            for tool_call in tool_calls
            if (tool_call.index or 0) not in state.hidden_indexes
        ]
        stripped = len(visible) != len(tool_calls)
        delta.tool_calls = visible or None
    if (
        choice.finish_reason == "tool_calls"
        and state.hidden_indexes
        and set(state.accumulator.tool_calls) <= state.hidden_indexes
    ):
        choice.finish_reason = None
        stripped = True
    return not (
        stripped
        and not getattr(delta, "content", None)
        and not getattr(delta, "tool_calls", None)
        and not choice.finish_reason
    )


async def _stream_round(
    response: AsyncIterator,
    state: _Round,
    search_names: set[str],
    inline_search: InlineSearchStream | None,
    on_chunk: OnChunk | None,
) -> AsyncIterator[str]:
    content = ""
    try:
        async for chunk in response:
            if inline_search is not None:
                for event in inline_search.pop_events():
                    if event:
                        yield event
            if state.response_id is None:
                state.response_id = getattr(chunk, "id", None)

            if chunk.choices:
                choice = chunk.choices[0]
                delta_content = getattr(choice.delta, "content", None)
                if inline_search is not None and delta_content:
                    content += delta_content
                    inline_search.feed(content)
                delta_tool_calls = getattr(choice.delta, "tool_calls", None)
                if delta_tool_calls or choice.finish_reason == "tool_calls":
                    state.tool_calls.extend(
                        parse_streaming_tool_calls(
                            state.accumulator,
                            delta_tool_calls,
                            choice.finish_reason,
                        )
                    )
                if search_names and not _hide_search_tool_calls(
                    chunk, state, search_names
                ):
                    continue

            try:
                if on_chunk is not None:
                    chunk = on_chunk(chunk)
                    if chunk is None:
                        continue
                yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as e:
                logger.warning(f"Failed to serialize chunk: {e}")
    finally:
        aclose = getattr(response, "aclose", None)
        if aclose is not None:
            await aclose()


def _assistant_tool_calls(tool_calls: list) -> list[ToolCall]:
    return [
        ToolCall(
            id=tool_call.id or f"call_{tool_call.index}",
            type="function",
            function=ToolCallFunction(
                name=tool_call.function_name,
                arguments=tool_call.arguments or "{}",
            ),
        )
        for tool_call in tool_calls
    ]


async def stream_with_search(
    response: AsyncIterator,
    *,
    model: str,
    messages: list[dict],
    inline_search: bool,
    search_tools: list[dict],
    mcp_executor: Any,
    next_round: NextRound,
    on_chunk: OnChunk | None = None,
) -> AsyncIterator[str]:
    search_names = {_tool_name(tool) for tool in search_tools} - {None}
    inline = InlineSearchStream() if inline_search else None
    try:
        for round_number in range(_MAX_TOOL_ROUNDS + 1):
            state = _Round()
            async for event in _stream_round(
                response, state, search_names, inline, on_chunk
            ):
                yield event

            search_calls = [
                tool_call
                for tool_call in state.tool_calls
                if tool_call.function_name in search_names
            ]
            if not search_calls or len(search_calls) != len(state.tool_calls):
                break
            if round_number == _MAX_TOOL_ROUNDS:
                logger.warning("Search tool round limit reached for %s", model)
                break

            tool_result_messages: list = []
            async for event in run_mcp_tool_round(
                search_calls,
                mcp_executor,
                state.response_id or "unknown",
                model,
                model,
                tool_result_messages,
            ):
                yield event
            if not tool_result_messages:
                break

            messages = build_tool_followup_messages(
                messages,
                _assistant_tool_calls(search_calls),
                search_calls,
                tool_result_messages,
            )
            response = await next_round(messages)
            if response is None:
                break

        if inline is not None:
            for event in await inline.drain_events():
                if event:
                    yield event
        yield "data: [DONE]\n\n"
    finally:
        if inline is not None:
            await inline.aclose()
