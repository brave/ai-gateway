"""
OpenAI API Adapter

This module provides OpenAI API-specific functionality including:
- Message-to-trace conversion for alignment scanning
- Alignment checking for tool calls
- Tool call buffering for alignment checking
- Emitting chunks with compaction metadata
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, get_args

from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from aichat.llm.metrics import ALIGNMENT_CHECK_TOTAL, PROMPT_INJECTION_SCAN_TOTAL
from aichat.protocol.open_ai_protocol import (
    AssistantMessage,
    AttachmentContentPartUnion,
    Capability,
    CapabilityOptions,
    MessageUnion,
    RequestContentPartUnion,
    TextContentPart,
    ToolMessage,
    UserMemoryContentPart,
    has_capability,
)
from aichat.responses import CompactionMetadata, PromptInjectionScanResult, SecurityScan
from aichat.serve.services.security_settings import security_settings
from aichat.serve.tool_parser import AlignmentCheckData
from aichat.services.alignment_check_service import (
    alignment_scanner,
)
from aichat.services.prompt_injection_service import (
    PromptInjectionResult,
    injection_scanner,
)

logger = logging.getLogger(__name__)

# Content part types that are considered trusted (user-provided text)
TRUSTED_CONTENT_PART_TYPES = (TextContentPart, UserMemoryContentPart)


class ChunkEmitter:
    """Helper to track first chunk emission and inject compaction metadata."""

    def __init__(self, compaction_info: CompactionMetadata | None = None):
        self.first_chunk_emitted = False
        self.compaction_info = compaction_info

    def emit(self, chunk_str: str) -> str:
        """Emit a chunk, injecting compaction metadata into the first chunk."""
        if (
            not self.first_chunk_emitted
            and self.compaction_info
            and chunk_str.startswith("data: ")
        ):
            # Parse the SSE format: "data: {...}\n\n"
            try:
                json_str = chunk_str[6:].rstrip("\n")
                chunk_dict = json.loads(json_str)
                chunk_dict["metadata"] = {
                    "compaction": self.compaction_info.model_dump()
                }
                logger.info(
                    f"Embedded compaction metadata in first chunk: "
                    f"{len(self.compaction_info.compacted_message_indices)} messages compacted"
                )
                chunk_str = f"data: {json.dumps(chunk_dict, separators=(',', ':'))}\n\n"
            except json.JSONDecodeError:
                pass

        self.first_chunk_emitted = True
        return chunk_str


class OpenAIToolCallBufferManager:
    """Manages the buffering of tool call chunks for alignment checking."""

    def __init__(self) -> None:
        self.buffer: list[ChatCompletionChunk] = []
        self.is_buffering: bool = False
        self.tool_info: dict[str, Any] = {}

    def should_start_buffering(
        self, chunk: ChatCompletionChunk, messages: list[MessageUnion]
    ) -> bool:
        """Check if we should start buffering for this chunk."""
        if (
            self.is_buffering
            or not (choices := getattr(chunk, "choices", None))
            or not (choice := choices[0])
            or not (tool_calls := getattr(choice.delta, "tool_calls", None))
        ):
            return False

        if not does_contain_untrusted_content(messages):
            return False

        # Check if first tool call has a function name (indicates start of tool call)
        first_tool_call = tool_calls[0]
        if first_tool_call.function and first_tool_call.function.name:
            tool_name = first_tool_call.function.name
            if (
                tool_name
                in security_settings.allowed_bypass_tools["alignment_check_bypass"]
            ):
                ALIGNMENT_CHECK_TOTAL.labels(
                    allowed=True,
                    tool=tool_name,
                    bypassed=True,
                ).inc()
                return False
            return True

        return False

    def start_buffering(self, chunk: ChatCompletionChunk) -> None:
        """Start buffering and extract tool information from the first chunk."""
        self.is_buffering = True

        # Extract tool name and ID from the first chunk
        choice = chunk.choices[0]
        if choice and choice.delta and choice.delta.tool_calls:
            first_tool_call = choice.delta.tool_calls[0]
            if first_tool_call.function and first_tool_call.function.name:
                self.tool_info["name"] = first_tool_call.function.name
                self.tool_info["id"] = first_tool_call.id

        self.buffer.append(chunk)

    def add_to_buffer(self, chunk: ChatCompletionChunk) -> None:
        """Add a chunk to the buffer."""
        self.buffer.append(chunk)

    def should_continue_buffering(self, chunk: ChatCompletionChunk) -> bool:
        """Check if we should continue buffering this chunk."""
        if not self.is_buffering:
            return False

        if not hasattr(chunk, "choices") or not chunk.choices:
            return False

        choice = chunk.choices[0]
        if choice is None:
            return False

        delta = choice.delta
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            return True

        return choice.finish_reason == "tool_calls"

    def get_buffered_data(self) -> dict[str, Any]:
        """Get the buffered data and reset the buffer."""
        data = {"tool_info": self.tool_info, "buffer": self.buffer}
        self.reset()
        return data

    def reset(self) -> None:
        """Reset the buffer state."""
        self.buffer = []
        self.is_buffering = False
        self.tool_info = {}


class StreamingInjectionScanEmitter:
    """Manages prompt injection scan task lifecycle for streaming responses."""

    def __init__(self, injection_scan_task: asyncio.Task[PromptInjectionResult] | None):
        self._task = injection_scan_task
        self._emitted = False

    def scan_started_event(self) -> SecurityScan | None:
        if self._task:
            return SecurityScan(type="prompt_injection_scan")
        return None

    def try_get_result(self) -> PromptInjectionScanResult | None:
        """Non-blocking: returns result if the task has completed, else None."""
        if not self._task or self._emitted or not self._task.done():
            return None
        self._emitted = True
        try:
            pi_result = self._task.result()
            return PromptInjectionScanResult(
                probability=pi_result.probability,
                reasoning=pi_result.reasoning,
            )
        except Exception:
            logger.error("Prompt injection scan failed")
            return None

    async def await_result(self) -> PromptInjectionScanResult | None:
        """Blocking: awaits the task if not yet emitted."""
        if not self._task or self._emitted:
            return None
        self._emitted = True
        try:
            pi_result = await self._task
            return PromptInjectionScanResult(
                probability=pi_result.probability,
                reasoning=pi_result.reasoning,
            )
        except Exception:
            logger.error("Prompt injection scan failed")
            return None


def does_contain_untrusted_content(messages: list[MessageUnion]) -> bool:
    """
    Check if the message history contains untrusted content.

    Untrusted content includes:
    - Tool call results (unless the tool is in the allowed bypass list)
    - User messages with page text, images, videos, files, etc.

    Args:
        messages: List of OpenAI messages

    Returns:
        True if the messages contain untrusted content, False otherwise
    """
    for message in messages:
        # Check user messages for untrusted content parts
        if (
            message.role == "user"
            and message.content
            and isinstance(message.content, list)
        ):
            for part in message.content:
                if not isinstance(part, TRUSTED_CONTENT_PART_TYPES):
                    return True

        # Check assistant messages with tool calls
        if message.role == "assistant":
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    if (
                        tool_name
                        not in security_settings.allowed_bypass_tools[
                            "tool_result_bypass"
                        ]
                    ):
                        return True
        # TODO: add compaction detection code here once it's implemented on the client
    return False


def _format_content_part_for_trace(
    part: Any, ignore_attachments: bool = False
) -> str | None:
    """
    Format a content part for the conversation trace.

    Args:
        part: A content part object

    Returns:
        Formatted string for the trace, or None if the part should be skipped
    """
    if isinstance(part, TextContentPart):
        return (
            f"USER: {part.text}"
            if hasattr(part, "text") and part.text is not None
            else None
        )
    elif isinstance(part, UserMemoryContentPart) and not ignore_attachments:
        trace_text = part.get_alignment_trace_text()
        return f"USER: {trace_text}" if trace_text else None
    elif (
        isinstance(part, get_args(AttachmentContentPartUnion))
        and not ignore_attachments
    ) or isinstance(part, get_args(RequestContentPartUnion)):
        return f"USER: {part.ALIGNMENT_TRACE_TEXT}"
    return None


def _maybe_truncate_response(response: str | None) -> str:
    """
    Maybe truncate a response.
    """
    if response and len(response) > security_settings.alignment_truncation_char_limit:
        return "..." + response[-security_settings.alignment_truncation_char_limit :]
    return response.strip()


def build_trace_from_openai_messages(
    messages: list[MessageUnion],
    assistant_response: str | None = None,
) -> str:
    """
    Build a conversation trace from OpenAI API messages.

    Args:
        messages: List of OpenAI messages from the request
        assistant_response: Optional assistant response content generated so far

    Returns:
        String representation of the full conversation trace
    """

    trace_parts = []

    for message in messages:
        role = message.role.upper()

        if message.role == "tool":
            trace_parts.append("TOOL: tool output is omitted for security reasons")
            continue

        elif message.role == "user" and message.content:
            if isinstance(message.content, str):
                trace_parts.append(f"{role}: {message.content}")
            elif isinstance(message.content, list):
                for part in message.content:
                    formatted_part = _format_content_part_for_trace(
                        part, ignore_attachments=False
                    )
                    if formatted_part:
                        trace_parts.append(formatted_part)
            continue

        elif message.role == "assistant":
            if message.content and isinstance(message.content, str):
                trace_parts.append(
                    f"{role}: {_maybe_truncate_response(message.content)}"
                )
            elif message.content and isinstance(message.content, list):
                for part in message.content:
                    trace_parts.append(f"{role}: {_maybe_truncate_response(part.text)}")
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    trace_parts.append("SELECTED ACTION:")
                    trace_parts.append(f"ACTION: {tool_call.function.name}")
                    trace_parts.append(f"ACTION INPUT: {tool_call.function.arguments}")
            continue
        else:
            trace_parts.append(f"{role}: {message.content}")

    if assistant_response and assistant_response.strip():
        trace_parts.append(f"ASSISTANT: {_maybe_truncate_response(assistant_response)}")

    return "\n".join(trace_parts)


def extract_user_message_from_openai_messages(messages: list[MessageUnion]) -> str:
    """
    Extract the original user message from OpenAI messages.

    Args:
        messages: List of OpenAI messages

    Returns:
        str: The original user message
    """
    for message in messages:
        if message.role == "user" and message.content:
            if isinstance(message.content, str):
                return message.content
            elif isinstance(message.content, list):
                text_parts = []
                for part in message.content:
                    formatted_part = _format_content_part_for_trace(
                        part, ignore_attachments=True
                    )
                    if formatted_part:
                        text_parts.append(formatted_part)
                return (
                    "\n".join(text_parts) if text_parts else "User message is not found"
                )

    return "User message is not found"


async def perform_alignment_check(
    tool_name: str,
    tool_arguments: str,
    messages: list[MessageUnion],
    assistant_response: str | None = None,
    injection_scan_task: asyncio.Task | None = None,
) -> AlignmentCheckData:
    """
    Perform alignment check for a single tool call.

    Args:
        tool_name: Name of the tool
        tool_arguments: Arguments for the tool call
        messages: List of OpenAI messages from the request
        assistant_response: Optional assistant response content generated so far
        injection_scan_task: Optional async task for prompt injection scan,
            forwarded to the alignment scanner so the result is awaited there

    Returns:
        AlignmentCheckData with allowed status and reasoning
    """

    if alignment_scanner is None:
        logger.error("OpenAI alignment: scanner not available, falling back to allow")
        return AlignmentCheckData(
            allowed=True,
            reasoning="Alignment scanner not available",
        )

    original_user_message = extract_user_message_from_openai_messages(messages)
    full_conversation_trace = build_trace_from_openai_messages(
        messages, assistant_response
    )

    tool_trace = f"ACTION: {tool_name}\nACTION INPUT: {tool_arguments}"
    complete_trace = (
        f"{full_conversation_trace}\n{tool_trace}"
        if full_conversation_trace
        else tool_trace
    )

    result = await alignment_scanner.scan(
        original_user_message,
        complete_trace,
        injection_scan_task=injection_scan_task,
    )

    ALIGNMENT_CHECK_TOTAL.labels(
        allowed=result.allowed,
        tool=tool_name,
        bypassed=False,
    ).inc()

    return AlignmentCheckData(
        allowed=result.allowed,
        reasoning=result.reasoning,
    )


def extract_tool_arguments_from_buffered_chunks(
    buffered_chunks: list[ChatCompletionChunk],
) -> str:
    """
    Extract complete tool arguments from buffered chunks.

    Args:
        buffered_chunks: List of buffered ChatCompletionChunk objects

    Returns:
        Complete tool arguments string
    """
    complete_args = ""
    for chunk in buffered_chunks:
        if not hasattr(chunk, "choices") or not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice is None:
            continue
        delta = choice.delta
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tool_call in delta.tool_calls:
                if tool_call.function and tool_call.function.arguments:
                    complete_args += tool_call.function.arguments
    return complete_args


async def process_non_streaming_alignment_check(
    response: dict,
    messages: list[MessageUnion],
    injection_scan_task: asyncio.Task | None = None,
) -> dict:
    """
    Process alignment check for non-streaming responses with tool calls.

    Args:
        response: The non-streaming response dictionary from the backend
        messages: List of OpenAI messages from the request
        injection_scan_task: Optional async task for prompt injection scan

    Returns:
        The response with alignment check metadata added to tool calls
    """
    # Check if response has tool calls
    if not response.get("choices"):
        return response

    first_choice = response["choices"][0]
    if not first_choice:
        return response

    message = first_choice.get("message", {})
    tool_calls = message.get("tool_calls")

    if not tool_calls:
        return response
    if not does_contain_untrusted_content(messages):
        return response
    # Get assistant content for alignment check context
    assistant_content = message.get("content")

    # Perform alignment check for each tool call
    for tool_call in tool_calls:
        function_info = tool_call.get("function", {})
        tool_name = function_info.get("name", "unknown_tool")
        tool_arguments = function_info.get("arguments", "{}")

        # Check if tool is in the bypass list
        if (
            tool_name
            in security_settings.allowed_bypass_tools["alignment_check_bypass"]
        ):
            ALIGNMENT_CHECK_TOTAL.labels(
                allowed=True,
                tool=tool_name,
                bypassed=True,
            ).inc()
            continue

        alignment_result = await perform_alignment_check(
            tool_name,
            tool_arguments,
            messages,
            assistant_content,
            injection_scan_task=injection_scan_task,
        )
        # Task is consumed after first await; clear so subsequent calls don't re-await
        injection_scan_task = None

        # Add alignment check to the tool call
        tool_call["alignment_check"] = {
            "allowed": alignment_result.allowed,
            "reasoning": alignment_result.reasoning,
        }

    return response


async def handle_alignment_buffering(
    tool_buffer_manager: OpenAIToolCallBufferManager,
    chunk,
    stop_reason: str | None,
    messages: list[MessageUnion],
    mcp_executor,
    backend,
    tool_calls_in_response: list,
    assistant_content: str | None,
    injection_scan_task: asyncio.Task | None = None,
) -> tuple[bool, list[str], str | None]:
    """
    Handle tool call buffering for alignment checking.

    Returns:
        Tuple of (should_continue, chunks_to_yield, started_tool_name):
        - should_continue: True if the main loop should skip to next iteration
        - chunks_to_yield: List of SSE-formatted chunk strings to emit
        - started_tool_name: Tool name if buffering just started, None otherwise
    """
    chunks_to_yield = []

    if tool_buffer_manager.should_start_buffering(chunk, messages):
        tool_buffer_manager.start_buffering(chunk)
        tool_name = tool_buffer_manager.tool_info.get("name")
        return True, [], tool_name

    if tool_buffer_manager.should_continue_buffering(chunk):
        tool_buffer_manager.add_to_buffer(chunk)
        if stop_reason == "tool_calls":
            buffered_data = tool_buffer_manager.get_buffered_data()
            async for chunk_str in process_buffered_tool_call_chunks(
                buffered_data["tool_info"],
                buffered_data["buffer"],
                messages,
                mcp_executor,
                backend,
                tool_calls_in_response,
                assistant_content,
                injection_scan_task=injection_scan_task,
            ):
                chunks_to_yield.append(chunk_str)
        return True, chunks_to_yield, None

    if tool_buffer_manager.is_buffering:
        buffered_data = tool_buffer_manager.get_buffered_data()
        async for chunk_str in process_buffered_tool_call_chunks(
            buffered_data["tool_info"],
            buffered_data["buffer"],
            messages,
            mcp_executor,
            backend,
            tool_calls_in_response,
            assistant_content,
            injection_scan_task=injection_scan_task,
        ):
            chunks_to_yield.append(chunk_str)
        return False, chunks_to_yield, None

    return False, [], None


async def process_buffered_tool_call_chunks(
    tool_info: dict,
    buffered_chunks: list,
    messages: list[MessageUnion],
    mcp_executor=None,
    backend=None,
    tool_calls_in_response: list | None = None,
    assistant_response: str | None = None,
    injection_scan_task: asyncio.Task | None = None,
) -> AsyncIterator[str]:
    """
    Process buffered tool call chunks with alignment checking.

    Performs alignment check on the buffered tool call and adds the alignment
    metadata to the first chunk that contains the tool name.

    Args:
        tool_info: Dict with tool name and id
        buffered_chunks: List of buffered ChatCompletionChunk objects
        messages: List of OpenAI messages from the request
        mcp_executor: Optional MCP executor
        backend: Optional backend
        tool_calls_in_response: Optional list of ToolCall objects to update with alignment result
        assistant_response: Optional assistant response content generated so far
        injection_scan_task: Optional async task for prompt injection scan

    Yields:
        SSE-formatted strings for each buffered chunk
    """
    if not buffered_chunks:
        return

    tool_name = tool_info.get("name", "unknown_tool")
    tool_id = tool_info.get("id")

    tool_arguments = extract_tool_arguments_from_buffered_chunks(buffered_chunks)
    alignment_result = await perform_alignment_check(
        tool_name,
        tool_arguments,
        messages,
        assistant_response,
        injection_scan_task=injection_scan_task,
    )

    # Update the matching ToolCall object in tool_calls_in_response if provided
    if tool_calls_in_response:
        for tc in tool_calls_in_response:
            if tc.function_name == tool_name and (tool_id is None or tc.id == tool_id):
                tc.alignment_check = alignment_result
                break

    first_chunk_with_name_processed = False
    for chunk in buffered_chunks:
        chunk_dict = chunk.model_dump(exclude_none=True)

        if "model" in chunk_dict:
            chunk_dict["model"] = chunk_dict["model"].replace(" ", "")

        # Add alignment metadata to the first chunk that has the tool name
        if not first_chunk_with_name_processed:
            choices = chunk_dict.get("choices", [])
            if choices and choices[0]:
                delta = choices[0].get("delta", {})
                tool_calls = delta.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        if tc.get("function", {}).get("name"):
                            tc["alignment_check"] = {
                                "allowed": alignment_result.allowed,
                                "reasoning": alignment_result.reasoning,
                            }
                            first_chunk_with_name_processed = True
                            break

        should_skip = False
        if (
            mcp_executor
            and backend
            and chunk_dict.get("choices", [{}])[0].get("finish_reason") == "tool_calls"
        ):
            logger.info(
                "Skipping finish_reason chunk for server-side tool execution (buffered)"
            )
            should_skip = True

        if not should_skip:
            chunk_json = json.dumps(chunk_dict, separators=(",", ":"))
            yield f"data: {chunk_json}\n\n"


def has_recent_untrusted_tool_results(messages: list[MessageUnion]) -> bool:
    """Check if the most recent messages are tool results requiring injection scanning.

    Walks backward from the end to collect all consecutive ToolMessage entries,
    finds the preceding AssistantMessage with tool_calls, and checks whether all
    the called tools are in the tool_result_bypass list. Returns False (skip scan)
    if every tool is bypassed, True otherwise.
    """
    if not messages:
        return False

    tool_messages: list[ToolMessage] = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            tool_messages.append(message)
        else:
            break

    if not tool_messages:
        return False

    assistant_idx = len(messages) - len(tool_messages) - 1
    if assistant_idx < 0:
        return True

    assistant_msg = messages[assistant_idx]
    if not isinstance(assistant_msg, AssistantMessage) or not assistant_msg.tool_calls:
        return True

    tool_call_names = {
        tc.id: tc.function.name
        for tc in assistant_msg.tool_calls
        if tc.id and tc.function.name
    }

    bypass_tools = security_settings.allowed_bypass_tools.get("tool_result_bypass", [])

    for tool_msg in tool_messages:
        tool_name = tool_call_names.get(tool_msg.tool_call_id)
        if tool_name not in bypass_tools:
            return True

    return False


def extract_latest_tool_call_results(
    messages: list[MessageUnion],
) -> str | None:
    """
    Extract content from the most recent consecutive block of tool call result
    messages for prompt injection scanning. Only tool results are considered
    untrusted; normal user messages are not scanned.

    Args:
        messages: List of OpenAI messages

    Returns:
        Combined tool result text truncated to the configured limit, or None
        if no tool results are found
    """
    tool_parts = []

    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.content:
            if isinstance(message.content, str):
                tool_parts.append(message.content)
            elif isinstance(message.content, list):
                for part in message.content:
                    if isinstance(part, TextContentPart):
                        tool_parts.append(part.text)
        else:
            break

    tool_parts.reverse()

    if not tool_parts:
        return None

    return "\n---\n".join(tool_parts)


def maybe_perform_prompt_injection_scan(
    messages: list[MessageUnion],
    brave_capability: CapabilityOptions,
) -> asyncio.Task[PromptInjectionResult] | None:
    """
    Launch a prompt injection scan as a background task if conditions are met.

    Returns an asyncio.Task if a scan was started, or None if skipped.
    Skips when scanning is disabled, capability is not content_agent,
    no recent untrusted tool results are present, or the scanner is unavailable.
    """
    if not security_settings.prompt_injection_scanning_enabled:
        return None

    if not has_capability(brave_capability, Capability.content_agent):
        logger.debug(
            f"Prompt injection scan skipped: capability is '{brave_capability}', "
            "not 'content_agent'"
        )
        return None

    if not has_recent_untrusted_tool_results(messages):
        logger.info(
            "Prompt injection scan skipped: no recent untrusted tool results in messages"
        )
        return None

    logger.info("Launching prompt injection scan task in parallel with LLM completion")
    return asyncio.create_task(_perform_prompt_injection_scan(messages))


async def _perform_prompt_injection_scan(
    messages: list[MessageUnion],
) -> PromptInjectionResult:
    """Run the actual prompt injection scan on the most recent tool call results."""
    if injection_scanner is None:
        logger.error("Prompt injection scanner not available, falling back to benign")
        return PromptInjectionResult(
            probability=1,
            reasoning="Prompt injection scanner not available",
        )

    tool_content = extract_latest_tool_call_results(messages)

    if not tool_content:
        logger.info("Prompt injection scan skipped: no tool call results found")
        return PromptInjectionResult(
            probability=1,
            reasoning="No tool call results found to scan",
        )

    tool_messages_count = sum(1 for m in messages if m.role == "tool")
    logger.info(
        f"Starting prompt injection scan: {tool_messages_count} tool message(s), "
        f"{len(tool_content)} chars of content to scan"
    )

    result = await injection_scanner.scan(tool_content)

    PROMPT_INJECTION_SCAN_TOTAL.labels(
        probability=result.probability,
    ).inc()

    logger.info(
        f"Prompt injection scan result: probability={result.probability}/5, "
        f"duration={result.duration_ms:.0f}ms"
    )

    return result


async def process_non_streaming_prompt_injection_scan(
    response,
    injection_scan_task: asyncio.Task[PromptInjectionResult] | None,
):
    """
    Await the prompt injection scan task and add its result to a non-streaming response.

    No-op if injection_scan_task is None. Returns the response unchanged on failure.
    """
    if not injection_scan_task:
        return response

    try:
        injection_result = await injection_scan_task
    except Exception:
        logger.error("Prompt injection scan failed")
        return response

    if not isinstance(response, dict):
        response = response.model_dump()
    response["prompt_injection_scan"] = {
        "probability": injection_result.probability,
        "reasoning": injection_result.reasoning,
    }
    return response
