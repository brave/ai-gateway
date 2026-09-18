import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
)

from aichat.llm.metrics import FUNCTION_CALL_COUNTER
from aichat.protocol.open_ai_protocol import (
    AssistantMessage,
    ToolCallFunction,
    ToolMessage,
)
from aichat.protocol.open_ai_protocol import (
    ToolCall as OpenAIToolCall,
)
from aichat.responses import ToolEnd, ToolError, ToolStart
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.services.mcp.handlers.deep_research import (
    execute_deep_research_streaming,
)
from aichat.serve.services.mcp.registry import get_global_registry
from aichat.serve.tool_parser import ToolExecutor


# Lazy import to avoid circular dependency with open_ai_api
def _get_create_openai_response():
    from aichat.serve.open_ai_api import create_openai_response

    return create_openai_response


logger = logging.getLogger(__name__)

# Streaming tools that require special handling
STREAMING_TOOLS = {"deep_research"}


async def execute_tools_and_stream_events(
    tool_calls: list,
    mcp_executor,
    conversation_log_id: str,
    actual_model: str,
    request_model: str,
):
    """
    Execute MCP tools and yield brave event chunks.

    Args:
        tool_calls: List of tool calls to execute
        mcp_executor: MCP executor instance
        conversation_log_id: Conversation log ID for event tracking
        actual_model: The actual model name from response
        request_model: The requested model name

    Yields:
        Tuples of (event_string, output_or_message) where:
        - event_string: SSE formatted event string for top-level events
          (toolStart, contentReceipt) or None
        - output_or_message: Either:
          - dict: Tool output chunk with output (for streaming)
          - ToolMessage: Tool result message (for conversation history)
          - None: No output/message
    """
    tool_executor = ToolExecutor(mcp_executor)
    registry = get_global_registry()

    if mcp_executor._tool_cache is None:
        mcp_client = await mcp_executor._get_mcp_client()
        mcp_executor._tool_cache = await mcp_executor._fetch_all_tools(mcp_client)

    model_name = (actual_model or request_model).replace(" ", "")

    for tool_call in tool_calls:
        try:
            logger.debug(f"Executing tool: {tool_call.function_name}")
            FUNCTION_CALL_COUNTER.labels(function_name=tool_call.function_name).inc()

            start_event = await _create_tool_start_event(
                tool_call,
                mcp_executor,
                registry,
                conversation_log_id,
                model_name,
            )
            if start_event:
                yield (start_event, None)

            result = await tool_executor.execute_tool_call(tool_call, model=model_name)
            logger.debug(f"Tool result type: {type(result)}, content: {result}")

            if isinstance(result, OpenAIToolCall):
                logger.debug(
                    f"Tool {tool_call.function_name} is client-side, skipping "
                    "server-side execution"
                )
                continue

            if result is None:
                error_msg = (
                    f"Tool {tool_call.function_name} failed to execute. "
                    "Please try again with different parameters."
                )
                logger.warning(
                    f"Tool {tool_call.function_name} returned None, "
                    "creating error event"
                )

                error_content = f"Error: {error_msg}"
                tool_message = ToolMessage(
                    role="tool",
                    tool_call_id=tool_call.id or f"call_{tool_call.index}",
                    content=error_content,
                )
                error_event = _create_tool_error_event(
                    tool_call.id or f"call_{tool_call.index}",
                    tool_call.function_name,
                    error_msg,
                    conversation_log_id,
                    model_name,
                )

                # Create and yield error chunk for streaming
                error_chunk = _create_tool_error_chunk(
                    tool_call,
                    error_content,
                    conversation_log_id,
                    model_name,
                )
                if error_chunk:
                    yield (None, error_chunk)

                if error_event:
                    yield (error_event, tool_message)

                continue

            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                if (
                    isinstance(first_result, dict)
                    and first_result.get("type") == "error"
                ):
                    error_content = first_result.get("content", "Tool execution failed")
                    error_msg = first_result.get("error", error_content)
                    logger.warning(
                        f"Tool {tool_call.function_name} returned error: "
                        f"{error_content}"
                    )
                    tool_message = ToolMessage(
                        role="tool",
                        tool_call_id=tool_call.id or f"call_{tool_call.index}",
                        content=error_content,
                    )
                    error_event = _create_tool_error_event(
                        tool_call.id or f"call_{tool_call.index}",
                        tool_call.function_name,
                        error_msg,
                        conversation_log_id,
                        model_name,
                    )

                    # Create and yield error chunk for streaming
                    error_chunk = _create_tool_error_chunk(
                        tool_call,
                        error_content,
                        conversation_log_id,
                        model_name,
                    )
                    if error_chunk:
                        yield (None, error_chunk)

                    if error_event:
                        yield (error_event, tool_message)

                    continue

            actual_result = (
                result[0] if isinstance(result, list) and len(result) > 0 else result
            )

            handler = None
            if isinstance(actual_result, dict) and "server_name" in actual_result:
                handler = registry.handlers.get(actual_result["server_name"])

            if handler:
                tool_message_content = handler.get_tool_message_content(
                    actual_result, tool_call
                )
                output_content_parts = handler.get_output_content_parts(
                    actual_result, tool_call
                )
            else:
                tool_content = _process_tool_result(actual_result)
                tool_message_content = (
                    tool_content.strip() or "Tool execution completed"
                )
                output_content_parts = []

            if output_content_parts:
                output_chunk = _create_tool_output_chunk_from_parts(
                    tool_call,
                    output_content_parts,
                    conversation_log_id,
                    model_name,
                )
                if output_chunk:
                    yield (None, output_chunk)

            logger.debug(
                f"Tool result content being sent to model (type: {type(tool_message_content)})"
            )

            tool_message = ToolMessage(
                role="tool",
                tool_call_id=tool_call.id or f"call_{tool_call.index}",
                content=tool_message_content,
            )

            yield (None, tool_message)

        except Exception as e:
            logger.exception("Tool execution failed")

            error_content = f"Tool execution failed: {e!s}"
            error_message = ToolMessage(
                role="tool",
                tool_call_id=tool_call.id or f"call_{tool_call.index}",
                content=error_content,
            )

            error_event = _create_tool_error_event(
                tool_call.id or f"call_{tool_call.index}",
                tool_call.function_name,
                str(e),
                conversation_log_id,
                model_name,
            )

            # Create and yield error chunk for streaming
            error_chunk = _create_tool_error_chunk(
                tool_call,
                error_content,
                conversation_log_id,
                model_name,
            )
            if error_chunk:
                yield (None, error_chunk)

            if error_event:
                yield (error_event, error_message)


async def _create_tool_start_event(
    tool_call,
    mcp_executor,
    registry,
    conversation_log_id: str,
    model_name: str,
) -> str | None:
    """Create brave-chat.toolStart event."""
    return _get_create_openai_response()(
        ToolStart(
            tool_call_id=tool_call.id or f"call_{tool_call.index}",
            tool_name=tool_call.function_name,
        )
    )


def _process_tool_result(result) -> str:
    """Process generic tool execution result and extract content."""
    tool_content = ""

    if not result:
        return tool_content

    results_to_process = result if isinstance(result, list) else [result]

    for single_result in results_to_process:
        if isinstance(single_result, dict):
            if "content" in single_result:
                tool_content += str(single_result["content"]) + "\n"
            else:
                tool_content += json.dumps(single_result) + "\n"
        else:
            tool_content += str(single_result) + "\n"

    return tool_content


def _create_tool_output_chunk_from_parts(
    tool_call,
    output_content_parts: list[dict],
    conversation_log_id: str,
    model_name: str,
) -> dict | None:
    """Create a streaming chunk with tool output in tool_calls delta."""
    if not output_content_parts:
        return None

    try:
        tool_call_id = tool_call.id or f"call_{tool_call.index}"

        tool_call_delta = ChoiceDeltaToolCall(
            id=tool_call_id,
            index=tool_call.index or 0,
        )

        chunk = ChatCompletionChunk(
            id=conversation_log_id,
            created=int(time.time()),
            model=model_name,
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(tool_calls=[tool_call_delta]),
                )
            ],
        )

        chunk_dict = chunk.model_dump(exclude_none=True)

        tool_call_data = chunk_dict["choices"][0]["delta"]["tool_calls"][0]
        tool_call_data["output_content"] = output_content_parts

        return chunk_dict
    except Exception as e:
        logger.warning(f"Failed to create tool output chunk: {e}")
        return None


def _create_tool_error_chunk(
    tool_call,
    error_content: str,
    conversation_log_id: str,
    model_name: str,
) -> dict | None:
    """Create a streaming chunk with error content in tool_calls delta."""
    try:
        tool_call_id = tool_call.id or f"call_{tool_call.index}"

        tool_call_delta = ChoiceDeltaToolCall(
            id=tool_call_id,
            index=tool_call.index or 0,
        )

        chunk = ChatCompletionChunk(
            id=conversation_log_id,
            created=int(time.time()),
            model=model_name,
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(tool_calls=[tool_call_delta]),
                )
            ],
        )

        chunk_dict = chunk.model_dump(exclude_none=True)

        tool_call_data = chunk_dict["choices"][0]["delta"]["tool_calls"][0]
        # Format error as text content part
        tool_call_data["output_content"] = [{"type": "text", "text": error_content}]

        return chunk_dict
    except Exception as e:
        logger.warning(f"Failed to create tool error chunk: {e}")
        return None


def _create_tool_error_event(
    tool_call_id: str,
    tool_name: str,
    error: str,
    conversation_log_id: str,
    model_name: str,
) -> str:
    """Create brave-chat.toolError event."""
    return _get_create_openai_response()(
        ToolError(tool_call_id=tool_call_id, tool_name=tool_name, error=error)
    )


def _format_web_sources(sources: list[dict], query: str | None) -> str:
    """Format web sources using search handler if available."""
    try:
        handler = get_global_registry().handlers.get("brave_search")
        if handler and hasattr(handler, "format_web_sources_content"):
            return handler.format_web_sources_content(sources, query)
    except Exception as e:
        logger.warning(f"Failed to format web sources: {e}")

    return f"Found {len(sources)} sources"


def _create_tool_end_event(
    tool_call_id: str,
    tool_name: str,
    conversation_log_id: str,
    model_name: str,
) -> str:
    """Create brave-chat.toolEnd event."""
    return _get_create_openai_response()(
        ToolEnd(tool_call_id=tool_call_id, tool_name=tool_name)
    )


def simplify_tool_message_for_llm(tool_message: ToolMessage) -> dict:
    """Convert a ToolMessage with rich content to a simple dict for LLM."""
    content = tool_message.content

    if isinstance(content, str):
        simplified_content = content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                part_dict = part
            elif hasattr(part, "model_dump"):
                part_dict = part.model_dump()
            elif hasattr(part, "text"):
                part_dict = {"text": part.text}
            else:
                logger.debug(f"Skipping unknown content part type: {type(part)}")
                continue

            part_type = part_dict.get("type")

            if part_type == "brave-chat.webSources":
                sources = part_dict.get("sources", [])
                query = part_dict.get("query")
                if sources:
                    text_parts.append(_format_web_sources(sources, query))
                elif "text" in part_dict:
                    text_parts.append(part_dict["text"])
            elif "text" in part_dict:
                text_parts.append(part_dict["text"])
            else:
                logger.debug(f"Content part has no text field, type={part_type}")
                if part_dict:
                    text_parts.append(str(part_dict))

        str_parts = [p for p in text_parts if p is not None and isinstance(p, str)]
        simplified_content = "\n\n".join(str_parts) if str_parts else "Tool completed"
    else:
        simplified_content = str(content)

    return {
        "role": "tool",
        "tool_call_id": tool_message.tool_call_id,
        "content": simplified_content,
    }


def simplify_messages_for_llm(messages: list[dict]) -> list[dict]:
    """
    Convert any tool messages with rich content (e.g., brave-chat.webSources) to plain text
    before sending to LLM, as models don't support rich content parts in tool messages.

    Uses the search handler's format_web_sources_content method via simplify_tool_message_for_llm.

    Args:
        messages: List of message dicts

    Returns:
        List of message dicts with tool messages simplified
    """
    simplified_messages = []
    for message in messages:
        if message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(part, dict) and part.get("type") == "brave-chat.webSources"
                for part in content
            ):
                tool_msg = ToolMessage(**message)
                simplified = simplify_tool_message_for_llm(tool_msg)
                simplified_messages.append(simplified)
            else:
                simplified_messages.append(message)
        else:
            simplified_messages.append(message)
    return simplified_messages


async def execute_streaming_tool_and_proxy_events(
    tool_call,
    conversation_log_id: str,
    model_name: str,
) -> AsyncGenerator[tuple[str | None, ToolMessage | None], None]:
    """Execute a streaming tool (like deep_research) and yield events.

    Args:
        tool_call: The tool call to execute
        conversation_log_id: Conversation log ID
        model_name: Model name for events

    Yields:
        Tuples of (event_string, tool_message_or_none)
    """
    tool_name = tool_call.function_name
    tool_call_id = tool_call.id or f"call_{tool_call.index}"
    FUNCTION_CALL_COUNTER.labels(function_name=tool_name).inc()

    # Emit tool start event
    try:
        tool_args = json.loads(tool_call.arguments) if tool_call.arguments else {}
    except json.JSONDecodeError:
        tool_args = {}

    start_event = _get_create_openai_response()(
        ToolStart(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
    )
    yield (start_event, None)

    # Execute streaming tool
    if tool_name == "deep_research":
        async for result in execute_deep_research_streaming(
            tool_call,
            tool_call_id,
            tool_args,
            conversation_log_id,
            model_name,
            _create_tool_output_chunk_from_parts,
        ):
            yield result

    else:
        # Unknown streaming tool
        error_event = _create_tool_error_event(
            tool_call_id,
            tool_name,
            f"Unknown streaming tool: {tool_name}",
            conversation_log_id,
            model_name,
        )
        yield (error_event, None)


async def handle_streaming_tool_calls(
    streaming_tool_calls: list,
    conversation_log_id: str,
    actual_model: str | None,
    mcp_tool_calls: list,
    client_tool_calls: list,
    assistant_tool_calls_for_message: list,
    request,
    backend,
    model_config,
    prompts_obj,
    original_client_tools: list,
    metrics_state: dict | None = None,
) -> AsyncIterator[str]:
    """Handle streaming tool execution and optional follow-up LLM call.

    Extracted from the route handler to keep open_ai_api.py focused on routing.

    Args:
        streaming_tool_calls: Tool calls requiring streaming execution
        conversation_log_id: Conversation log ID for event tracking
        actual_model: The actual model name from response
        mcp_tool_calls: Remaining MCP tool calls (non-streaming)
        client_tool_calls: Client-side tool calls
        assistant_tool_calls_for_message: Accumulated assistant tool calls
        request: The original OpenAI request
        backend: The backend service
        model_config: Model configuration
        prompts_obj: Prompts instance
        original_client_tools: Original client tools list
        metrics_state: Shared per-request TTFT/tools-use tracking state

    Yields:
        SSE-formatted event strings
    """
    logger.info(f"Executing {len(streaming_tool_calls)} streaming tool calls")
    tool_result_messages = []
    for tool_call in streaming_tool_calls:
        async for (
            event_str,
            tool_message,
        ) in execute_streaming_tool_and_proxy_events(
            tool_call,
            conversation_log_id or "unknown",
            actual_model or request.model,
        ):
            if event_str:
                yield event_str
            if tool_message:
                tool_result_messages.append(tool_message)

    # Skip follow-up for deep_research since it already emits the final answer
    is_deep_research_only = all(
        tc.function_name == "deep_research" for tc in streaming_tool_calls
    )
    if (
        tool_result_messages
        and not mcp_tool_calls
        and not client_tool_calls
        and not is_deep_research_only
    ):
        new_messages = [
            (
                message.model_dump(exclude_none=True)
                if hasattr(message, "model_dump")
                else message
            )
            for message in request.messages
        ]

        filtered_tool_calls = [
            tc
            for tc in assistant_tool_calls_for_message
            if any(tc.function.name == st.function_name for st in streaming_tool_calls)
        ]

        assistant_message = AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                (
                    OpenAIToolCall(
                        id=tc.id if hasattr(tc, "id") else tc.function.name,
                        type="function",
                        function=ToolCallFunction(
                            name=tc.function.name,
                            arguments=tc.function.arguments or "{}",
                        ),
                    )
                    if not hasattr(tc, "type")
                    else tc
                )
                for tc in filtered_tool_calls
            ],
        )
        new_messages.append(assistant_message.model_dump(exclude_none=True))

        new_messages.extend(
            (msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else msg)
            for msg in tool_result_messages
        )

        if prompts_obj:
            for prompt in prompts_obj.prompts:
                new_messages = prompt.augment(
                    new_messages,
                    model_config=model_config,
                    lang_code=request.selected_language,
                    tools=original_client_tools,
                )

        params = backend.build_params(
            request.stream, original_client_tools, new_messages
        )

        if request.model.startswith("near-"):
            params["api_key"] = external_service_settings.near_api_key

        logger.info("Making follow-up call to model with streaming tool results")
        followup_response = await backend.converse(new_messages, request.stream, params)

        # Lazy import to avoid circular dependency
        from aichat.serve.open_ai_api import process_streaming_response

        followup_request = request.model_copy(update={"messages": new_messages})

        async for chunk_str in process_streaming_response(
            followup_response,
            followup_request,
            original_client_tools,
            None,  # mcp_executor not needed for follow-up
            backend,
            model_config,
            prompts_obj,
            metrics_state=metrics_state,
        ):
            yield chunk_str
