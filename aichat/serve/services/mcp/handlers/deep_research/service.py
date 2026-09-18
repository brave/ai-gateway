"""Deep research streaming service.

Orchestrates the deep research streaming execution: streams events from
the deep research service, transforms them into Brave SSE format,
formats the final answer with citations, and emits WebSources output.
"""

import json
import logging
from collections.abc import AsyncGenerator

from aichat.protocol.open_ai_protocol import ToolMessage
from aichat.responses import ToolEnd, ToolError
from aichat.serve.services.mcp.handlers.deep_research.citations import (
    format_answer_with_citations,
    format_citations_plaintext,
)
from aichat.serve.services.mcp.handlers.deep_research.event_transformers import (
    create_completion_chunk,
    transform_deep_research_event,
)
from aichat.serve.services.mcp.handlers.deep_research.streaming_executor import (
    DeepResearchStreamingExecutor,
    iter_events_with_keepalive,
)
from aichat.serve.services.mcp.registry import get_global_registry

logger = logging.getLogger(__name__)


# Lazy import to avoid circular dependency with open_ai_api
def _get_create_openai_response():
    from aichat.serve.open_ai_api import create_openai_response

    return create_openai_response


def _create_tool_error_event(
    tool_call_id: str,
    tool_name: str,
    error: str,
) -> str:
    """Create brave-chat.toolError event."""
    return _get_create_openai_response()(
        ToolError(tool_call_id=tool_call_id, tool_name=tool_name, error=error)
    )


def _create_tool_end_event(
    tool_call_id: str,
    tool_name: str,
) -> str:
    """Create brave-chat.toolEnd event."""
    return _get_create_openai_response()(
        ToolEnd(tool_call_id=tool_call_id, tool_name=tool_name)
    )


async def execute_deep_research_streaming(
    tool_call,
    tool_call_id: str,
    tool_args: dict,
    conversation_log_id: str,
    model_name: str,
    create_tool_output_chunk_from_parts,
) -> AsyncGenerator[tuple[str | None, ToolMessage | None], None]:
    """Execute deep research and yield streaming events and the final tool message.

    Args:
        tool_call: The tool call being executed
        tool_call_id: The tool call ID
        tool_args: Parsed tool arguments
        conversation_log_id: Conversation log ID
        model_name: Model name for events
        create_tool_output_chunk_from_parts: Callback to create output chunks

    Yields:
        Tuples of (event_string, tool_message_or_none)
    """
    tool_name = tool_call.function_name
    query = tool_args.get("query", "")

    if not query:
        error_event = _create_tool_error_event(
            tool_call_id, tool_name, "Missing required 'query' argument"
        )
        yield (error_event, None)
        return

    executor = DeepResearchStreamingExecutor()
    final_answer = None
    final_citations = []

    async for event in iter_events_with_keepalive(executor.execute_streaming(query)):
        event_type = event.get("event")

        # Transform and yield event
        brave_event = transform_deep_research_event(
            event, conversation_log_id, model_name
        )
        if brave_event:
            yield (brave_event, None)

        # Track final answer for tool result
        if event_type == "answer" and event.get("final"):
            final_answer = event.get("answer", "")
            final_citations = event.get("citations", [])
        elif event_type == "error" or event.get("type") == "error":
            error_msg = event.get("error", "Unknown error")
            error_event = _create_tool_error_event(tool_call_id, tool_name, error_msg)
            yield (error_event, None)

    # Create tool result message and emit WebSources for citation URLs
    if final_answer:
        registry = get_global_registry()
        handler = registry.handlers.get("deep_research")

        if handler:
            raw_result = {
                "event": "answer",
                "final": True,
                "answer": final_answer,
                "citations": final_citations,
            }
            formatted_result = handler.format_result(tool_name, raw_result)

            # Emit WebSources tool output chunk BEFORE the completion chunk so
            # the browser's allowedLinks store is populated when it processes
            # [N] inline citations in the answer (avoids removeCitationsWithMissingLinks
            # stripping them).
            output_content_parts = handler.get_output_content_parts(
                formatted_result, tool_call
            )
            if output_content_parts:
                output_chunk = create_tool_output_chunk_from_parts(
                    tool_call,
                    output_content_parts,
                    conversation_log_id,
                    model_name,
                )
                if output_chunk:
                    chunk_json = json.dumps(output_chunk, separators=(",", ":"))
                    yield (f"data: {chunk_json}\n\n", None)

            # Emit the completion chunk with [N]-style citations now that
            # webSources has been sent and the browser can resolve them.
            formatted_answer = format_answer_with_citations(
                final_answer, final_citations
            )
            completion_event = create_completion_chunk(
                formatted_answer,
                conversation_log_id,
                model_name,
            )
            yield (completion_event, None)

            tool_message_content = handler.get_tool_message_content(
                formatted_result, tool_call
            )
        else:
            # Fallback without handler — emit answer then plaintext citations
            formatted_answer = format_answer_with_citations(
                final_answer, final_citations
            )
            completion_event = create_completion_chunk(
                formatted_answer,
                conversation_log_id,
                model_name,
            )
            yield (completion_event, None)
            tool_message_content = format_citations_plaintext(
                final_answer, final_citations
            )

        tool_message = ToolMessage(
            role="tool",
            tool_call_id=tool_call_id,
            content=tool_message_content,
        )
    else:
        # No final answer (timeout, error, or empty result).
        # Emit a minimal output_content chunk so the browser marks the
        # deep_research tool_use_event as server-handled (is_server_result=True).
        # Without this, the browser falls back to local tool lookup and emits
        # "The deep_research tool is not available."
        error_parts = [
            {
                "type": "text",
                "text": "Deep research did not produce a result.",
            }
        ]
        error_chunk = create_tool_output_chunk_from_parts(
            tool_call,
            error_parts,
            conversation_log_id,
            model_name,
        )
        if error_chunk:
            chunk_json = json.dumps(error_chunk, separators=(",", ":"))
            yield (f"data: {chunk_json}\n\n", None)

        tool_message = ToolMessage(
            role="tool",
            tool_call_id=tool_call_id,
            content="Deep research completed but no final answer was produced.",
        )

    # Emit tool end event
    end_event = _create_tool_end_event(tool_call_id, tool_name)
    yield (end_event, tool_message)
