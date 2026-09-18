import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

from aichat.llm.metrics import DUPLICATE_TOOL_CALL_ID_TOTAL
from aichat.protocol.open_ai_protocol import (
    ToolCall as OpenAIToolCall,
)
from aichat.serve.services.mcp.executor import MCPToolExecutor

logger = logging.getLogger(__name__)


@dataclass
class AlignmentCheckData:
    """Alignment check result with allowed status and reasoning."""

    allowed: bool
    reasoning: str | None = None


@dataclass
class ToolCall:
    """Represents a complete tool call with all accumulated data."""

    index: int
    id: str | None = None
    type: str = "function"
    function_name: str | None = None
    arguments: str = ""
    is_complete: bool = False
    alignment_check: AlignmentCheckData | None = None


@dataclass
class ToolCallAccumulator:
    """Accumulates streaming tool call data until complete."""

    tool_calls: dict[int, ToolCall] = field(default_factory=dict)

    def process_chunk(
        self, tool_calls_data: list[ChoiceDeltaToolCall]
    ) -> list[ToolCall]:
        """
        Process a chunk of tool call data from the streaming response.

        Args:
            tool_calls_data: List of OpenAI ChoiceDeltaToolCall objects from the streaming response

        Returns:
            List of completed tool calls ready for execution
        """
        completed_tools = []

        for tool_data in tool_calls_data:
            index = getattr(tool_data, "index", 0) or 0

            # Initialize tool call if not exists
            if index not in self.tool_calls:
                self.tool_calls[index] = ToolCall(index=index)

            tool_call = self.tool_calls[index]

            # Update tool call data
            if tool_data.id is not None:
                tool_call.id = tool_data.id

            if tool_data.type is not None:
                tool_call.type = tool_data.type

            if tool_data.function is not None:
                if tool_data.function.name is not None:
                    tool_call.function_name = tool_data.function.name

                if tool_data.function.arguments is not None:
                    tool_call.arguments += tool_data.function.arguments

        return completed_tools

    def finalize_tools(self) -> list[ToolCall]:
        """
        Finalize all tool calls when streaming ends.

        Args:
            stop_reason: The stop reason from the model response

        Returns:
            List of all completed tool calls
        """
        completed_tools = []

        for tool_call in self.tool_calls.values():
            if tool_call.function_name and self._is_valid_arguments(
                tool_call.arguments
            ):
                tool_call.is_complete = True
                completed_tools.append(tool_call)

        return completed_tools

    def _is_valid_arguments(self, arguments: str) -> bool:
        """
        Check if the arguments string forms valid JSON.

        Args:
            arguments: The arguments string to validate

        Returns:
            True if arguments is valid JSON or empty, False otherwise
        """
        if not arguments:
            # Empty arguments are valid (some functions might not need arguments)
            return True

        try:
            json.loads(arguments)
            return True
        except json.JSONDecodeError:
            return False


class ToolExecutor:
    """Executes tool calls using function implementations or MCP tools."""

    def __init__(self, mcp_executor: MCPToolExecutor | None = None):
        """
        Initialize the tool executor.

        Args:
            mcp_executor: Optional MCP tool executor for handling MCP tools
        """
        self.mcp_executor = mcp_executor

    async def execute_tool_call(
        self, tool_call: ToolCall, model: str = "unknown", **kwargs
    ) -> list[Any] | OpenAIToolCall | None:
        """
        Execute a single tool call.

        Args:
            tool_call: The tool call to execute
            **kwargs: Additional parameters needed by tool implementations
        """
        if not tool_call.is_complete or not tool_call.function_name:
            logger.warning(
                f"Tool call {tool_call.index} is not complete or missing function name"
            )
            return None

        function_name = tool_call.function_name

        if self.mcp_executor and await self.mcp_executor.is_mcp_tool(function_name):
            try:
                args = json.loads(tool_call.arguments) if tool_call.arguments else {}
                result = await self.mcp_executor.execute_mcp_tool(
                    function_name, args, model=model
                )
                logger.debug(f"MCP tool {function_name} returned: {result}")

                if result:
                    if isinstance(result, dict) and result.get("type") == "error":
                        logger.info(f"Detected error result for {function_name}")
                    return [result]
                return None
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse MCP tool arguments: {e}")
                return None
            except Exception as e:
                logger.error(f"Failed to execute MCP tool {function_name}: {e}")
                return None

    async def execute_tool_calls(
        self, tool_calls: list[ToolCall], **kwargs
    ) -> dict[str, list[Any] | OpenAIToolCall | None]:
        """
        Execute multiple tool calls and return a mapping of tool_call_id to results.

        Args:
            tool_calls: List of tool calls to execute
            **kwargs: Additional parameters needed by tool implementations

        Returns:
            Dictionary mapping tool_call_id to tool execution results:
            - For server-side tools: list of events
            - For client-forwarded tools: OpenAIToolCall object
            - For failed executions: None
        """
        results = {}

        for tool_call in tool_calls:
            tool_call_id = tool_call.id or f"tool_call_{tool_call.index}"
            result = await self.execute_tool_call(tool_call, **kwargs)
            results[tool_call_id] = result

        return results


# Create singleton instance
tool_executor = ToolExecutor()


def parse_streaming_tool_calls(
    accumulator: ToolCallAccumulator,
    tool_calls: list[ChoiceDeltaToolCall] | None,
    stop_reason: str | None,
) -> list[ToolCall]:
    """
    Parse tool calls from a streaming chunk.

    Args:
        accumulator: Tool call accumulator instance
        tool_calls: List of OpenAI ChoiceDeltaToolCall objects
        stop_reason: The finish reason from the chunk

    Returns:
        List of completed tool calls (empty if none completed in this chunk)
    """
    if tool_calls:
        completed_tools = accumulator.process_chunk(tool_calls)

        # Check if this is the final chunk with tool calls
        if stop_reason == "tool_calls":
            completed_tools.extend(accumulator.finalize_tools())
            return completed_tools

        return completed_tools
    elif stop_reason == "tool_calls":
        # Handle case where stop_reason is tool_calls but no tool_calls in this chunk
        # This means all tool calls are complete
        return accumulator.finalize_tools()

    return []


def report_duplicate_tool_call_ids(tool_calls: list[ToolCall], model: str) -> None:
    """
    Scan accumulated tool calls for duplicate IDs and fire a metric for each
    duplicate found. Logs a warning for observability. Detection only - the
    duplicates are not removed from the list.
    """
    seen_ids: dict[str, str] = {}
    for tc in tool_calls:
        tc_id = tc.id or f"call_{tc.index}"
        if tc_id in seen_ids:
            logger.warning(
                f"Duplicate tool call ID detected: id={tc_id!r} "
                f"tool={tc.function_name!r} model={model!r}"
            )
            DUPLICATE_TOOL_CALL_ID_TOTAL.labels(
                model=model,
                tool=tc.function_name or "unknown",
            ).inc()
        else:
            seen_ids[tc_id] = tc.function_name or "unknown"
