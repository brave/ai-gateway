"""Tests for MCP tool execution and event streaming."""

import os

# Set minimal env vars to avoid validation errors
os.environ.setdefault("RATE_LIMITING_ENABLED", "false")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("NEAR_API_KEY", "test_near_key")

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aichat.protocol.open_ai_protocol import ToolMessage
from aichat.serve.mcp_tool_execution import execute_tools_and_stream_events
from aichat.serve.services.mcp.executor import MCPToolExecutor
from aichat.serve.tool_parser import ToolCall


@pytest.fixture
def mock_mcp_executor():
    """Create a mock MCP executor."""
    executor = Mock(spec=MCPToolExecutor)
    executor._tool_cache = {}
    executor._get_mcp_client = AsyncMock()
    executor._fetch_all_tools = AsyncMock(return_value=[])
    return executor


@pytest.fixture
def sample_tool_call():
    """Create a sample tool call for testing."""
    tool_call = ToolCall(index=0)
    tool_call.id = "call_123"
    tool_call.function_name = "test_tool"
    tool_call.arguments = json.dumps({"query": "test"})
    tool_call.is_complete = True
    return tool_call


@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor."""
    executor = Mock()
    executor.execute_tool_call = AsyncMock()
    return executor


@pytest.mark.asyncio
async def test_tool_start_event_generated(
    mock_mcp_executor, sample_tool_call, mock_tool_executor
):
    """Test that ToolStart event is generated when tool execution begins."""
    # Mock successful tool execution
    mock_tool_executor.execute_tool_call.return_value = [
        {"type": "text", "content": "Tool result"}
    ]

    with (
        patch(
            "aichat.serve.mcp_tool_execution.ToolExecutor",
            return_value=mock_tool_executor,
        ),
        patch(
            "aichat.serve.mcp_tool_execution.get_global_registry",
            return_value=Mock(),
        ),
        patch(
            "aichat.serve.mcp_tool_execution._get_create_openai_response"
        ) as mock_create_response,
    ):
        mock_create_response.return_value = (
            lambda x: f"data: {json.dumps({'object': 'brave-chat.toolStart', 'tool_call_id': x.tool_call_id, 'tool_name': x.tool_name})}\n\n"
        )

        events = []
        async for event_str, _output_or_message in execute_tools_and_stream_events(
            [sample_tool_call],
            mock_mcp_executor,
            "conv_123",
            "test-model",
            "test-model",
        ):
            if event_str:
                events.append(event_str)

        # Verify ToolStart event was generated
        assert len(events) >= 1
        assert "brave-chat.toolStart" in events[0]
        assert "call_123" in events[0]
        assert "test_tool" in events[0]


@pytest.mark.asyncio
async def test_tool_error_event_generated_on_failure(
    mock_mcp_executor, sample_tool_call, mock_tool_executor
):
    """Test that ToolError event is generated when tool execution fails."""
    # Mock tool execution returning None (failure)
    mock_tool_executor.execute_tool_call.return_value = None

    with (
        patch(
            "aichat.serve.mcp_tool_execution.ToolExecutor",
            return_value=mock_tool_executor,
        ),
        patch(
            "aichat.serve.mcp_tool_execution.get_global_registry",
            return_value=Mock(),
        ),
        patch(
            "aichat.serve.mcp_tool_execution._get_create_openai_response"
        ) as mock_create_response,
    ):

        def create_response(response_obj):
            if hasattr(response_obj, "error"):
                return f"data: {json.dumps({'object': 'brave-chat.toolError', 'tool_call_id': response_obj.tool_call_id, 'tool_name': response_obj.tool_name, 'error': response_obj.error})}\n\n"
            else:
                return f"data: {json.dumps({'object': 'brave-chat.toolStart', 'tool_call_id': response_obj.tool_call_id, 'tool_name': response_obj.tool_name})}\n\n"

        mock_create_response.return_value = create_response

        events = []
        async for event_str, _output_or_message in execute_tools_and_stream_events(
            [sample_tool_call],
            mock_mcp_executor,
            "conv_123",
            "test-model",
            "test-model",
        ):
            if event_str:
                events.append(event_str)

        # Verify ToolError event was generated
        error_events = [e for e in events if "brave-chat.toolError" in e]
        assert len(error_events) >= 1, "ToolError event should be generated on failure"

        # Verify ToolEnd event was NOT generated even after error
        tool_end_events = [
            e for e in events if "brave-chat.toolEnd" in e or "toolEnd" in e
        ]
        assert (
            len(tool_end_events) == 0
        ), "ToolEnd events should not be generated even after errors"


@pytest.mark.asyncio
async def test_tool_error_event_generated_on_exception(
    mock_mcp_executor, sample_tool_call, mock_tool_executor
):
    """Test that ToolError event is generated when tool execution raises exception."""
    # Mock tool execution raising an exception
    mock_tool_executor.execute_tool_call.side_effect = Exception(
        "Tool execution failed"
    )

    with (
        patch(
            "aichat.serve.mcp_tool_execution.ToolExecutor",
            return_value=mock_tool_executor,
        ),
        patch(
            "aichat.serve.mcp_tool_execution.get_global_registry",
            return_value=Mock(),
        ),
        patch(
            "aichat.serve.mcp_tool_execution._get_create_openai_response"
        ) as mock_create_response,
    ):

        def create_response(response_obj):
            if hasattr(response_obj, "error"):
                return f"data: {json.dumps({'object': 'brave-chat.toolError', 'tool_call_id': response_obj.tool_call_id, 'tool_name': response_obj.tool_name, 'error': response_obj.error})}\n\n"
            else:
                return f"data: {json.dumps({'object': 'brave-chat.toolStart', 'tool_call_id': response_obj.tool_call_id, 'tool_name': response_obj.tool_name})}\n\n"

        mock_create_response.return_value = create_response

        events = []
        async for event_str, _output_or_message in execute_tools_and_stream_events(
            [sample_tool_call],
            mock_mcp_executor,
            "conv_123",
            "test-model",
            "test-model",
        ):
            if event_str:
                events.append(event_str)

        # Verify ToolError event was generated
        error_events = [e for e in events if "brave-chat.toolError" in e]
        assert (
            len(error_events) >= 1
        ), "ToolError event should be generated on exception"

        # Verify ToolEnd event was NOT generated even after exception
        tool_end_events = [
            e for e in events if "brave-chat.toolEnd" in e or "toolEnd" in e
        ]
        assert (
            len(tool_end_events) == 0
        ), "ToolEnd events should not be generated even after exceptions"


@pytest.mark.asyncio
async def test_tool_message_returned_on_success(
    mock_mcp_executor, sample_tool_call, mock_tool_executor
):
    """Test that ToolMessage is returned after successful tool execution."""
    # Mock successful tool execution
    mock_tool_executor.execute_tool_call.return_value = [
        {"type": "text", "content": "Tool result"}
    ]

    with (
        patch(
            "aichat.serve.mcp_tool_execution.ToolExecutor",
            return_value=mock_tool_executor,
        ),
        patch(
            "aichat.serve.mcp_tool_execution.get_global_registry",
            return_value=Mock(),
        ),
        patch(
            "aichat.serve.mcp_tool_execution._get_create_openai_response"
        ) as mock_create_response,
    ):
        mock_create_response.return_value = (
            lambda x: f"data: {json.dumps({'object': 'brave-chat.toolStart', 'tool_call_id': x.tool_call_id, 'tool_name': x.tool_name})}\n\n"
        )

        messages = []
        async for _event_str, output_or_message in execute_tools_and_stream_events(
            [sample_tool_call],
            mock_mcp_executor,
            "conv_123",
            "test-model",
            "test-model",
        ):
            if output_or_message and isinstance(output_or_message, ToolMessage):
                messages.append(output_or_message)

        # Verify ToolMessage was returned
        assert len(messages) >= 1
        assert messages[0].role == "tool"
        assert messages[0].tool_call_id == "call_123"


@pytest.mark.asyncio
async def test_web_sources_output_chunk_generated(
    mock_mcp_executor, sample_tool_call, mock_tool_executor
):
    """Test that web sources are sent as output chunks."""
    # Mock tool execution returning web sources with server_name
    mock_tool_executor.execute_tool_call.return_value = [
        {
            "type": "brave-web-sources",
            "server_name": "brave_search",
            "sources": [
                {"title": "Test Source", "url": "https://example.com", "favicon": None}
            ],
            "content": "Found sources",
        }
    ]

    # Create a mock handler that returns output content parts
    mock_handler = Mock()
    mock_handler.get_output_content_parts.return_value = [
        {
            "type": "brave-chat.webSources",
            "sources": [
                {"title": "Test Source", "url": "https://example.com", "favicon": None}
            ],
        }
    ]

    # Create a mock registry with the handler
    mock_registry = Mock()
    mock_registry.handlers = {"brave_search": mock_handler}

    with (
        patch(
            "aichat.serve.mcp_tool_execution.ToolExecutor",
            return_value=mock_tool_executor,
        ),
        patch(
            "aichat.serve.mcp_tool_execution.get_global_registry",
            return_value=mock_registry,
        ),
        patch(
            "aichat.serve.mcp_tool_execution._get_create_openai_response"
        ) as mock_create_response,
    ):
        mock_create_response.return_value = (
            lambda x: f"data: {json.dumps({'object': 'brave-chat.toolStart', 'tool_call_id': x.tool_call_id, 'tool_name': x.tool_name})}\n\n"
        )

        output_chunks = []
        async for event_str, output_or_message in execute_tools_and_stream_events(
            [sample_tool_call],
            mock_mcp_executor,
            "conv_123",
            "test-model",
            "test-model",
        ):
            # Check for output chunks (dicts with choices and output_content in tool_calls)
            if (
                event_str is None
                and isinstance(output_or_message, dict)
                and "choices" in output_or_message
            ):
                choices = output_or_message.get("choices", [])
                if choices and "delta" in choices[0]:
                    delta = choices[0]["delta"]
                    if delta.get("tool_calls"):
                        tool_call = delta["tool_calls"][0]
                        if "output_content" in tool_call:
                            output_chunks.append(output_or_message)

        # Verify output chunk with web sources was generated
        assert len(output_chunks) >= 1
        chunk = output_chunks[0]
        assert "tool_calls" in chunk.get("choices", [{}])[0].get("delta", {})
        tool_calls = chunk["choices"][0]["delta"]["tool_calls"]
        assert len(tool_calls) > 0
        assert "output_content" in tool_calls[0]


@pytest.mark.asyncio
async def test_multiple_tools_executed_sequentially(
    mock_mcp_executor, mock_tool_executor
):
    """Test that multiple tools are executed sequentially without ToolEnd."""
    tool_call1 = ToolCall(index=0)
    tool_call1.id = "call_1"
    tool_call1.function_name = "tool1"
    tool_call1.arguments = json.dumps({"param": "value1"})
    tool_call1.is_complete = True

    tool_call2 = ToolCall(index=1)
    tool_call2.id = "call_2"
    tool_call2.function_name = "tool2"
    tool_call2.arguments = json.dumps({"param": "value2"})
    tool_call2.is_complete = True

    # Mock successful tool execution
    mock_tool_executor.execute_tool_call.return_value = [
        {"type": "text", "content": "Tool result"}
    ]

    with (
        patch(
            "aichat.serve.mcp_tool_execution.ToolExecutor",
            return_value=mock_tool_executor,
        ),
        patch(
            "aichat.serve.mcp_tool_execution.get_global_registry",
            return_value=Mock(),
        ),
        patch(
            "aichat.serve.mcp_tool_execution._get_create_openai_response"
        ) as mock_create_response,
    ):
        mock_create_response.return_value = (
            lambda x: f"data: {json.dumps({'object': 'brave-chat.toolStart', 'tool_call_id': x.tool_call_id, 'tool_name': x.tool_name})}\n\n"
        )

        events = []
        async for event_str, _output_or_message in execute_tools_and_stream_events(
            [tool_call1, tool_call2],
            mock_mcp_executor,
            "conv_123",
            "test-model",
            "test-model",
        ):
            if event_str:
                events.append(event_str)

        # Verify both tools got ToolStart events
        tool_start_events = [e for e in events if "brave-chat.toolStart" in e]
        assert len(tool_start_events) == 2

        # Verify no ToolEnd events were generated
        tool_end_events = [
            e for e in events if "brave-chat.toolEnd" in e or "toolEnd" in e
        ]
        assert len(tool_end_events) == 0


@pytest.mark.asyncio
async def test_client_side_tool_skipped(
    mock_mcp_executor, sample_tool_call, mock_tool_executor
):
    """Test that client-side tools are skipped without generating events."""
    from aichat.protocol.open_ai_protocol import ToolCall as OpenAIToolCall

    # Mock tool execution returning OpenAIToolCall (client-side)
    client_tool_call = OpenAIToolCall(
        function={"name": "client_tool", "arguments": "{}"}, type="function"
    )
    mock_tool_executor.execute_tool_call.return_value = client_tool_call

    with (
        patch(
            "aichat.serve.mcp_tool_execution.ToolExecutor",
            return_value=mock_tool_executor,
        ),
        patch(
            "aichat.serve.mcp_tool_execution.get_global_registry",
            return_value=Mock(),
        ),
        patch(
            "aichat.serve.mcp_tool_execution._get_create_openai_response"
        ) as mock_create_response,
    ):
        mock_create_response.return_value = (
            lambda x: f"data: {json.dumps({'object': 'brave-chat.toolStart', 'tool_call_id': x.tool_call_id, 'tool_name': x.tool_name})}\n\n"
        )

        events = []
        messages = []
        async for event_str, output_or_message in execute_tools_and_stream_events(
            [sample_tool_call],
            mock_mcp_executor,
            "conv_123",
            "test-model",
            "test-model",
        ):
            if event_str:
                events.append(event_str)
            if output_or_message:
                messages.append(output_or_message)

        # Verify ToolStart was generated
        tool_start_events = [e for e in events if "brave-chat.toolStart" in e]
        assert len(tool_start_events) == 1

        # Verify no ToolMessage was returned (client-side tool skipped)
        assert len(messages) == 0

        # Verify no ToolEnd events
        tool_end_events = [
            e for e in events if "brave-chat.toolEnd" in e or "toolEnd" in e
        ]
        assert len(tool_end_events) == 0


@pytest.mark.asyncio
async def test_error_chunk_yielded_on_failure(
    mock_mcp_executor, sample_tool_call, mock_tool_executor
):
    """Test that error chunk is yielded when tool execution fails."""
    # Mock tool execution returning None (failure)
    mock_tool_executor.execute_tool_call.return_value = None

    with (
        patch(
            "aichat.serve.mcp_tool_execution.ToolExecutor",
            return_value=mock_tool_executor,
        ),
        patch(
            "aichat.serve.mcp_tool_execution.get_global_registry",
            return_value=Mock(),
        ),
        patch(
            "aichat.serve.mcp_tool_execution._get_create_openai_response"
        ) as mock_create_response,
    ):

        def create_response(response_obj):
            if hasattr(response_obj, "error"):
                return f"data: {json.dumps({'object': 'brave-chat.toolError', 'tool_call_id': response_obj.tool_call_id, 'tool_name': response_obj.tool_name, 'error': response_obj.error})}\n\n"
            else:
                return f"data: {json.dumps({'object': 'brave-chat.toolStart', 'tool_call_id': response_obj.tool_call_id, 'tool_name': response_obj.tool_name})}\n\n"

        mock_create_response.return_value = create_response

        error_chunks = []
        async for event_str, output_or_message in execute_tools_and_stream_events(
            [sample_tool_call],
            mock_mcp_executor,
            "conv_123",
            "test-model",
            "test-model",
        ):
            # Error chunks are yielded as (None, chunk_dict)
            if (
                event_str is None
                and isinstance(output_or_message, dict)
                and "choices" in output_or_message
            ):
                error_chunks.append(output_or_message)

        # Verify error chunk was yielded
        assert len(error_chunks) >= 1, "Error chunk should be yielded on failure"

        # Verify error chunk structure
        chunk = error_chunks[0]
        assert "choices" in chunk
        assert len(chunk["choices"]) > 0
        assert "delta" in chunk["choices"][0]
        assert "tool_calls" in chunk["choices"][0]["delta"]
        assert len(chunk["choices"][0]["delta"]["tool_calls"]) > 0

        tool_call_data = chunk["choices"][0]["delta"]["tool_calls"][0]
        assert "output_content" in tool_call_data
        assert len(tool_call_data["output_content"]) > 0
        assert tool_call_data["output_content"][0]["type"] == "text"
        assert "Error:" in tool_call_data["output_content"][0]["text"]


@pytest.mark.asyncio
async def test_error_chunk_yielded_on_exception(
    mock_mcp_executor, sample_tool_call, mock_tool_executor
):
    """Test that error chunk is yielded when tool execution raises exception."""
    # Mock tool execution raising an exception
    mock_tool_executor.execute_tool_call.side_effect = Exception(
        "Tool execution failed"
    )

    with (
        patch(
            "aichat.serve.mcp_tool_execution.ToolExecutor",
            return_value=mock_tool_executor,
        ),
        patch(
            "aichat.serve.mcp_tool_execution.get_global_registry",
            return_value=Mock(),
        ),
        patch(
            "aichat.serve.mcp_tool_execution._get_create_openai_response"
        ) as mock_create_response,
    ):

        def create_response(response_obj):
            if hasattr(response_obj, "error"):
                return f"data: {json.dumps({'object': 'brave-chat.toolError', 'tool_call_id': response_obj.tool_call_id, 'tool_name': response_obj.tool_name, 'error': response_obj.error})}\n\n"
            else:
                return f"data: {json.dumps({'object': 'brave-chat.toolStart', 'tool_call_id': response_obj.tool_call_id, 'tool_name': response_obj.tool_name})}\n\n"

        mock_create_response.return_value = create_response

        error_chunks = []
        async for event_str, output_or_message in execute_tools_and_stream_events(
            [sample_tool_call],
            mock_mcp_executor,
            "conv_123",
            "test-model",
            "test-model",
        ):
            # Error chunks are yielded as (None, chunk_dict)
            if (
                event_str is None
                and isinstance(output_or_message, dict)
                and "choices" in output_or_message
            ):
                error_chunks.append(output_or_message)

        # Verify error chunk was yielded
        assert len(error_chunks) >= 1, "Error chunk should be yielded on exception"

        # Verify error chunk structure
        chunk = error_chunks[0]
        assert "choices" in chunk
        assert len(chunk["choices"]) > 0
        assert "delta" in chunk["choices"][0]
        assert "tool_calls" in chunk["choices"][0]["delta"]
        assert len(chunk["choices"][0]["delta"]["tool_calls"]) > 0

        tool_call_data = chunk["choices"][0]["delta"]["tool_calls"][0]
        assert "output_content" in tool_call_data
        assert len(tool_call_data["output_content"]) > 0
        assert tool_call_data["output_content"][0]["type"] == "text"
        assert "Tool execution failed" in tool_call_data["output_content"][0]["text"]


@pytest.mark.asyncio
async def test_error_chunk_format(
    mock_mcp_executor, sample_tool_call, mock_tool_executor
):
    """Test that error chunk has correct format with output_content."""
    # Mock tool execution returning None (failure)
    mock_tool_executor.execute_tool_call.return_value = None

    with (
        patch(
            "aichat.serve.mcp_tool_execution.ToolExecutor",
            return_value=mock_tool_executor,
        ),
        patch(
            "aichat.serve.mcp_tool_execution.get_global_registry",
            return_value=Mock(),
        ),
        patch(
            "aichat.serve.mcp_tool_execution._get_create_openai_response"
        ) as mock_create_response,
    ):

        def create_response(response_obj):
            if hasattr(response_obj, "error"):
                return f"data: {json.dumps({'object': 'brave-chat.toolError', 'tool_call_id': response_obj.tool_call_id, 'tool_name': response_obj.tool_name, 'error': response_obj.error})}\n\n"
            else:
                return f"data: {json.dumps({'object': 'brave-chat.toolStart', 'tool_call_id': response_obj.tool_call_id, 'tool_name': response_obj.tool_name})}\n\n"

        mock_create_response.return_value = create_response

        error_chunks = []
        async for event_str, output_or_message in execute_tools_and_stream_events(
            [sample_tool_call],
            mock_mcp_executor,
            "conv_123",
            "test-model",
            "test-model",
        ):
            # Error chunks are yielded as (None, chunk_dict)
            if (
                event_str is None
                and isinstance(output_or_message, dict)
                and "choices" in output_or_message
            ):
                error_chunks.append(output_or_message)

        # Verify error chunk format
        assert len(error_chunks) >= 1

        chunk = error_chunks[0]
        # Verify chunk has required fields
        assert "id" in chunk
        assert "created" in chunk
        assert "model" in chunk
        assert chunk["model"] == "test-model"
        assert "object" in chunk
        assert chunk["object"] == "chat.completion.chunk"

        # Verify tool_call structure
        tool_call_data = chunk["choices"][0]["delta"]["tool_calls"][0]
        assert "id" in tool_call_data
        assert tool_call_data["id"] == "call_123"
        assert "index" in tool_call_data
        assert tool_call_data["index"] == 0

        # Verify output_content format
        assert "output_content" in tool_call_data
        output_content = tool_call_data["output_content"]
        assert isinstance(output_content, list)
        assert len(output_content) == 1
        assert output_content[0]["type"] == "text"
        assert "text" in output_content[0]
        assert isinstance(output_content[0]["text"], str)


# Tests for simplify_tool_message_for_llm and simplify_messages_for_llm
from aichat.serve.mcp_tool_execution import (
    simplify_messages_for_llm,
    simplify_tool_message_for_llm,
)


def test_simplify_tool_message_with_string_content():
    """Test simplifying a tool message with plain string content."""
    tool_message = ToolMessage(
        role="tool",
        tool_call_id="call_123",
        content="Simple string result",
    )

    result = simplify_tool_message_for_llm(tool_message)

    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_123"
    assert result["content"] == "Simple string result"


def test_simplify_tool_message_with_text_parts():
    """Test simplifying a tool message with text content parts."""
    tool_message = ToolMessage(
        role="tool",
        tool_call_id="call_123",
        content=[
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ],
    )

    result = simplify_tool_message_for_llm(tool_message)

    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_123"
    assert "First part" in result["content"]
    assert "Second part" in result["content"]
    assert "\n\n" in result["content"]  # Double newline separator


def test_simplify_tool_message_with_web_sources():
    """Test simplifying a tool message with web sources."""
    tool_message = ToolMessage(
        role="tool",
        tool_call_id="call_123",
        content=[
            {"type": "text", "text": "Search results:"},
            {
                "type": "brave-chat.webSources",
                "sources": [
                    {"title": "Test", "url": "https://test.com"},
                    {"title": "Example", "url": "https://example.com"},
                ],
                "query": "test query",
            },
        ],
    )

    with patch("aichat.serve.mcp_tool_execution.get_global_registry") as mock_registry:
        mock_handler = Mock()
        mock_handler.format_web_sources_content.return_value = (
            "Found 2 sources: Test, Example"
        )
        mock_registry.return_value.handlers.get.return_value = mock_handler

        result = simplify_tool_message_for_llm(tool_message)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert "Search results:" in result["content"]
        assert "Found 2 sources" in result["content"]


def test_simplify_tool_message_with_empty_web_sources_fallback():
    """Test that empty web sources fall back to text field."""
    tool_message = ToolMessage(
        role="tool",
        tool_call_id="call_123",
        content=[
            {
                "type": "brave-chat.webSources",
                "sources": [],  # Empty sources
                "text": "No sources found",
            },
        ],
    )

    result = simplify_tool_message_for_llm(tool_message)

    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_123"
    assert result["content"] == "No sources found"


def test_simplify_tool_message_with_mixed_content():
    """Test handling mixed text and web sources content."""
    tool_message = ToolMessage(
        role="tool",
        tool_call_id="call_123",
        content=[
            {"type": "text", "text": "Introduction text"},
            {"type": "text", "text": "More details"},
        ],
    )

    result = simplify_tool_message_for_llm(tool_message)

    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_123"
    assert "Introduction text" in result["content"]
    assert "More details" in result["content"]
    # Check for double newline separator
    assert result["content"] == "Introduction text\n\nMore details"


def test_simplify_tool_message_web_sources_with_text_fallback():
    """Test web sources without handler returns count fallback."""
    tool_message = ToolMessage(
        role="tool",
        tool_call_id="call_123",
        content=[
            {
                "type": "brave-chat.webSources",
                "sources": [{"title": "Test", "url": "https://test.com"}],
            },
        ],
    )

    with patch("aichat.serve.mcp_tool_execution.get_global_registry") as mock_registry:
        # No handler available
        mock_registry.return_value.handlers.get.return_value = None

        result = simplify_tool_message_for_llm(tool_message)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        # Should get the default fallback when no handler
        assert "Found 1 sources" in result["content"]


def test_simplify_tool_message_handles_text_content_part():
    """Test handling TextContentPart with text attribute."""
    from aichat.protocol.open_ai_protocol import TextContentPart

    content_part = TextContentPart(type="text", text="Content from part")

    tool_message = ToolMessage(
        role="tool",
        tool_call_id="call_123",
        content=[content_part],
    )

    result = simplify_tool_message_for_llm(tool_message)

    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_123"
    assert result["content"] == "Content from part"


def test_simplify_tool_message_with_only_web_sources():
    """Test simplifying a tool message with only web sources (no text part)."""
    tool_message = ToolMessage(
        role="tool",
        tool_call_id="call_123",
        content=[
            {
                "type": "brave-chat.webSources",
                "sources": [
                    {"title": "Test Source", "url": "https://test.com"},
                ],
                "query": "test query",
            },
        ],
    )

    with patch("aichat.serve.mcp_tool_execution.get_global_registry") as mock_registry:
        mock_handler = Mock()
        mock_handler.format_web_sources_content.return_value = (
            "Found 1 source: Test Source"
        )
        mock_registry.return_value.handlers.get.return_value = mock_handler

        result = simplify_tool_message_for_llm(tool_message)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert isinstance(result["content"], str)
        assert "Found 1 source" in result["content"]
        assert "Test Source" in result["content"]


# Tests for simplify_messages_for_llm
def test_simplify_messages_for_llm_with_tool_message_web_sources():
    """Test simplifying messages list with tool message containing web sources."""
    messages = [
        {"role": "user", "content": "Search for something"},
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": [
                {
                    "type": "brave-chat.webSources",
                    "sources": [
                        {"title": "Test Source", "url": "https://test.com"},
                    ],
                    "query": "test query",
                }
            ],
        },
        {"role": "assistant", "content": "Response"},
    ]

    with patch("aichat.serve.mcp_tool_execution.get_global_registry") as mock_registry:
        mock_handler = Mock()
        mock_handler.format_web_sources_content.return_value = (
            "Found 1 source: Test Source"
        )
        mock_registry.return_value.handlers.get.return_value = mock_handler

        result = simplify_messages_for_llm(messages)

        assert len(result) == 3
        assert result[0] == messages[0]  # User message unchanged
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_123"
        assert isinstance(result[1]["content"], str)
        assert "Found 1 source" in result[1]["content"]
        assert result[2] == messages[2]  # Assistant message unchanged


def test_simplify_messages_for_llm_with_plain_text_tool_message():
    """Test that plain text tool messages are left unchanged."""
    messages = [
        {"role": "user", "content": "Get weather"},
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "Temperature: 72°F",
        },
    ]

    result = simplify_messages_for_llm(messages)

    assert len(result) == 2
    assert result[0] == messages[0]
    assert result[1] == messages[1]  # Plain text tool message unchanged


def test_simplify_messages_for_llm_with_mixed_tool_content():
    """Test tool message with mixed content (text and web sources)."""
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": [
                {"type": "text", "text": "Search results:"},
                {
                    "type": "brave-chat.webSources",
                    "sources": [{"title": "Test", "url": "https://test.com"}],
                    "query": "test",
                },
            ],
        }
    ]

    with patch("aichat.serve.mcp_tool_execution.get_global_registry") as mock_registry:
        mock_handler = Mock()
        mock_handler.format_web_sources_content.return_value = "Found 1 source: Test"
        mock_registry.return_value.handlers.get.return_value = mock_handler

        result = simplify_messages_for_llm(messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert isinstance(result[0]["content"], str)
        assert "Search results:" in result[0]["content"]
        assert "Found 1 source" in result[0]["content"]


def test_simplify_messages_for_llm_with_non_tool_messages():
    """Test that non-tool messages pass through unchanged."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = simplify_messages_for_llm(messages)

    assert result == messages  # All messages unchanged


def test_simplify_messages_for_llm_with_rich_results():
    """Test tool message with web sources that includes rich_results."""
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": [
                {
                    "type": "brave-chat.webSources",
                    "sources": [{"title": "Test", "url": "https://test.com"}],
                    "query": "test",
                    "rich_results": [{"type": "rich", "subtype": "weather"}],
                }
            ],
        }
    ]

    with patch("aichat.serve.mcp_tool_execution.get_global_registry") as mock_registry:
        mock_handler = Mock()
        mock_handler.format_web_sources_content.return_value = "Found 1 source: Test"
        mock_registry.return_value.handlers.get.return_value = mock_handler

        result = simplify_messages_for_llm(messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert isinstance(result[0]["content"], str)
        assert "Found 1 source" in result[0]["content"]
