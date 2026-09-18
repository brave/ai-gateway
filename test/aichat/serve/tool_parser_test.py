from unittest.mock import patch

import pytest
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from aichat.protocol.open_ai_protocol import (
    ToolCall as OpenAIToolCall,
)
from aichat.protocol.open_ai_protocol import (
    ToolCallFunction,
)
from aichat.serve.tool_parser import (
    ToolCall,
    ToolCallAccumulator,
    parse_streaming_tool_calls,
    tool_executor,
)


class TestToolCall:
    def test_tool_call_initialization_defaults(self):
        """Test ToolCall initialization with default values"""
        tool_call = ToolCall(index=0)

        assert tool_call.index == 0
        assert tool_call.id is None
        assert tool_call.type == "function"
        assert tool_call.function_name is None
        assert tool_call.arguments == ""
        assert tool_call.is_complete is False

    def test_tool_call_initialization_with_values(self):
        """Test ToolCall initialization with specific values"""
        tool_call = ToolCall(
            index=1,
            id="call_123",
            type="function",
            function_name="get_weather",
            arguments='{"location": "San Francisco"}',
            is_complete=True,
        )

        assert tool_call.index == 1
        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function_name == "get_weather"
        assert tool_call.arguments == '{"location": "San Francisco"}'
        assert tool_call.is_complete is True


class TestToolCallAccumulator:
    @pytest.fixture
    def accumulator(self):
        """Create a fresh ToolCallAccumulator for each test"""
        return ToolCallAccumulator()

    def test_process_chunk_new_tool_call(self, accumulator):
        """Test processing a chunk with a new tool call"""
        tool_calls_data = [
            ChoiceDeltaToolCall(
                index=0,
                id="call_123",
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name="get_weather", arguments='{"location":'
                ),
            )
        ]

        completed_tools = accumulator.process_chunk(tool_calls_data)

        assert len(completed_tools) == 0  # Not complete yet
        assert 0 in accumulator.tool_calls
        tool_call = accumulator.tool_calls[0]
        assert tool_call.id == "call_123"
        assert tool_call.function_name == "get_weather"
        assert tool_call.arguments == '{"location":'

    def test_process_chunk_continue_tool_call(self, accumulator):
        """Test processing a chunk that continues an existing tool call"""
        # First chunk
        tool_calls_data_1 = [
            ChoiceDeltaToolCall(
                index=0,
                id="call_123",
                function=ChoiceDeltaToolCallFunction(
                    name="get_weather", arguments='{"location":'
                ),
            )
        ]
        accumulator.process_chunk(tool_calls_data_1)

        # Second chunk continues arguments
        tool_calls_data_2 = [
            ChoiceDeltaToolCall(
                index=0,
                function=ChoiceDeltaToolCallFunction(arguments=' "San Francisco"}'),
            )
        ]

        completed_tools = accumulator.process_chunk(tool_calls_data_2)

        assert len(completed_tools) == 0  # Still not marked complete
        tool_call = accumulator.tool_calls[0]
        assert tool_call.arguments == '{"location": "San Francisco"}'

    def test_process_chunk_multiple_tool_calls(self, accumulator):
        """Test processing a chunk with multiple tool calls"""
        tool_calls_data = [
            ChoiceDeltaToolCall(
                index=0,
                id="call_123",
                function=ChoiceDeltaToolCallFunction(
                    name="get_weather", arguments='{"location": "SF"}'
                ),
            ),
            ChoiceDeltaToolCall(
                index=1,
                id="call_456",
                function=ChoiceDeltaToolCallFunction(name="get_time", arguments="{}"),
            ),
        ]

        completed_tools = accumulator.process_chunk(tool_calls_data)

        assert len(completed_tools) == 0
        assert len(accumulator.tool_calls) == 2
        assert accumulator.tool_calls[0].function_name == "get_weather"
        assert accumulator.tool_calls[1].function_name == "get_time"

    def test_process_chunk_with_defaults(self, accumulator):
        """Test processing chunk data with missing fields"""
        tool_calls_data = [
            ChoiceDeltaToolCall(
                index=0,  # Index is required for ChoiceDeltaToolCall
                function=ChoiceDeltaToolCallFunction(name="simple_function"),
            )
        ]

        accumulator.process_chunk(tool_calls_data)

        assert 0 in accumulator.tool_calls
        tool_call = accumulator.tool_calls[0]
        assert tool_call.index == 0
        assert tool_call.function_name == "simple_function"

    def test_finalize_tools_valid_complete_tools(self, accumulator):
        """Test finalizing tools with valid complete tool calls"""
        # Setup tool calls
        accumulator.tool_calls[0] = ToolCall(
            index=0,
            id="call_123",
            function_name="get_weather",
            arguments='{"location": "San Francisco"}',
        )
        accumulator.tool_calls[1] = ToolCall(
            index=1, id="call_456", function_name="get_time", arguments="{}"
        )

        completed_tools = accumulator.finalize_tools()

        assert len(completed_tools) == 2
        assert all(tool.is_complete for tool in completed_tools)
        assert completed_tools[0].function_name == "get_weather"
        assert completed_tools[1].function_name == "get_time"

    def test_finalize_tools_invalid_arguments(self, accumulator):
        """Test finalizing tools with invalid JSON arguments"""
        accumulator.tool_calls[0] = ToolCall(
            index=0,
            function_name="get_weather",
            arguments='{"location": invalid_json',  # Invalid JSON
        )

        completed_tools = accumulator.finalize_tools()

        assert len(completed_tools) == 0  # Should be excluded

    def test_finalize_tools_missing_function_name(self, accumulator):
        """Test finalizing tools without function names"""
        accumulator.tool_calls[0] = ToolCall(
            index=0,
            arguments='{"param": "value"}',
            # Missing function_name
        )

        completed_tools = accumulator.finalize_tools()

        assert len(completed_tools) == 0  # Should be excluded

    def test_is_valid_arguments_valid_json(self, accumulator):
        """Test _is_valid_arguments with valid JSON"""
        assert accumulator._is_valid_arguments('{"key": "value"}') is True
        assert accumulator._is_valid_arguments("{}") is True
        assert accumulator._is_valid_arguments("[]") is True
        assert accumulator._is_valid_arguments('"string"') is True

    def test_is_valid_arguments_empty_string(self, accumulator):
        """Test _is_valid_arguments with empty string (should be valid)"""
        assert accumulator._is_valid_arguments("") is True

    def test_is_valid_arguments_invalid_json(self, accumulator):
        """Test _is_valid_arguments with invalid JSON"""
        assert accumulator._is_valid_arguments('{"invalid": json}') is False
        assert accumulator._is_valid_arguments('{"unclosed":') is False
        assert accumulator._is_valid_arguments("invalid") is False


class TestToolExecutor:
    @pytest.fixture
    def executor(self):
        """Use the singleton tool executor"""
        return tool_executor

    @pytest.fixture
    def complete_tool_call(self):
        """Create a complete tool call for testing"""
        return ToolCall(
            index=0,
            id="call_123",
            function_name="test_function",
            arguments='{"param": "value"}',
            is_complete=True,
        )

    @pytest.fixture
    def incomplete_tool_call(self):
        """Create an incomplete tool call for testing"""
        return ToolCall(
            index=0,
            function_name="test_function",
            arguments='{"param": "value"}',
            is_complete=False,
        )

    @pytest.mark.asyncio
    async def test_execute_tool_call_incomplete(self, executor, incomplete_tool_call):
        """Test executing an incomplete tool call"""
        result = await executor.execute_tool_call(incomplete_tool_call)

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_tool_call_missing_function_name(self, executor):
        """Test executing a tool call without function name"""
        tool_call = ToolCall(
            index=0,
            id="call_123",
            arguments='{"param": "value"}',
            is_complete=True,
            # Missing function_name
        )

        result = await executor.execute_tool_call(tool_call)

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_tool_call_non_mcp_without_executor(
        self, executor, complete_tool_call
    ):
        """Non-MCP tools without an MCP executor are not executed server-side."""
        result = await executor.execute_tool_call(complete_tool_call)

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_tool_calls_multiple(self, executor):
        """Test executing multiple tool calls"""
        tool_calls = [
            ToolCall(
                index=0,
                id="call_123",
                function_name="unsupported_function",
                arguments='{"param1": "value1"}',
                is_complete=True,
            ),
            ToolCall(
                index=1,
                id="call_456",
                function_name="another_unsupported",
                arguments='{"param2": "value2"}',
                is_complete=True,
            ),
        ]

        results = await executor.execute_tool_calls(tool_calls)

        assert len(results) == 2
        assert results["call_123"] is None
        assert results["call_456"] is None

    @pytest.mark.asyncio
    async def test_execute_tool_calls_mixed_results(self, executor):
        """Test executing tool calls with mixed server/client results"""
        tool_calls = [
            ToolCall(
                index=0,
                id="call_123",
                function_name="unsupported_function",
                arguments='{"param": "value"}',
                is_complete=True,
            )
        ]

        # Mock one as supported (returns list) and one as unsupported (returns OpenAIToolCall)
        with patch.object(executor, "execute_tool_call") as mock_execute:
            # First call returns server result (list), second returns client forward (OpenAIToolCall)
            mock_execute.side_effect = [
                [{"server_result": "data"}],
                OpenAIToolCall(
                    id="call_456",
                    type="function",
                    function=ToolCallFunction(name="client_func", arguments="{}"),
                ),
            ]

            # Add a second tool call
            tool_calls.append(
                ToolCall(
                    index=1,
                    id="call_456",
                    function_name="client_function",
                    arguments="{}",
                    is_complete=True,
                )
            )

            results = await executor.execute_tool_calls(tool_calls)

            assert len(results) == 2
            assert "call_123" in results
            assert "call_456" in results
            assert results["call_123"] == [{"server_result": "data"}]
            assert isinstance(results["call_456"], OpenAIToolCall)

    @pytest.mark.asyncio
    async def test_execute_tool_calls_with_none_results(self, executor):
        """Test executing tool calls where some return None"""
        tool_calls = [
            ToolCall(index=0, is_complete=False),  # Should return None
            ToolCall(
                index=1,
                id="call_123",
                function_name="test_function",
                arguments="{}",
                is_complete=True,
            ),
        ]

        results = await executor.execute_tool_calls(tool_calls)

        assert len(results) == 2
        assert results["tool_call_0"] is None
        assert results["call_123"] is None


class TestParseStreamingToolCalls:
    @pytest.fixture
    def accumulator(self):
        """Create a fresh accumulator for each test"""
        return ToolCallAccumulator()

    def test_parse_streaming_tool_calls_with_data(self, accumulator):
        """Test parsing streaming tool calls with tool call data"""
        tool_calls = [
            ChoiceDeltaToolCall(
                index=0,
                id="call_123",
                function=ChoiceDeltaToolCallFunction(
                    name="get_weather",
                    arguments='{"location": "SF"}',
                ),
            )
        ]

        completed_tools = parse_streaming_tool_calls(accumulator, tool_calls, None)

        assert len(completed_tools) == 0  # Not finalized yet
        assert 0 in accumulator.tool_calls

    def test_parse_streaming_tool_calls_with_stop_reason(self, accumulator):
        """Test parsing with tool_calls stop reason"""
        # First add some tool call data
        accumulator.tool_calls[0] = ToolCall(
            index=0,
            id="call_123",
            function_name="get_weather",
            arguments='{"location": "SF"}',
        )

        completed_tools = parse_streaming_tool_calls(accumulator, [], "tool_calls")

        assert len(completed_tools) == 1
        assert completed_tools[0].is_complete is True

    def test_parse_streaming_tool_calls_final_chunk_with_data(self, accumulator):
        """Test parsing final chunk that has both data and stop reason"""
        tool_calls = [
            ChoiceDeltaToolCall(
                index=0,
                id="call_123",
                function=ChoiceDeltaToolCallFunction(
                    name="get_weather",
                    arguments='{"location": "SF"}',
                ),
            )
        ]

        completed_tools = parse_streaming_tool_calls(
            accumulator, tool_calls, "tool_calls"
        )

        assert len(completed_tools) == 1
        assert completed_tools[0].is_complete is True
        assert completed_tools[0].function_name == "get_weather"

    def test_parse_streaming_tool_calls_no_data_no_stop(self, accumulator):
        """Test parsing chunk with no tool calls and no stop reason"""
        completed_tools = parse_streaming_tool_calls(accumulator, None, None)

        assert len(completed_tools) == 0

    def test_parse_streaming_tool_calls_empty_tool_calls(self, accumulator):
        """Test parsing chunk with empty tool_calls list"""
        completed_tools = parse_streaming_tool_calls(accumulator, [], None)

        assert len(completed_tools) == 0

    def test_parse_streaming_tool_calls_stop_reason_only(self, accumulator):
        """Test parsing chunk with only stop_reason (no tool_calls key)"""
        # Setup existing tool call
        accumulator.tool_calls[0] = ToolCall(
            index=0,
            id="call_123",
            function_name="get_weather",
            arguments='{"location": "SF"}',
        )

        completed_tools = parse_streaming_tool_calls(accumulator, None, "tool_calls")

        assert len(completed_tools) == 1
        assert completed_tools[0].is_complete is True
