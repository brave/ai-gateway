import copy
from unittest.mock import patch

import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from aichat.protocol.open_ai_protocol import Request as OpenAIRequest
from aichat.protocol.open_ai_protocol import UserMessage
from aichat.serve.open_ai_api import (
    convert_litellm_response_to_openai_chat_completion,
    create_content_chunk,
    process_streaming_response,
)


class TestProcessStreamingResponse:
    @pytest.fixture
    def openai_request(self):
        """Mock OpenAI request for streaming tests"""
        return OpenAIRequest(
            model="test-model",
            messages=[UserMessage(content="Hello")],
            stream=True,
        )

    @pytest.fixture
    def mock_chunk_with_content(self):
        """Mock streaming chunk with content"""
        chunk = ChatCompletionChunk(
            id="chunk-123",
            choices=[
                Choice(
                    delta=ChoiceDelta(content="Hello there!", role="assistant"),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        return chunk

    @pytest.fixture
    def mock_chunk_with_finish_reason(self):
        """Mock streaming chunk with finish reason"""
        chunk = ChatCompletionChunk(
            id="chunk-124",
            choices=[
                Choice(
                    delta=ChoiceDelta(content=None, role="assistant"),
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        return chunk

    @pytest.fixture
    def mock_chunk_with_tool_calls(self):
        """Mock streaming chunk with tool calls"""
        tool_call = ChoiceDeltaToolCall(
            index=0,
            id="call_123",
            function=ChoiceDeltaToolCallFunction(
                name="get_weather", arguments='{"location":"SF"}'
            ),
            type="function",
        )
        chunk = ChatCompletionChunk(
            id="chunk-125",
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=None, role="assistant", tool_calls=[tool_call]
                    ),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        return chunk

    @pytest.fixture
    def mock_chunk_empty_choices(self):
        """Mock streaming chunk with empty choices"""
        chunk = ChatCompletionChunk(
            id="chunk-126",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        return chunk

    @pytest.fixture
    def mock_chunk_none_choice(self):
        """Mock streaming chunk with None choice (edge case - empty list instead)"""
        chunk = ChatCompletionChunk(
            id="chunk-127",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        return chunk

    @pytest.mark.asyncio
    async def test_process_streaming_response_with_content(
        self, openai_request, mock_chunk_with_content
    ):
        """Test processing streaming response with content"""

        async def mock_response():
            yield mock_chunk_with_content

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        # Should have content chunk + done marker
        assert len(result_chunks) == 2

        # First chunk should be SSE formatted
        assert result_chunks[0].startswith("data: ")
        assert result_chunks[0].endswith("\n\n")
        assert "Hello there!" in result_chunks[0]
        assert "test-model" in result_chunks[0]
        assert "chunk-123" in result_chunks[0]

        # Last chunk should be done marker
        assert result_chunks[1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_process_streaming_response_with_finish_reason(
        self, openai_request, mock_chunk_with_finish_reason
    ):
        """Test processing streaming response with finish reason"""

        async def mock_response():
            yield mock_chunk_with_finish_reason

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        assert len(result_chunks) == 2
        assert "stop" in result_chunks[0]

    @pytest.mark.asyncio
    @patch(
        "aichat.serve.open_ai_api.security_settings.alignment_checking_enabled", False
    )
    async def test_process_streaming_response_with_tool_calls(
        self, openai_request, mock_chunk_with_tool_calls
    ):
        """Test processing streaming response with tool calls"""

        async def mock_response():
            yield mock_chunk_with_tool_calls

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        assert len(result_chunks) == 2
        assert "get_weather" in result_chunks[0]

    @pytest.mark.asyncio
    async def test_process_streaming_response_empty_choices(
        self, openai_request, mock_chunk_empty_choices
    ):
        """Test processing streaming response with empty choices"""

        async def mock_response():
            yield mock_chunk_empty_choices

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        # Should only have done marker, no content chunk
        assert len(result_chunks) == 1
        assert result_chunks[0] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_process_streaming_response_none_choice(
        self, openai_request, mock_chunk_none_choice
    ):
        """Test processing streaming response with None choice"""

        async def mock_response():
            yield mock_chunk_none_choice

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        # Should only have done marker, no content chunk
        assert len(result_chunks) == 1
        assert result_chunks[0] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_process_streaming_response_multiple_chunks(
        self, openai_request, mock_chunk_with_content, mock_chunk_with_finish_reason
    ):
        """Test processing multiple streaming chunks"""

        async def mock_response():
            yield mock_chunk_with_content
            yield mock_chunk_with_finish_reason

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        # Should have 2 content chunks + done marker
        assert len(result_chunks) == 3
        assert "Hello there!" in result_chunks[0]
        assert "stop" in result_chunks[1]
        assert result_chunks[2] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_process_streaming_response_response_serialization(
        self, openai_request, mock_chunk_with_content
    ):
        """Test that Response objects are properly serialized"""

        async def mock_response():
            yield mock_chunk_with_content

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            if chunk != "data: [DONE]\n\n":
                result_chunks.append(chunk)

        # Parse the JSON to verify structure
        import json

        chunk_data = result_chunks[0]
        json_str = chunk_data.replace("data: ", "").strip()
        parsed = json.loads(json_str)

        # Verify OpenAI ChatCompletionChunk format structure
        assert parsed["model"] == "test-model"
        assert parsed["id"] == "chunk-123"
        assert parsed["object"] == "chat.completion.chunk"
        assert parsed["created"] == 1234567890
        assert "choices" in parsed
        assert len(parsed["choices"]) == 1
        assert parsed["choices"][0]["index"] == 0
        assert parsed["choices"][0]["delta"]["content"] == "Hello there!"
        assert parsed["choices"][0]["delta"]["role"] == "assistant"
        # These fields should be excluded when None due to exclude_none=True
        assert "finish_reason" not in parsed["choices"][0]  # None values are excluded
        assert "logprobs" not in parsed["choices"][0]  # None values are excluded

    @pytest.mark.asyncio
    async def test_process_streaming_response_exclude_none_values(
        self, openai_request, mock_chunk_with_content
    ):
        """Test that None values are excluded from serialization"""

        async def mock_response():
            yield mock_chunk_with_content

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            if chunk != "data: [DONE]\n\n":
                result_chunks.append(chunk)

        # Verify None values are excluded (they shouldn't appear in JSON at all)
        chunk_data = result_chunks[0]
        # Parse the JSON to check exclusion properly
        import json

        json_str = chunk_data.replace("data: ", "").strip()
        parsed = json.loads(json_str)

        # These fields should not appear in the JSON when they're None due to exclude_none=True
        assert "stop_sequence" not in parsed
        assert "tool_calls" not in parsed

    @pytest.mark.asyncio
    async def test_process_streaming_response_empty_stream(self, openai_request):
        """Test processing empty streaming response"""

        async def mock_response():
            return
            yield  # Never reached

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        # Should only have done marker
        assert len(result_chunks) == 1
        assert result_chunks[0] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_process_streaming_response_mid_stream_error(
        self, openai_request, mock_chunk_with_content
    ):
        """A mid-stream exception should surface as an error chunk + [DONE]
        instead of propagating and aborting the connection."""
        import json

        class MidStreamFallbackError(Exception):
            pass

        async def mock_response():
            yield mock_chunk_with_content
            raise MidStreamFallbackError("No fallback model group found")

        result_chunks = []
        # Must not raise even though the underlying stream errors mid-way.
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        # content chunk, then error chunk, then done marker
        assert len(result_chunks) == 3
        assert "Hello there!" in result_chunks[0]

        error_json = json.loads(result_chunks[1].replace("data: ", "").strip())
        assert error_json["object"] == "chat.completion.chunk"
        assert error_json["choices"][0]["finish_reason"] == "stop"
        assert (
            error_json["choices"][0]["delta"]["content"]
            == "There was an issue connecting to the model."
        )

        assert result_chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_process_streaming_response_mid_stream_error_first_chunk(
        self, openai_request
    ):
        """A mid-stream error raised before any chunk is emitted should still
        produce an error chunk followed by [DONE]."""
        import json

        class MidStreamFallbackError(Exception):
            pass

        async def mock_response():
            raise MidStreamFallbackError("No fallback model group found")
            yield  # Never reached

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        assert len(result_chunks) == 2
        error_json = json.loads(result_chunks[0].replace("data: ", "").strip())
        assert (
            error_json["choices"][0]["delta"]["content"]
            == "There was an issue connecting to the model."
        )
        assert result_chunks[1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_process_streaming_response_non_midstream_error_no_error_chunk(
        self, openai_request, mock_chunk_with_content
    ):
        """Non mid-stream exceptions terminate the stream cleanly with [DONE]
        but must NOT emit the generic error message chunk (only mid-stream
        errors get a user-facing error message)."""

        async def mock_response():
            yield mock_chunk_with_content
            raise RuntimeError("boom")

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        # content chunk + done marker, no error message chunk in between
        assert len(result_chunks) == 2
        assert "Hello there!" in result_chunks[0]
        assert result_chunks[-1] == "data: [DONE]\n\n"
        assert "There was an issue connecting to the model." not in "".join(
            result_chunks
        )


# Additional fixtures for testing the new functions moved from litellm
@pytest.fixture
def mock_chunk():
    """Create a mock ChatCompletionChunk for testing"""
    return ChatCompletionChunk(
        id="chat-test-123",
        choices=[
            Choice(
                delta=ChoiceDelta(
                    content="Hello world",
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                    refusal=None,
                ),
                finish_reason=None,
                index=0,
                logprobs=None,
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
        system_fingerprint=None,
        service_tier=None,
        usage=None,
    )


@pytest.fixture
def mock_chunk_with_tool_calls():
    """Create a mock ChatCompletionChunk with tool calls for testing"""
    tool_call = ChoiceDeltaToolCall(
        index=0,
        id="call_123",
        function=ChoiceDeltaToolCallFunction(arguments="{}", name="test_function"),
        type="function",
    )

    return ChatCompletionChunk(
        id="chat-test-123",
        choices=[
            Choice(
                delta=ChoiceDelta(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=[tool_call],
                    refusal=None,
                ),
                finish_reason=None,
                index=0,
                logprobs=None,
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
        system_fingerprint=None,
        service_tier=None,
        usage=None,
    )


@pytest.fixture
def mock_chunk_with_reasoning():
    """Create a mock ChatCompletionChunk with reasoning content for testing"""
    chunk = ChatCompletionChunk(
        id="chat-test-123",
        choices=[
            Choice(
                delta=ChoiceDelta(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                    refusal=None,
                ),
                finish_reason=None,
                index=0,
                logprobs=None,
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
        system_fingerprint=None,
        service_tier=None,
        usage=None,
    )

    # Add reasoning_content attribute to delta
    chunk.choices[0].delta.reasoning_content = "This is reasoning content"
    return chunk


async def mock_generator_simple():
    """Simple mock generator for testing"""
    chunk = ChatCompletionChunk(
        id="chat-test-123",
        choices=[
            Choice(
                delta=ChoiceDelta(content="Hello", role="assistant"),
                finish_reason=None,
                index=0,
                logprobs=None,
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
    )
    yield chunk


async def mock_generator_with_usage():
    """Mock generator with usage information for testing"""
    chunk = ChatCompletionChunk(
        id="chat-test-123",
        choices=[
            Choice(
                delta=ChoiceDelta(content="Hello", role="assistant"),
                finish_reason=None,
                index=0,
                logprobs=None,
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
    )

    # Add usage attribute
    chunk.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    yield chunk


async def mock_generator_no_choices():
    """Mock generator with no choices for testing"""
    chunk = ChatCompletionChunk(
        id="chat-test-123",
        choices=[],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
    )
    yield chunk


async def mock_generator_with_tool_calls():
    """Mock generator with tool calls for testing"""
    tool_call = ChoiceDeltaToolCall(
        index=0,
        id="call_123",
        function=ChoiceDeltaToolCallFunction(arguments="{}", name="test_function"),
        type="function",
    )

    chunk = ChatCompletionChunk(
        id="chat-test-123",
        choices=[
            Choice(
                delta=ChoiceDelta(
                    content=None, role="assistant", tool_calls=[tool_call]
                ),
                finish_reason=None,
                index=0,
                logprobs=None,
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
    )
    yield chunk


class TestConvertLitellmResponse:
    @pytest.mark.asyncio
    async def test_convert_simple_stream(self):
        """Test converting a simple stream"""
        results = []
        async for chunk in convert_litellm_response_to_openai_chat_completion(
            mock_generator_simple(), "test-model"
        ):
            results.append(chunk)

        assert len(results) == 1
        assert results[0].choices[0].delta.content == "Hello"

    @pytest.mark.asyncio
    async def test_convert_stream_with_usage_no_debug(self):
        """Test converting stream with usage passes through correctly"""
        results = []
        async for chunk in convert_litellm_response_to_openai_chat_completion(
            mock_generator_with_usage()
        ):
            results.append(chunk)

        with patch("aichat.serve.open_ai_api.logger") as mock_logger:
            results = []
            async for chunk in convert_litellm_response_to_openai_chat_completion(
                mock_generator_with_usage(), "test-model"
            ):
                results.append(chunk)

            assert len(results) == 1
            mock_logger.debug.assert_not_called()

    @pytest.mark.asyncio
    async def test_convert_stream_no_choices(self):
        """Test converting stream with no choices"""
        results = []
        async for chunk in convert_litellm_response_to_openai_chat_completion(
            mock_generator_no_choices(), "test-model"
        ):
            results.append(chunk)

        assert len(results) == 1
        assert len(results[0].choices) == 0

    @pytest.mark.asyncio
    async def test_convert_stream_with_tool_calls(self):
        """Test converting stream with tool calls"""
        results = []
        async for chunk in convert_litellm_response_to_openai_chat_completion(
            mock_generator_with_tool_calls(), "test-model"
        ):
            results.append(chunk)

        assert len(results) == 1
        tool_call = results[0].choices[0].delta.tool_calls[0]
        assert tool_call.function.arguments == ""  # Should be converted from "{}"

    @pytest.mark.asyncio
    async def test_tool_call_arguments_modification(self):
        """Test that tool call arguments are properly modified"""

        # Create a generator that yields a chunk with "{}" arguments
        async def mock_tool_generator():
            tool_call = ChoiceDeltaToolCall(
                index=0,
                id="call_123",
                function=ChoiceDeltaToolCallFunction(
                    arguments="{}", name="test_function"
                ),
                type="function",
            )

            chunk = ChatCompletionChunk(
                id="chat-test-123",
                choices=[
                    Choice(
                        delta=ChoiceDelta(
                            content=None, role="assistant", tool_calls=[tool_call]
                        ),
                        finish_reason=None,
                        index=0,
                        logprobs=None,
                    )
                ],
                created=1234567890,
                model="test-model",
                object="chat.completion.chunk",
            )
            yield chunk

        results = []
        async for chunk in convert_litellm_response_to_openai_chat_completion(
            mock_tool_generator(), "test-model"
        ):
            results.append(chunk)

        assert len(results) == 1
        tool_call = results[0].choices[0].delta.tool_calls[0]
        assert tool_call.function.arguments == ""

    @pytest.mark.asyncio
    async def test_reasoning_block_first_chunk(self):
        """Test that the first reasoning chunk gets <think> tag"""

        async def reasoning_generator():
            chunk = ChatCompletionChunk(
                id="chat-test-123",
                choices=[
                    Choice(
                        delta=ChoiceDelta(content=None, role="assistant"),
                        finish_reason=None,
                        index=0,
                        logprobs=None,
                    )
                ],
                created=1234567890,
                model="test-model",
                object="chat.completion.chunk",
            )
            # Add reasoning content
            chunk.choices[0].delta.reasoning_content = "I need to think about this"
            yield chunk

        results = []
        async for chunk in convert_litellm_response_to_openai_chat_completion(
            reasoning_generator(), "test-model"
        ):
            results.append(chunk)

        assert len(results) == 1
        assert (
            results[0].choices[0].delta.content == "<think>I need to think about this"
        )
        assert results[0].choices[0].delta.reasoning_content == ""

    @pytest.mark.asyncio
    async def test_reasoning_block_subsequent_chunks(self):
        """Test that subsequent reasoning chunks don't get additional <think> tags"""

        async def reasoning_generator():
            # First reasoning chunk
            chunk1 = ChatCompletionChunk(
                id="chat-test-123",
                choices=[
                    Choice(
                        delta=ChoiceDelta(content=None, role="assistant"),
                        finish_reason=None,
                        index=0,
                        logprobs=None,
                    )
                ],
                created=1234567890,
                model="test-model",
                object="chat.completion.chunk",
            )
            chunk1.choices[0].delta.reasoning_content = "First reasoning"
            yield chunk1

            # Second reasoning chunk
            chunk2 = copy.deepcopy(chunk1)
            chunk2.choices[0].delta.reasoning_content = "Second reasoning"
            yield chunk2

        results = []
        async for chunk in convert_litellm_response_to_openai_chat_completion(
            reasoning_generator(), "test-model"
        ):
            results.append(chunk)

        assert len(results) == 2
        assert results[0].choices[0].delta.content == "<think>First reasoning"
        assert results[1].choices[0].delta.content == "Second reasoning"

    @pytest.mark.asyncio
    async def test_reasoning_to_content_transition(self):
        """Test that transitioning from reasoning to content adds </think> tag"""

        async def mixed_generator():
            # Reasoning chunk
            reasoning_chunk = ChatCompletionChunk(
                id="chat-test-123",
                choices=[
                    Choice(
                        delta=ChoiceDelta(content=None, role="assistant"),
                        finish_reason=None,
                        index=0,
                        logprobs=None,
                    )
                ],
                created=1234567890,
                model="test-model",
                object="chat.completion.chunk",
            )
            reasoning_chunk.choices[0].delta.reasoning_content = "Let me think..."
            yield reasoning_chunk

            # Regular content chunk
            content_chunk = ChatCompletionChunk(
                id="chat-test-123",
                choices=[
                    Choice(
                        delta=ChoiceDelta(content="Hello!", role="assistant"),
                        finish_reason=None,
                        index=0,
                        logprobs=None,
                    )
                ],
                created=1234567890,
                model="test-model",
                object="chat.completion.chunk",
            )
            yield content_chunk

        results = []
        async for chunk in convert_litellm_response_to_openai_chat_completion(
            mixed_generator(), "test-model"
        ):
            results.append(chunk)

        assert len(results) == 2
        assert results[0].choices[0].delta.content == "<think>Let me think..."
        assert results[1].choices[0].delta.content == "</think>Hello!"

    @pytest.mark.asyncio
    async def test_multiple_reasoning_content_transitions(self):
        """Test multiple transitions between reasoning and content"""

        async def complex_generator():
            # First reasoning chunk
            reasoning1 = ChatCompletionChunk(
                id="chat-test-123",
                choices=[
                    Choice(
                        delta=ChoiceDelta(content=None, role="assistant"),
                        finish_reason=None,
                        index=0,
                        logprobs=None,
                    )
                ],
                created=1234567890,
                model="test-model",
                object="chat.completion.chunk",
            )
            reasoning1.choices[0].delta.reasoning_content = "First thought"
            yield reasoning1

            # First content chunk
            content1 = ChatCompletionChunk(
                id="chat-test-123",
                choices=[
                    Choice(
                        delta=ChoiceDelta(content="First response", role="assistant"),
                        finish_reason=None,
                        index=0,
                        logprobs=None,
                    )
                ],
                created=1234567890,
                model="test-model",
                object="chat.completion.chunk",
            )
            yield content1

            # Second reasoning chunk
            reasoning2 = copy.deepcopy(reasoning1)
            reasoning2.choices[0].delta.reasoning_content = "Second thought"
            yield reasoning2

            # Second content chunk
            content2 = copy.deepcopy(content1)
            content2.choices[0].delta.content = "Second response"
            yield content2

        results = []
        async for chunk in convert_litellm_response_to_openai_chat_completion(
            complex_generator(), "test-model"
        ):
            results.append(chunk)

        assert len(results) == 4
        assert results[0].choices[0].delta.content == "<think>First thought"
        assert results[1].choices[0].delta.content == "</think>First response"
        assert results[2].choices[0].delta.content == "<think>Second thought"
        assert results[3].choices[0].delta.content == "</think>Second response"

    @pytest.mark.asyncio
    async def test_content_without_reasoning_unchanged(self):
        """Test that regular content without reasoning is unchanged"""

        async def content_generator():
            chunk = ChatCompletionChunk(
                id="chat-test-123",
                choices=[
                    Choice(
                        delta=ChoiceDelta(content="Regular content", role="assistant"),
                        finish_reason=None,
                        index=0,
                        logprobs=None,
                    )
                ],
                created=1234567890,
                model="test-model",
                object="chat.completion.chunk",
            )
            yield chunk

        results = []
        async for chunk in convert_litellm_response_to_openai_chat_completion(
            content_generator(), "test-model"
        ):
            results.append(chunk)

        assert len(results) == 1
        assert results[0].choices[0].delta.content == "Regular content"


class TestCreateContentChunk:
    def test_create_content_chunk_success(self, mock_chunk):
        """Test creating a content chunk successfully"""
        new_content = "New content"
        result = create_content_chunk(mock_chunk, new_content)

        assert result is not None
        assert result.choices[0].delta.content == new_content
        assert result.id == mock_chunk.id
        assert result.model == mock_chunk.model

        # The function now modifies the original chunk directly
        assert mock_chunk.choices[0].delta.content == new_content
        assert result is mock_chunk  # Same object reference

    def test_create_content_chunk_no_choices(self):
        """Test creating a content chunk with no choices"""
        chunk_no_choices = ChatCompletionChunk(
            id="chat-test-123",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )

        result = create_content_chunk(chunk_no_choices, "test content")
        assert result is None

    def test_create_content_chunk_with_reasoning(self, mock_chunk_with_reasoning):
        """Test creating a content chunk that clears reasoning content"""
        new_content = "Regular content"
        result = create_content_chunk(mock_chunk_with_reasoning, new_content)

        assert result is not None
        assert result.choices[0].delta.content == new_content
        assert result.choices[0].delta.reasoning_content == ""

    def test_create_content_chunk_modifies_original(self, mock_chunk):
        """Test that create_content_chunk modifies the original chunk directly"""
        new_content = "Modified content"
        result = create_content_chunk(mock_chunk, new_content)

        # The function now modifies the original chunk directly
        assert result is mock_chunk  # Same object reference
        assert mock_chunk.choices[0].delta.content == new_content
        assert result.choices[0].delta.content == new_content

        # Further modify the result - since it's the same object, original is affected
        result.choices[0].delta.content = "Further modified"
        assert mock_chunk.choices[0].delta.content == "Further modified"
