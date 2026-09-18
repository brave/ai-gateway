import copy
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pydantic import BaseModel

from aichat.protocol.open_ai_protocol import (
    PageTextContentPart,
    TextContentPart,
    Tool,
    ToolFunction,
    ToolMessage,
    UserMessage,
)
from aichat.protocol.open_ai_protocol import (
    Request as OpenAIRequest,
)
from aichat.responses import (
    ContentReceipt,
    InlineSearch,
    InlineSearchResult,
    InlineSearchResultMeta,
    InlineSearchResultThumbnail,
    SearchQueries,
    WebSource,
    WebSources,
)
from aichat.serve.open_ai_adapter import process_non_streaming_alignment_check
from aichat.serve.open_ai_api import (
    convert_litellm_response_to_openai_chat_completion,
    create_content_chunk,
    create_openai_response,
    process_streaming_response,
    v1_chat_completions,
)
from aichat.serve.tool_parser import AlignmentCheckData


class TestV1ChatCompletions:
    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI Request object"""
        request = MagicMock(spec=Request)
        request.url = "http://test.com/v1/chat/completions"
        request.headers = {"authorization": "Bearer test-token"}
        return request

    @pytest.fixture
    def mock_background_tasks(self):
        """Mock FastAPI BackgroundTasks object"""
        return MagicMock(spec=BackgroundTasks)

    @pytest.fixture
    def mock_fastapi_response(self):
        """Mock FastAPI Response object"""
        from fastapi import Response

        response = MagicMock(spec=Response)
        response.headers = {}
        return response

    @pytest.fixture
    def openai_request(self):
        """Mock OpenAI request object"""
        return OpenAIRequest(
            model="test-model",
            messages=[UserMessage(content="Hello, how are you?")],
            stream=False,
            tools=None,
        )

    @pytest.fixture
    def openai_request_streaming(self):
        """Mock OpenAI request object with streaming enabled"""
        return OpenAIRequest(
            model="test-model",
            messages=[UserMessage(content="Hello, how are you?")],
            stream=True,
            tools=None,
        )

    @pytest.fixture
    def openai_request_with_tools(self):
        """Mock OpenAI request object with tools"""
        return OpenAIRequest(
            model="test-model",
            messages=[UserMessage(content="What's the weather like?")],
            stream=False,
            tools=[
                Tool(
                    type="function",
                    function=ToolFunction(
                        name="get_weather",
                        description="Get current weather",
                        parameters={"type": "object", "properties": {}},
                    ),
                )
            ],
        )

    @pytest.fixture
    def mock_common_params(self):
        """Mock common parameters"""
        return {
            "model": "test-model",
            "user_id": "test-user",
            "session_id": "test-session",
            "is_premium_host": False,
            "has_valid_premium_credential": False,
        }

    @pytest.fixture
    def mock_backend(self):
        """Mock backend object"""
        backend = AsyncMock()
        backend.build_params.return_value = {"temperature": 0.7, "max_tokens": 1000}
        backend.converse.return_value = {
            "id": "resp-123",
            "choices": [{"message": {"content": "Hello! I'm doing well, thank you."}}],
            "model": "test-model",
        }
        return backend

    @pytest.fixture
    def mock_model_config(self):
        """Mock model configuration"""
        config = MagicMock()
        config.model_id = "test-model"
        config.backend = "litellm"
        config.tool_support = True
        config.conversation_token_limit = None
        config.conversation_token_limit_premium = None
        return config

    @patch("aichat.serve.open_ai_api.mcp_settings")
    @patch("aichat.serve.open_ai_api.check_requests_common")
    @patch("aichat.serve.open_ai_api.get_backend")
    @patch("aichat.serve.open_ai_api.get_model_config")
    @patch("aichat.serve.open_ai_api.prompts")
    @pytest.mark.asyncio
    async def test_v1_chat_completions_non_streaming(
        self,
        mock_prompts,
        mock_get_model_config,
        mock_get_backend,
        mock_check_requests,
        mock_mcp_settings,
        mock_request,
        mock_background_tasks,
        mock_fastapi_response,
        openai_request,
        mock_common_params,
        mock_backend,
        mock_model_config,
    ):
        """Test non-streaming chat completions"""
        mock_check_requests.return_value = None
        mock_get_backend.return_value = mock_backend
        mock_get_model_config.return_value = mock_model_config
        mock_prompts.prompts = []
        mock_mcp_settings.mcp_enabled = False

        with patch(
            "aichat.serve.open_ai_api.select_model_for_request",
            return_value="test-model",
        ):
            result = await v1_chat_completions(
                raw_request=mock_request,
                request=openai_request,
                background_tasks=mock_background_tasks,
                fastapi_response=mock_fastapi_response,
                common=mock_common_params,
            )

        mock_check_requests.assert_called_once_with(mock_request, mock_common_params)
        mock_get_backend.assert_called_once_with("test-model")
        mock_get_model_config.assert_called_once_with("test-model")
        mock_backend.build_params.assert_called_once_with(
            False, [], [{"content": "Hello, how are you?", "role": "user"}]
        )
        mock_backend.converse.assert_called_once()

        assert result == mock_backend.converse.return_value

    @patch("aichat.serve.open_ai_api.mcp_settings")
    @patch("aichat.serve.open_ai_api.check_requests_common")
    @patch("aichat.serve.open_ai_api.get_backend")
    @patch("aichat.serve.open_ai_api.get_model_config")
    @patch("aichat.serve.open_ai_api.prompts")
    @pytest.mark.asyncio
    async def test_v1_chat_completions_streaming(
        self,
        mock_prompts,
        mock_get_model_config,
        mock_get_backend,
        mock_check_requests,
        mock_mcp_settings,
        mock_request,
        mock_background_tasks,
        mock_fastapi_response,
        openai_request_streaming,
        mock_common_params,
        mock_backend,
        mock_model_config,
    ):
        """Test streaming chat completions"""
        mock_check_requests.return_value = None
        mock_get_backend.return_value = mock_backend
        mock_get_model_config.return_value = mock_model_config
        mock_prompts.prompts = []
        mock_mcp_settings.mcp_enabled = False

        async def mock_stream():
            yield {"chunk": "data"}

        mock_backend.converse.return_value = mock_stream()

        with patch(
            "aichat.serve.open_ai_api.select_model_for_request",
            return_value="test-model",
        ):
            result = await v1_chat_completions(
                raw_request=mock_request,
                request=openai_request_streaming,
                background_tasks=mock_background_tasks,
                fastapi_response=mock_fastapi_response,
                common=mock_common_params,
            )

        assert isinstance(result, StreamingResponse)
        assert result.media_type == "text/event-stream"

        mock_backend.build_params.assert_called_once_with(
            True,
            [],
            [{"content": "Hello, how are you?", "role": "user"}],
        )

    @patch("aichat.serve.open_ai_api.check_requests_common")
    @pytest.mark.asyncio
    async def test_v1_chat_completions_request_error(
        self,
        mock_check_requests,
        mock_request,
        mock_background_tasks,
        mock_fastapi_response,
        openai_request,
        mock_common_params,
    ):
        """Test chat completions with request validation error"""
        error_response = {"error": "Invalid request"}
        mock_check_requests.return_value = error_response

        result = await v1_chat_completions(
            raw_request=mock_request,
            request=openai_request,
            background_tasks=mock_background_tasks,
            fastapi_response=mock_fastapi_response,
            common=mock_common_params,
        )

        assert result == error_response
        mock_check_requests.assert_called_once_with(mock_request, mock_common_params)

    @patch("aichat.serve.open_ai_api.mcp_settings")
    @patch("aichat.serve.open_ai_api.check_requests_common")
    @patch("aichat.serve.open_ai_api.get_backend")
    @patch("aichat.serve.open_ai_api.get_model_config")
    @patch("aichat.serve.open_ai_api.prompts")
    @pytest.mark.asyncio
    async def test_v1_chat_completions_with_tools(
        self,
        mock_prompts,
        mock_get_model_config,
        mock_get_backend,
        mock_check_requests,
        mock_mcp_settings,
        mock_request,
        mock_background_tasks,
        mock_fastapi_response,
        openai_request_with_tools,
        mock_common_params,
        mock_backend,
        mock_model_config,
    ):
        """Test chat completions with tools"""
        mock_check_requests.return_value = None
        mock_get_backend.return_value = mock_backend
        mock_get_model_config.return_value = mock_model_config
        mock_prompts.prompts = []
        mock_mcp_settings.mcp_enabled = False

        with patch(
            "aichat.serve.open_ai_api.select_model_for_request",
            return_value="test-model",
        ):
            await v1_chat_completions(
                raw_request=mock_request,
                request=openai_request_with_tools,
                background_tasks=mock_background_tasks,
                fastapi_response=mock_fastapi_response,
                common=mock_common_params,
            )

        mock_backend.build_params.assert_called_once_with(
            False,
            openai_request_with_tools.tools,
            [{"content": "What's the weather like?", "role": "user"}],
        )

    @patch("aichat.serve.open_ai_api.mcp_settings")
    @patch("aichat.serve.open_ai_api.check_requests_common")
    @patch("aichat.serve.open_ai_api.get_backend")
    @patch("aichat.serve.open_ai_api.get_model_config")
    @patch("aichat.serve.open_ai_api.prompts")
    @pytest.mark.asyncio
    async def test_v1_chat_completions_with_prompts(
        self,
        mock_prompts,
        mock_get_model_config,
        mock_get_backend,
        mock_check_requests,
        mock_mcp_settings,
        mock_request,
        mock_background_tasks,
        mock_fastapi_response,
        openai_request,
        mock_common_params,
        mock_backend,
        mock_model_config,
    ):
        """Test chat completions with prompt augmentation"""
        mock_check_requests.return_value = None
        mock_get_backend.return_value = mock_backend
        mock_get_model_config.return_value = mock_model_config
        mock_mcp_settings.mcp_enabled = False

        mock_prompt = MagicMock()
        mock_prompt.augment.return_value = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        mock_prompts.prompts = [mock_prompt]

        with patch(
            "aichat.serve.open_ai_api.select_model_for_request",
            return_value="test-model",
        ):
            await v1_chat_completions(
                raw_request=mock_request,
                request=openai_request,
                background_tasks=mock_background_tasks,
                fastapi_response=mock_fastapi_response,
                common=mock_common_params,
            )

        mock_prompt.augment.assert_called_once_with(
            [{"role": "user", "content": "Hello, how are you?"}],
            model_config=mock_model_config,
            tools=[],
            brave_capability=None,
        )

        call_args = mock_backend.converse.call_args[0]
        assert len(call_args[0]) == 2
        assert call_args[0][0]["role"] == "system"

    @patch("aichat.serve.open_ai_api.check_requests_common")
    @patch("aichat.serve.open_ai_api.get_backend")
    @patch("aichat.serve.open_ai_api.get_model_config")
    @patch("aichat.serve.open_ai_api.prompts")
    @pytest.mark.asyncio
    async def test_v1_chat_completions_message_serialization(
        self,
        mock_prompts,
        mock_get_model_config,
        mock_get_backend,
        mock_check_requests,
        mock_request,
        mock_background_tasks,
        mock_fastapi_response,
        openai_request,
        mock_common_params,
        mock_backend,
        mock_model_config,
    ):
        """Test that messages are properly serialized to dictionaries"""
        mock_check_requests.return_value = None
        mock_get_backend.return_value = mock_backend
        mock_get_model_config.return_value = mock_model_config
        mock_prompts.prompts = []

        with patch(
            "aichat.serve.open_ai_api.select_model_for_request",
            return_value="test-model",
        ):
            await v1_chat_completions(
                raw_request=mock_request,
                request=openai_request,
                background_tasks=mock_background_tasks,
                fastapi_response=mock_fastapi_response,
                common=mock_common_params,
            )

        call_args = mock_backend.converse.call_args[0]
        messages = call_args[0]
        assert isinstance(messages, list)
        assert isinstance(messages[0], dict)
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, how are you?"


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
        return ChatCompletionChunk(
            id="chunk-123",
            object="chat.completion.chunk",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="Hello there!", role="assistant"),
                    finish_reason=None,
                    logprobs=None,
                )
            ],
        )

    @pytest.fixture
    def mock_chunk_with_finish_reason(self):
        """Mock streaming chunk with finish reason"""
        return ChatCompletionChunk(
            id="chunk-124",
            object="chat.completion.chunk",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content=None, role="assistant"),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
        )

    @pytest.fixture
    def mock_chunk_with_tool_calls(self):
        """Mock streaming chunk with tool calls"""
        tool_call = ChoiceDeltaToolCall(
            index=0,
            id="call_123",
            function=ChoiceDeltaToolCallFunction(arguments="", name="get_weather"),
            type="function",
        )
        return ChatCompletionChunk(
            id="chunk-125",
            object="chat.completion.chunk",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content=None, role="assistant", tool_calls=[tool_call]
                    ),
                    finish_reason=None,
                    logprobs=None,
                )
            ],
        )

    @pytest.fixture
    def mock_chunk_empty_choices(self):
        """Mock streaming chunk with empty choices"""
        return ChatCompletionChunk(
            id="chunk-126",
            object="chat.completion.chunk",
            created=1234567890,
            model="test-model",
            choices=[],
        )

    @pytest.fixture
    def mock_chunk_none_choice(self):
        """Mock streaming chunk with None choice - not a valid OpenAI format, but testing edge case"""
        chunk = MagicMock()
        chunk.id = "chunk-127"
        chunk.choices = [None]
        chunk.usage = None
        chunk.model_dump = MagicMock(
            return_value={"id": "chunk-127", "choices": [None]}
        )
        chunk.model_dump_json = MagicMock(
            return_value='{"id":"chunk-127","choices":[null]}'
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

        assert len(result_chunks) == 2

        assert result_chunks[0].startswith("data: ")
        assert result_chunks[0].endswith("\n\n")
        assert "Hello there!" in result_chunks[0]
        assert "test-model" in result_chunks[0]
        assert "chunk-123" in result_chunks[0]

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

        assert len(result_chunks) == 1
        assert result_chunks[0] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_process_streaming_response_none_choice(
        self, openai_request, mock_chunk_none_choice
    ):
        """Test processing streaming response with None choice - edge case"""

        async def mock_response():
            yield mock_chunk_none_choice

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        # choices=[None] is a non-empty list, so it passes the if check and gets serialized
        assert len(result_chunks) == 2
        assert result_chunks[0] == 'data: {"id":"chunk-127","choices":[null]}\n\n'
        assert result_chunks[1] == "data: [DONE]\n\n"

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

        assert len(result_chunks) == 3
        assert "Hello there!" in result_chunks[0]
        assert "stop" in result_chunks[1]
        assert result_chunks[2] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_process_streaming_response_response_serialization(
        self, openai_request, mock_chunk_with_content
    ):
        """Test that ChatCompletionChunk objects are properly serialized to OpenAI format"""

        async def mock_response():
            yield mock_chunk_with_content

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            if chunk != "data: [DONE]\n\n":
                result_chunks.append(chunk)

        import json

        chunk_data = result_chunks[0]
        json_str = chunk_data.replace("data: ", "").strip()
        parsed = json.loads(json_str)

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
        assert "finish_reason" not in parsed["choices"][0]
        assert "logprobs" not in parsed["choices"][0]

    @pytest.mark.asyncio
    async def test_process_streaming_response_exclude_none_values(
        self, openai_request, mock_chunk_with_content
    ):
        """Test that None values are excluded from serialization in OpenAI format"""

        async def mock_response():
            yield mock_chunk_with_content

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            if chunk != "data: [DONE]\n\n":
                result_chunks.append(chunk)

        chunk_data = result_chunks[0]
        import json

        json_str = chunk_data.replace("data: ", "").strip()
        parsed = json.loads(json_str)

        # In OpenAI format, these fields should not appear when they're None due to exclude_none=True
        assert "finish_reason" not in parsed["choices"][0]
        assert "logprobs" not in parsed["choices"][0]
        # tool_calls should also not appear in delta when None
        assert "tool_calls" not in parsed["choices"][0]["delta"]

    @pytest.mark.asyncio
    async def test_process_streaming_response_empty_stream(self, openai_request):
        """Test processing empty streaming response"""

        async def mock_response():
            return
            yield  # Required: without this, Python treats mock_response as a coroutine, not an async generator

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        assert len(result_chunks) == 1
        assert result_chunks[0] == "data: [DONE]\n\n"


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
        assert tool_call.function.arguments == ""

    @pytest.mark.asyncio
    async def test_tool_call_arguments_modification(self):
        """Test that tool call arguments are properly modified"""

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


class TestCreateOpenAIResponse:
    """Test the create_openai_response function."""

    def test_content_receipt_response(self):
        """Test ContentReceipt gets converted to OpenAI format."""

        receipt = ContentReceipt(total_tokens=100, trimmed_tokens=10)
        result = create_openai_response(receipt)

        assert result is not None
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

        json_str = result.replace("data: ", "").strip()
        data = json.loads(json_str)

        assert data["object"] == "brave-chat.contentReceipt"
        assert data["total_tokens"] == 100
        assert data["trimmed_tokens"] == 10

    def test_web_sources_response(self):
        """Test WebSources gets converted to OpenAI format."""

        sources = [WebSource(title="Example", url="https://example.com", favicon=None)]
        web_sources = WebSources(sources=sources)
        result = create_openai_response(web_sources)

        assert result is not None

        json_str = result.replace("data: ", "").strip()
        data = json.loads(json_str)

        assert data["object"] == "brave-chat.webSources"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["title"] == "Example"
        assert data["sources"][0]["url"] == "https://example.com"

    def test_web_sources_with_rich_results(self):
        """Test WebSources with rich_results gets converted correctly."""

        sources = [WebSource(title="Test", url="https://test.com", favicon=None)]
        rich_results = [{"type": "answer", "content": "42"}]
        web_sources = WebSources(sources=sources, rich_results=rich_results)
        result = create_openai_response(web_sources)

        assert result is not None

        json_str = result.replace("data: ", "").strip()
        data = json.loads(json_str)

        assert data["object"] == "brave-chat.webSources"
        assert "rich_results" in data
        assert data["rich_results"] == rich_results

    def test_search_queries_response(self):
        """Test SearchQueries gets converted to OpenAI format."""

        queries = SearchQueries(queries=["python async", "SSE streaming"])
        result = create_openai_response(queries)

        assert result is not None

        json_str = result.replace("data: ", "").strip()
        data = json.loads(json_str)

        assert data["object"] == "brave-chat.searchQueries"
        assert data["queries"] == ["python async", "SSE streaming"]

    def test_unknown_response_type_returns_none(self):
        """Test that unknown response types return None."""

        class UnknownResponse(BaseModel):
            foo: str

        unknown = UnknownResponse(foo="bar")
        result = create_openai_response(unknown)

        assert result is None

    def test_response_with_none_values_excluded(self):
        """Test that None values are excluded from the response."""

        sources = [WebSource(title="Test", url="https://test.com", favicon=None)]
        web_sources = WebSources(sources=sources, rich_results=None)
        result = create_openai_response(web_sources)

        assert result is not None

        json_str = result.replace("data: ", "").strip()
        data = json.loads(json_str)

        assert "favicon" not in data["sources"][0]
        assert "rich_results" not in data

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


class TestProcessNonStreamingAlignmentCheck:
    """Test the process_non_streaming_alignment_check function."""

    @pytest.fixture
    def mock_response_with_tool_calls(self):
        """Mock non-streaming response with tool calls"""
        return {
            "id": "resp-123",
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

    @pytest.fixture
    def mock_response_without_tool_calls(self):
        """Mock non-streaming response without tool calls"""
        return {
            "id": "resp-123",
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello, how can I help you?",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    @pytest.fixture
    def mock_messages(self):
        """Mock messages for alignment check with untrusted content (page text)"""
        return [
            UserMessage(
                content=[
                    TextContentPart(
                        type="text",
                        text="What's the weather in San Francisco?",
                    ),
                    PageTextContentPart(
                        type="brave-page-text",
                        text="Some webpage content",
                    ),
                ]
            ),
        ]

    @pytest.fixture
    def mock_messages_trusted_only(self):
        """Mock messages with only trusted content (plain text + bypassed tool call)"""
        assistant_tool_call = MagicMock()
        assistant_tool_call.function.name = "user_choice_tool"

        assistant_msg = MagicMock()
        assistant_msg.role = "assistant"
        assistant_msg.tool_calls = [assistant_tool_call]

        return [
            UserMessage(content="What's the weather in San Francisco?"),
            assistant_msg,
            ToolMessage(content="User selected option A", tool_call_id="call_prev"),
        ]

    @pytest.mark.asyncio
    @patch("aichat.serve.open_ai_adapter.perform_alignment_check")
    async def test_alignment_check_with_tool_calls(
        self, mock_perform_alignment, mock_response_with_tool_calls, mock_messages
    ):
        """Test alignment check is performed for tool calls"""
        mock_perform_alignment.return_value = AlignmentCheckData(
            allowed=True, reasoning="Tool call is aligned with user request"
        )

        result = await process_non_streaming_alignment_check(
            mock_response_with_tool_calls, mock_messages
        )

        mock_perform_alignment.assert_called_once_with(
            "get_weather",
            '{"location": "San Francisco"}',
            mock_messages,
            None,
            injection_scan_task=None,
        )

        tool_call = result["choices"][0]["message"]["tool_calls"][0]
        assert "alignment_check" in tool_call
        assert tool_call["alignment_check"]["allowed"] is True
        assert (
            tool_call["alignment_check"]["reasoning"]
            == "Tool call is aligned with user request"
        )

    @pytest.mark.asyncio
    async def test_no_alignment_check_without_tool_calls(
        self, mock_response_without_tool_calls, mock_messages
    ):
        """Test no alignment check when there are no tool calls"""
        result = await process_non_streaming_alignment_check(
            mock_response_without_tool_calls, mock_messages
        )

        assert result == mock_response_without_tool_calls
        assert "alignment_check" not in str(result)

    @pytest.mark.asyncio
    @patch("aichat.serve.open_ai_adapter.perform_alignment_check")
    async def test_multiple_tool_calls(self, mock_perform_alignment, mock_messages):
        """Test alignment check for multiple tool calls"""
        mock_perform_alignment.side_effect = [
            AlignmentCheckData(allowed=True, reasoning="First tool aligned"),
            AlignmentCheckData(allowed=False, reasoning="Second tool not aligned"),
        ]

        response = {
            "id": "resp-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me help you with that.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "send_email",
                                    "arguments": '{"to": "user@example.com"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        result = await process_non_streaming_alignment_check(response, mock_messages)

        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["alignment_check"]["allowed"] is True
        assert tool_calls[0]["alignment_check"]["reasoning"] == "First tool aligned"
        assert tool_calls[1]["alignment_check"]["allowed"] is False
        assert (
            tool_calls[1]["alignment_check"]["reasoning"] == "Second tool not aligned"
        )

    @pytest.mark.asyncio
    @patch("aichat.serve.open_ai_adapter.security_settings")
    @patch("aichat.serve.open_ai_adapter.perform_alignment_check")
    async def test_bypass_tool_skips_alignment_check(
        self, mock_perform_alignment, mock_security_settings, mock_messages
    ):
        """Test that bypass tools skip alignment check"""
        mock_security_settings.allowed_bypass_tools = {
            "alignment_check_bypass": ["safe_tool"]
        }

        response = {
            "id": "resp-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "safe_tool",
                                    "arguments": "{}",
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        with patch("aichat.serve.open_ai_adapter.ALIGNMENT_CHECK_TOTAL"):
            result = await process_non_streaming_alignment_check(
                response, mock_messages
            )

        tool_call = result["choices"][0]["message"]["tool_calls"][0]
        assert "alignment_check" not in tool_call

    @pytest.mark.asyncio
    @patch("aichat.serve.open_ai_adapter.perform_alignment_check")
    async def test_no_alignment_check_with_trusted_content_only(
        self,
        mock_perform_alignment,
        mock_response_with_tool_calls,
        mock_messages_trusted_only,
    ):
        """Test alignment check is skipped when messages contain only trusted content"""
        result = await process_non_streaming_alignment_check(
            mock_response_with_tool_calls, mock_messages_trusted_only
        )

        mock_perform_alignment.assert_not_called()

        tool_call = result["choices"][0]["message"]["tool_calls"][0]
        assert "alignment_check" not in tool_call

    @pytest.mark.asyncio
    @patch("aichat.serve.open_ai_adapter.perform_alignment_check")
    async def test_alignment_check_with_untrusted_tool_results(
        self,
        mock_perform_alignment,
    ):
        """Test alignment check runs when conversation has tool call results (untrusted)"""
        mock_perform_alignment.return_value = AlignmentCheckData(
            allowed=True, reasoning="Tool call is aligned"
        )

        # Messages with assistant tool call + tool result (untrusted content)
        assistant_tool_call = MagicMock()
        assistant_tool_call.function.name = "get_weather"

        assistant_msg = MagicMock()
        assistant_msg.role = "assistant"
        assistant_msg.tool_calls = [assistant_tool_call]

        messages = [
            UserMessage(content="What's the weather?"),
            assistant_msg,
            ToolMessage(content="72°F and sunny", tool_call_id="call_prev"),
        ]

        response = {
            "id": "resp-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_456",
                                "type": "function",
                                "function": {
                                    "name": "send_email",
                                    "arguments": '{"to": "user@example.com"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        result = await process_non_streaming_alignment_check(response, messages)

        # Verify alignment check was called (because of untrusted tool results)
        mock_perform_alignment.assert_called_once()

        tool_call = result["choices"][0]["message"]["tool_calls"][0]
        assert "alignment_check" in tool_call
        assert tool_call["alignment_check"]["allowed"] is True


class TestInlineSearchIntegration:
    """Test cases for inline search integration in streaming responses"""

    @pytest.fixture
    def openai_request(self):
        """Mock OpenAI request for streaming tests"""
        return OpenAIRequest(
            model="test-model",
            messages=[UserMessage(content="Tell me about Python")],
            stream=True,
        )

    @pytest.fixture
    def mock_chunk_with_inline_search_pattern(self):
        """Mock streaming chunk with inline search pattern"""
        return ChatCompletionChunk(
            id="chunk-inline-1",
            object="chat.completion.chunk",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="Here's some info:\n::search[python tutorial]{type=web}\n",
                        role="assistant",
                    ),
                    finish_reason=None,
                    logprobs=None,
                )
            ],
        )

    @pytest.fixture
    def mock_inline_search_result(self):
        """Mock InlineSearch result"""
        return InlineSearch(
            query="python tutorial",
            results=[
                InlineSearchResult(
                    type="search_result",
                    title="Learn Python",
                    url="https://example.com/python",
                    description="Python tutorial for beginners",
                    meta_url=InlineSearchResultMeta(
                        netloc="example.com", path="/python", favicon=""
                    ),
                    thumbnail=None,
                    age=None,
                )
            ],
        )

    @pytest.mark.asyncio
    @patch("aichat.serve.open_ai_api.security_settings")
    @patch("aichat.serve.open_ai_api.InlineSearchHelper")
    async def test_inline_search_detection_in_streaming_response(
        self,
        mock_helper_class,
        mock_security_settings,
        openai_request,
        mock_chunk_with_inline_search_pattern,
        mock_inline_search_result,
    ):
        """Test that handle_received_completion is called and results are yielded"""
        mock_security_settings.alignment_checking_enabled = False
        mock_helper = MagicMock()
        mock_helper.get_inline_search_chunks = AsyncMock(
            return_value=[mock_inline_search_result]
        )
        mock_helper_class.return_value = mock_helper

        async def mock_response():
            yield mock_chunk_with_inline_search_pattern

        result_chunks = []
        async for chunk in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk)

        mock_helper.handle_received_completion.assert_called()
        mock_helper.get_inline_search_chunks.assert_called_once()

        assert result_chunks[-1] == "data: [DONE]\n\n"
        assert any(
            "brave-chat.inlineSearch" in c for c in result_chunks if isinstance(c, str)
        )

    @pytest.mark.asyncio
    @patch("aichat.serve.open_ai_api.security_settings")
    @patch("aichat.serve.open_ai_api.InlineSearchHelper")
    async def test_inline_search_multiple_results_emitted(
        self,
        mock_helper_class,
        mock_security_settings,
        openai_request,
        mock_inline_search_result,
    ):
        """Test that multiple inline search results are all yielded"""
        mock_security_settings.alignment_checking_enabled = False
        result2 = InlineSearch(
            query="second query",
            results=[
                InlineSearchResult(
                    type="search_result",
                    title="Second Result",
                    url="https://example.com/second",
                    description="Second",
                    meta_url=InlineSearchResultMeta(
                        netloc="example.com", path="/second", favicon=""
                    ),
                    thumbnail=None,
                    age=None,
                )
            ],
        )
        mock_helper = MagicMock()
        mock_helper.get_inline_search_chunks = AsyncMock(
            return_value=[mock_inline_search_result, result2]
        )
        mock_helper_class.return_value = mock_helper

        chunk = ChatCompletionChunk(
            id="chunk-1",
            object="chat.completion.chunk",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="Some text", role="assistant"),
                    finish_reason=None,
                    logprobs=None,
                )
            ],
        )

        async def mock_response():
            yield chunk

        result_chunks = [
            c
            async for c in process_streaming_response(
                mock_response(), openai_request, []
            )
        ]

        inline_search_chunks = [
            c
            for c in result_chunks
            if isinstance(c, str) and "brave-chat.inlineSearch" in c
        ]
        assert len(inline_search_chunks) == 2

    @pytest.mark.asyncio
    @patch("aichat.serve.open_ai_api.security_settings")
    @patch("aichat.serve.open_ai_api.InlineSearchHelper")
    async def test_inline_search_handle_completion_called_per_chunk(
        self,
        mock_helper_class,
        mock_security_settings,
        openai_request,
        mock_inline_search_result,
    ):
        """Test that handle_received_completion is called for each content chunk"""
        mock_security_settings.alignment_checking_enabled = False
        mock_helper = MagicMock()
        mock_helper.get_inline_search_chunks = AsyncMock(return_value=[])
        mock_helper_class.return_value = mock_helper

        chunks = [
            ChatCompletionChunk(
                id=f"chunk-{i}",
                object="chat.completion.chunk",
                created=1234567890,
                model="test-model",
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(content=f"content {i}", role="assistant"),
                        finish_reason=None,
                        logprobs=None,
                    )
                ],
            )
            for i in range(3)
        ]

        async def mock_response():
            for chunk in chunks:
                yield chunk

        async for _ in process_streaming_response(mock_response(), openai_request, []):
            pass

        assert mock_helper.handle_received_completion.call_count == 3

    @pytest.mark.asyncio
    @patch("aichat.serve.open_ai_api.security_settings")
    @patch("aichat.serve.open_ai_api.InlineSearchHelper")
    async def test_inline_search_no_results_when_no_patterns(
        self,
        mock_helper_class,
        mock_security_settings,
        openai_request,
    ):
        """Test that no inline search chunks are emitted when helper returns none"""
        mock_security_settings.alignment_checking_enabled = False
        mock_helper = MagicMock()
        mock_helper.get_inline_search_chunks = AsyncMock(return_value=[])
        mock_helper_class.return_value = mock_helper

        chunk = ChatCompletionChunk(
            id="chunk-1",
            object="chat.completion.chunk",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="Plain text", role="assistant"),
                    finish_reason=None,
                    logprobs=None,
                )
            ],
        )

        async def mock_response():
            yield chunk

        result_chunks = [
            c
            async for c in process_streaming_response(
                mock_response(), openai_request, []
            )
        ]

        inline_search_chunks = [
            c
            for c in result_chunks
            if isinstance(c, str) and "brave-chat.inlineSearch" in c
        ]
        assert len(inline_search_chunks) == 0
        assert result_chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    @patch("aichat.serve.open_ai_api.security_settings")
    async def test_inline_search_emitted_at_end(
        self, mock_security_settings, openai_request
    ):
        """Test that inline search results are emitted at the end of the stream"""
        mock_security_settings.alignment_checking_enabled = False

        chunk = ChatCompletionChunk(
            id="chunk-1",
            object="chat.completion.chunk",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="Some content", role="assistant"),
                    finish_reason=None,
                    logprobs=None,
                )
            ],
        )

        async def mock_response():
            yield chunk

        result_chunks = []
        async for chunk_str in process_streaming_response(
            mock_response(), openai_request, []
        ):
            result_chunks.append(chunk_str)

        assert result_chunks[-1] == "data: [DONE]\n\n"

    def test_create_openai_response_with_inline_search(self, mock_inline_search_result):
        """Test that InlineSearch is converted to OpenAI format correctly"""
        result = create_openai_response(mock_inline_search_result)

        assert result is not None
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

        json_str = result.replace("data: ", "").strip()
        data = json.loads(json_str)

        assert data["object"] == "brave-chat.inlineSearch"
        assert data["query"] == "python tutorial"
        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "Learn Python"
        assert data["results"][0]["url"] == "https://example.com/python"

    def test_create_openai_response_with_inline_search_multiple_results(self):
        """Test InlineSearch with multiple results"""
        inline_search = InlineSearch(
            query="python tutorials",
            results=[
                InlineSearchResult(
                    type="search_result",
                    title="Tutorial 1",
                    url="https://example.com/1",
                    description="First tutorial",
                    meta_url=InlineSearchResultMeta(
                        netloc="example.com", path="/1", favicon=""
                    ),
                    thumbnail=None,
                    age=None,
                ),
                InlineSearchResult(
                    type="search_result",
                    title="Tutorial 2",
                    url="https://example.com/2",
                    description="Second tutorial",
                    meta_url=InlineSearchResultMeta(
                        netloc="example.com", path="/2", favicon=""
                    ),
                    thumbnail=InlineSearchResultThumbnail(
                        src="https://example.com/thumb.jpg"
                    ),
                    age="2 days ago",
                ),
            ],
        )

        result = create_openai_response(inline_search)

        assert result is not None

        json_str = result.replace("data: ", "").strip()
        data = json.loads(json_str)

        assert data["object"] == "brave-chat.inlineSearch"
        assert data["query"] == "python tutorials"
        assert len(data["results"]) == 2
        assert data["results"][0]["title"] == "Tutorial 1"
        assert data["results"][1]["title"] == "Tutorial 2"
        assert "thumbnail" in data["results"][1]
        assert data["results"][1]["thumbnail"]["src"] == "https://example.com/thumb.jpg"
        assert data["results"][1]["age"] == "2 days ago"

    def test_create_openai_response_with_inline_search_no_results(self):
        """Test InlineSearch with no results"""
        inline_search = InlineSearch(query="nonexistent query", results=[])

        result = create_openai_response(inline_search)

        assert result is not None

        json_str = result.replace("data: ", "").strip()
        data = json.loads(json_str)

        assert data["object"] == "brave-chat.inlineSearch"
        assert data["query"] == "nonexistent query"
        assert data["results"] == []

    @pytest.mark.asyncio
    @patch("aichat.serve.open_ai_api.security_settings")
    @patch("aichat.serve.open_ai_api.InlineSearchHelper")
    async def test_inline_search_content_passed_to_helper(
        self,
        mock_helper_class,
        mock_security_settings,
        openai_request,
    ):
        """Test that accumulated content is passed to handle_received_completion"""
        mock_security_settings.alignment_checking_enabled = False
        mock_helper = MagicMock()
        mock_helper.get_inline_search_chunks = AsyncMock(return_value=[])
        mock_helper_class.return_value = mock_helper

        content = (
            "::search[web query]{type=web}\n" "::search[image query]{type=images}\n"
        )
        chunk = ChatCompletionChunk(
            id="chunk-1",
            object="chat.completion.chunk",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content=content, role="assistant"),
                    finish_reason=None,
                    logprobs=None,
                )
            ],
        )

        async def mock_response():
            yield chunk

        async for _ in process_streaming_response(mock_response(), openai_request, []):
            pass

        mock_helper.handle_received_completion.assert_called_once_with(content)
