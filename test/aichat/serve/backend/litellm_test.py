import os
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from openai.types.chat.chat_completion import ChatCompletion

from aichat.protocol.open_ai_protocol import Tool
from aichat.serve.backend.litellm import (
    LitellmBackend,
    _build_router_entry,
    _get_litellm_model_string,
    apply_claude_upstream_sampling_params,
    get_global_router,
    merge_model_extra_body,
)
from aichat.serve.services.models import ModelConfig


@pytest.fixture
def mock_model_config():
    """Mock ModelConfig for testing"""
    return ModelConfig(
        model_id="test-model",
        upstream_model="test-upstream-model",
        backend="litellm",
        api_base="https://test-api.com",
        api_key="test-api-key",
        inference_profile="TEST_INFERENCE_PROFILE",
        system_prompt_support=True,
        prompt_caching_support=None,
        prompt_caching_enabled=None,
        tool_support=True,
        image_support=False,
        audio_support=False,
        video_support=False,
        file_support=True,
        friendly_name="Test Model",
        maker="Test Maker",
        max_tokens=4096,
        max_tokens_premium=8192,
        conversation_token_limit=16384,
        conversation_token_limit_premium=32768,
        free=False,
        key="test-key",
        rate_limit=100,
        rate_limit_interval_seconds=60,
        max_pages=5,
        max_pages_premium=10,
        extra_body={"chat_template_kwargs": {"test_key": "test_value"}},
    )


@pytest.fixture
def mock_bedrock_model_config():
    """Mock ModelConfig for Bedrock testing"""
    return ModelConfig(
        model_id="test-bedrock-model",
        upstream_model="test-bedrock-upstream",
        backend="bedrock",
        api_base="https://bedrock-api.com",
        api_key=None,
        inference_profile="TEST_BEDROCK_PROFILE",
        system_prompt_support=True,
        prompt_caching_support=True,
        prompt_caching_enabled=True,
        tool_support=True,
        image_support=True,
        audio_support=False,
        video_support=False,
        file_support=True,
        friendly_name="Test Bedrock Model",
        maker="AWS",
        max_tokens=4096,
        max_tokens_premium=8192,
        conversation_token_limit=16384,
        conversation_token_limit_premium=32768,
        free=False,
        key="bedrock-key",
        rate_limit=50,
        rate_limit_interval_seconds=60,
        max_pages=3,
        max_pages_premium=8,
        extra_body=None,
    )


@pytest.fixture
def mock_tools():
    """Mock tools for testing"""
    return [
        Tool(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        )
    ]


class TestLitellmBackend:
    def test_init_with_standard_config(self, mock_model_config):
        """Test initialization with standard model config"""
        with patch.dict(os.environ, {"TEST_INFERENCE_PROFILE": "test-profile"}):
            backend = LitellmBackend(mock_model_config)

            assert backend.config == mock_model_config
            assert backend.inference_profile == "test-profile"
            assert backend.chat_template_kwargs == {"test_key": "test_value"}
            assert backend.tokenizer is not None
            assert backend.router is not None

    def test_init_with_bedrock_config(self, mock_bedrock_model_config):
        """Test initialization with Bedrock model config"""
        with patch.dict(os.environ, {"TEST_BEDROCK_PROFILE": "bedrock-profile"}):
            backend = LitellmBackend(mock_bedrock_model_config)

            assert backend.config == mock_bedrock_model_config
            assert backend.inference_profile == "bedrock-profile"
            assert backend.router is not None

    def test_init_without_inference_profile(self, mock_model_config):
        """Test initialization when inference profile env var is not set"""
        mock_model_config.inference_profile = None
        backend = LitellmBackend(mock_model_config)

        assert backend.inference_profile is None
        assert backend.router is not None

    def test_init_without_extra_body(self, mock_model_config):
        """Test initialization when extra_body is None"""
        mock_model_config.extra_body = None
        backend = LitellmBackend(mock_model_config)

        assert backend.chat_template_kwargs == {}

    @pytest.mark.asyncio
    async def test_converse_streaming(self, mock_model_config):
        """Test streaming conversation"""
        backend = LitellmBackend(mock_model_config)

        # Mock the router completion call
        mock_response = AsyncMock()
        messages = [{"role": "user", "content": "Hello"}]
        params = {"temperature": 0.7}

        mock_completion = AsyncMock(return_value=mock_response)
        backend.router.acompletion = mock_completion
        result = await backend.converse(messages, stream=True, params=params)

        # Verify router.acompletion was called with correct parameters
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == mock_model_config.model_id
        assert call_kwargs["messages"] == messages
        assert call_kwargs["stream"] is True
        assert call_kwargs["temperature"] == 0.7

        # Verify the backend returns the raw response (converter is called elsewhere)
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_converse_non_streaming(self, mock_model_config):
        """Test non-streaming conversation"""
        backend = LitellmBackend(mock_model_config)

        # Mock the router completion call
        mock_response = MagicMock(spec=ChatCompletion)
        messages = [{"role": "user", "content": "Hello"}]
        params = {"temperature": 0.7}

        mock_completion = AsyncMock(return_value=mock_response)
        backend.router.acompletion = mock_completion
        result = await backend.converse(messages, stream=False, params=params)

        # Verify router.acompletion was called with correct parameters
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == mock_model_config.model_id
        assert call_kwargs["messages"] == messages
        assert call_kwargs["stream"] is False
        assert call_kwargs["temperature"] == 0.7

        assert result == mock_response

    def test_build_params_with_tools_non_bedrock(self, mock_model_config, mock_tools):
        """Test building parameters with tools for non-Bedrock backend"""
        backend = LitellmBackend(mock_model_config)
        messages = [{"role": "user", "content": "Hello"}]

        params = backend.build_params(stream=True, tools=mock_tools, messages=messages)

        assert "tools" in params
        assert params["tools"] == [tool.model_dump() for tool in mock_tools]
        assert params["tool_choice"] == "auto"
        # api_key is not added to params by build_params
        assert params["chat_template_kwargs"] == {"test_key": "test_value"}

    def test_build_params_with_tools_bedrock(
        self, mock_bedrock_model_config, mock_tools
    ):
        """Test building parameters with tools for Bedrock backend"""
        backend = LitellmBackend(mock_bedrock_model_config)
        messages = [{"role": "user", "content": "Hello"}]

        with patch(
            "aichat.serve.backend.litellm.format_tools_for_bedrock"
        ) as mock_format:
            mock_formatted_tools = [{"formatted": "tool"}]
            mock_format.return_value = mock_formatted_tools

            params = backend.build_params(
                stream=True, tools=mock_tools, messages=messages
            )

            assert "tools" in params
            # prompt_caching_support is enabled on this config, so the tools
            # prefix is cached by tagging the last tool with cache_control.
            assert params["tools"] == [
                {"formatted": "tool", "cache_control": {"type": "ephemeral"}}
            ]
            assert params["tool_choice"] == "auto"
            assert "api_key" not in params  # Bedrock doesn't use api_key
            mock_format.assert_called_once_with(mock_tools)

    def test_build_params_without_tools(self, mock_model_config):
        """Test building parameters without tools"""
        backend = LitellmBackend(mock_model_config)
        messages = [{"role": "user", "content": "Hello"}]

        params = backend.build_params(stream=False, tools=[], messages=messages)

        # OpenAIChatParams includes tools and tool_choice as None by default
        assert params["tools"] is None
        assert params["tool_choice"] is None
        # api_key is not added to params by build_params

    def test_build_params_without_chat_template_kwargs(self, mock_model_config):
        """Test building parameters when chat_template_kwargs is None"""
        mock_model_config.extra_body = None
        backend = LitellmBackend(mock_model_config)
        messages = [{"role": "user", "content": "Hello"}]

        params = backend.build_params(stream=False, tools=[], messages=messages)

        assert "chat_template_kwargs" not in params

    def test_build_params_applies_extra_body_sampling(self, mock_model_config):
        mock_model_config.extra_body = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        backend = LitellmBackend(mock_model_config)
        params = backend.build_params(stream=False, tools=[], messages=[])

        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.8
        assert params["top_k"] == 20
        assert params["presence_penalty"] == 1.5
        assert params["repetition_penalty"] == 1.0
        assert params["chat_template_kwargs"] == {"enable_thinking": False}

    def test_merge_model_extra_body_merges_chat_template_kwargs(self):
        params: dict = {"chat_template_kwargs": {"enable_thinking": False}}
        merge_model_extra_body(
            params, {"chat_template_kwargs": {"repetition_penalty": 1.1}}
        )
        assert params["chat_template_kwargs"] == {
            "enable_thinking": False,
            "repetition_penalty": 1.1,
        }

    def test_build_params_caps_max_tokens_using_conversation_limit_by_default(
        self, mock_model_config
    ):
        """Without an override, ``build_params`` caps ``max_tokens`` against
        the business-level conversation_token_limit_premium (32768)."""
        backend = LitellmBackend(mock_model_config)
        # Input larger than (32768 - 4096) so available headroom < max_tokens.
        with patch("aichat.serve.backend.litellm.count_tokens", return_value=30_000):
            params = backend.build_params(
                stream=False, tools=[], messages=[{"role": "user", "content": "x"}]
            )
        # available = 32768 - 30000 = 2768, which is less than configured 4096
        assert params["max_tokens"] == 2768

    def test_build_params_uses_context_window_override_when_supplied(
        self, mock_model_config
    ):
        """When ``context_window_override`` is supplied (e.g. by compaction),
        the cap uses that ceiling instead of conversation_token_limit*."""
        backend = LitellmBackend(mock_model_config)
        # Same input that would have triggered the cap above; with an override
        # of 200k it should leave max_tokens at the configured value.
        with patch("aichat.serve.backend.litellm.count_tokens", return_value=30_000):
            params = backend.build_params(
                stream=False,
                tools=[],
                messages=[{"role": "user", "content": "x"}],
                context_window_override=200_000,
            )
        assert params["max_tokens"] == 4096

    @pytest.mark.parametrize(
        "upstream_model",
        [
            "us.anthropic.claude-opus",
            "us.anthropic.claude-sonnet-4-5-v1:0",
        ],
    )
    def test_build_params_claude_opus_and_sonnet_omit_sampling_params(
        self, mock_model_config, upstream_model
    ):
        """Claude Opus and Sonnet reject temperature/top_p/top_k; build_params omits them."""
        mock_model_config.upstream_model = upstream_model
        backend = LitellmBackend(mock_model_config)
        params = backend.build_params(stream=False, tools=[], messages=[])
        assert "temperature" not in params
        assert "top_p" not in params
        assert "top_k" not in params

    def test_build_params_claude_haiku_drops_top_p_only(self, mock_model_config):
        """Other Claude models (e.g. Haiku) keep temperature but drop top_p."""
        mock_model_config.upstream_model = "us.anthropic.claude-haiku-4-5-v1:0"
        backend = LitellmBackend(mock_model_config)
        params = backend.build_params(stream=False, tools=[], messages=[])
        assert "top_p" not in params
        assert "temperature" in params

    def test_apply_claude_upstream_sampling_params(self):
        p = {"temperature": 0.5, "top_p": 0.9, "top_k": 5, "max_tokens": 100}
        apply_claude_upstream_sampling_params("claude-opus", p)
        assert p == {"max_tokens": 100}
        p2 = {"temperature": 0.5, "top_p": 0.9, "top_k": 5, "max_tokens": 100}
        apply_claude_upstream_sampling_params("claude-sonnet-4", p2)
        assert p2 == {"max_tokens": 100}
        p3 = {"top_p": 1.0}
        apply_claude_upstream_sampling_params("meta-llama", p3)
        assert p3 == {"top_p": 1.0}

    def test_set_litellm_model_id_with_inference_profile(self, mock_model_config):
        """Test that backend uses config.model_id, not a computed model_id"""
        with patch.dict(os.environ, {"TEST_INFERENCE_PROFILE": "test-profile"}):
            backend = LitellmBackend(mock_model_config)
            # The backend uses config.model_id directly, not a computed model_id
            assert backend.config.model_id == mock_model_config.model_id

    def test_set_litellm_model_id_without_inference_profile(self, mock_model_config):
        """Test that backend uses config.model_id"""
        mock_model_config.inference_profile = None
        backend = LitellmBackend(mock_model_config)
        # The backend uses config.model_id directly
        assert backend.config.model_id == mock_model_config.model_id

    @pytest.mark.asyncio
    @patch("aichat.serve.backend.litellm.model_settings")
    async def test_converse_retry_on_bedrock_rate_limit_streaming(
        self, mock_model_settings, mock_bedrock_model_config
    ):
        """Test that router is called correctly for streaming requests"""
        mock_model_settings.bedrock_max_retries = 3
        mock_model_settings.bedrock_retry_delay_seconds = 0.1

        backend = LitellmBackend(mock_bedrock_model_config)
        messages = [{"role": "user", "content": "Hello"}]
        params = {"temperature": 0.7}

        # Mock successful response
        mock_success_response = AsyncMock()
        backend.router.acompletion = AsyncMock(return_value=mock_success_response)
        result = await backend.converse(messages, stream=True, params=params)

        # Should return the router's response
        assert result == mock_success_response
        backend.router.acompletion.assert_called_once()

    @pytest.mark.asyncio
    @patch("aichat.serve.backend.litellm.model_settings")
    async def test_converse_retry_on_bedrock_rate_limit_non_streaming(
        self, mock_model_settings, mock_bedrock_model_config
    ):
        """Test that router is called correctly for non-streaming requests"""
        mock_model_settings.bedrock_max_retries = 3
        mock_model_settings.bedrock_retry_delay_seconds = 0.1

        backend = LitellmBackend(mock_bedrock_model_config)
        messages = [{"role": "user", "content": "Hello"}]
        params = {"temperature": 0.7}

        mock_success_response = MagicMock(spec=ChatCompletion)
        backend.router.acompletion = AsyncMock(return_value=mock_success_response)
        result = await backend.converse(messages, stream=False, params=params)

        assert result == mock_success_response
        backend.router.acompletion.assert_called_once()

    @pytest.mark.asyncio
    @patch("aichat.serve.backend.litellm.model_settings")
    async def test_converse_retry_exhausted(
        self, mock_model_settings, mock_bedrock_model_config
    ):
        """Test that RateLimitError is handled and converted to error response"""
        mock_model_settings.bedrock_max_retries = 3
        mock_model_settings.bedrock_retry_delay_seconds = 0.01

        backend = LitellmBackend(mock_bedrock_model_config)
        messages = [{"role": "user", "content": "Hello"}]
        params = {"temperature": 0.7}

        rate_limit_error = litellm.RateLimitError(
            message='BedrockException - {"message":"Too many tokens, please wait before trying again."}',
            model="test-model",
            llm_provider="bedrock",
        )

        backend.router.acompletion = AsyncMock(side_effect=rate_limit_error)
        # When router raises an error, converse catches it and returns error response
        result = await backend.converse(messages, stream=True, params=params)

        # The error should be handled and returned as an error response
        # Since it's streaming, it should be an async generator
        error_chunk = None
        async for response in result:
            error_chunk = response
            break

        # Verify error response structure
        assert error_chunk is not None
        assert hasattr(error_chunk, "choices")
        assert len(error_chunk.choices) > 0
        # RateLimitError gets converted to INTERNAL_ERROR by handle_litellm_error
        assert error_chunk.choices[0].delta.content is not None

    @pytest.mark.asyncio
    @patch("aichat.serve.backend.litellm.model_settings")
    async def test_converse_no_retry_on_non_retryable_error(
        self, mock_model_settings, mock_bedrock_model_config
    ):
        """Test that non-retryable errors are raised immediately"""
        mock_model_settings.bedrock_max_retries = 3
        mock_model_settings.bedrock_retry_delay_seconds = 0.1

        backend = LitellmBackend(mock_bedrock_model_config)
        messages = [{"role": "user", "content": "Hello"}]
        params = {"temperature": 0.7}

        other_error = Exception("Some other error")

        backend.router.acompletion = AsyncMock(side_effect=other_error)
        result = await backend.converse(messages, stream=True, params=params)

        # Should return an async generator that yields error response as ChatCompletionChunk
        error_chunk = None
        async for response in result:
            error_chunk = response
            break

        # Verify error response structure - should be a ChatCompletionChunk
        assert error_chunk is not None
        assert hasattr(error_chunk, "choices")
        assert len(error_chunk.choices) > 0
        assert error_chunk.choices[0].delta.content is not None
        assert (
            "There was an issue connecting to the model."
            in error_chunk.choices[0].delta.content
        )
        assert error_chunk.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_converse_bedrock_streaming_with_validation(
        self, mock_bedrock_model_config
    ):
        """Test Bedrock streaming conversation with validation"""
        backend = LitellmBackend(mock_bedrock_model_config)

        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Result"},
        ]

        mock_response = AsyncMock()
        params = {"temperature": 0.7}

        mock_completion = AsyncMock(return_value=mock_response)
        backend.router.acompletion = mock_completion
        result = await backend.converse(messages, stream=True, params=params)

        # Verify the response is returned
        assert result == mock_response

        # Verify router was called (validation happens internally)
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        # Messages should be passed through validation
        assert "messages" in call_args[1]

    @pytest.mark.asyncio
    async def test_converse_bedrock_non_streaming_with_validation(
        self, mock_bedrock_model_config
    ):
        """Test Bedrock non-streaming conversation with validation"""
        backend = LitellmBackend(mock_bedrock_model_config)

        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Result"},
        ]

        mock_response = MagicMock(spec=ChatCompletion)
        params = {"temperature": 0.7}

        mock_completion = AsyncMock(return_value=mock_response)
        backend.router.acompletion = mock_completion
        result = await backend.converse(messages, stream=False, params=params)

        # Verify the response is returned
        assert result == mock_response

        # Verify router was called (validation happens internally)
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        # Messages should be passed through validation
        assert "messages" in call_args[1]

    @pytest.mark.asyncio
    async def test_converse_bedrock_filters_invalid_json_tool_calls(
        self, mock_bedrock_model_config
    ):
        """Test that Bedrock backend filters out tool calls with invalid JSON arguments"""
        backend = LitellmBackend(mock_bedrock_model_config)

        # Message with invalid JSON in tool call arguments
        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_invalid",
                        "type": "function",
                        "function": {
                            "name": "click_element",
                            # Missing quotes around document_identifier value
                            "arguments": '{"target": {"node_id":940,"document_identifier":ABC123}}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_invalid",
                "content": "Error: Failed to parse input JSON",
            },
        ]

        mock_response = AsyncMock()
        params = {"temperature": 0.7}

        mock_completion = AsyncMock(return_value=mock_response)
        backend.router.acompletion = mock_completion
        result = await backend.converse(messages, stream=True, params=params)

        # Verify the response is returned
        assert result == mock_response

        # Verify router was called
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args

        # Get the messages that were actually passed to router
        filtered_messages = call_args[1]["messages"]

        # The invalid tool call and its result should be filtered out
        # Should have user message + assistant message (without tool_calls) = 2 messages
        assert len(filtered_messages) == 2
        assert filtered_messages[0]["role"] == "user"
        assert filtered_messages[1]["role"] == "assistant"
        # Assistant message should not have tool_calls since it was invalid
        assert "tool_calls" not in filtered_messages[1]

    @pytest.mark.asyncio
    @patch("aichat.serve.services.models.model_settings")
    @patch("aichat.serve.backend.litellm.filter_tool_call_result_pairing")
    async def test_converse_filters_when_bedrock_fallback_configured(
        self, mock_filter, mock_model_settings, mock_model_config
    ):
        """A non-bedrock primary with a bedrock fallback runs the cleanup filter."""
        mock_model_config.fallback_models = ["fallback-bedrock-model"]
        mock_model_settings.models = {
            "fallback-bedrock-model": {"backend": "bedrock"},
        }
        backend = LitellmBackend(mock_model_config)

        messages = [{"role": "user", "content": "Hi"}]
        cleaned_messages = [{"role": "user", "content": "Hi (cleaned)"}]
        mock_filter.return_value = cleaned_messages

        backend.router.acompletion = AsyncMock(return_value=AsyncMock())
        await backend.converse(messages, stream=True, params={})

        mock_filter.assert_called_once_with(
            messages, model_id=mock_model_config.model_id
        )
        # Cleaned messages should be the ones forwarded to litellm
        forwarded = backend.router.acompletion.call_args[1]["messages"]
        assert forwarded == cleaned_messages

    @pytest.mark.asyncio
    @patch("aichat.serve.services.models.model_settings")
    @patch("aichat.serve.backend.litellm.filter_tool_call_result_pairing")
    async def test_converse_does_not_filter_without_bedrock_fallback(
        self, mock_filter, mock_model_settings, mock_model_config
    ):
        """Non-bedrock primary with no bedrock fallback skips the cleanup filter."""
        mock_model_config.fallback_models = ["another-vllm-model"]
        mock_model_settings.models = {
            "another-vllm-model": {"backend": "vllm"},
        }
        backend = LitellmBackend(mock_model_config)

        messages = [{"role": "user", "content": "Hi"}]
        backend.router.acompletion = AsyncMock(return_value=AsyncMock())
        await backend.converse(messages, stream=True, params={})

        mock_filter.assert_not_called()
        # Original messages should be forwarded unchanged
        forwarded = backend.router.acompletion.call_args[1]["messages"]
        assert forwarded == messages

    @pytest.mark.asyncio
    async def test_converse_bedrock_keeps_valid_filters_invalid_tool_calls(
        self, mock_bedrock_model_config
    ):
        """Test that Bedrock backend keeps valid tool calls and filters invalid ones"""
        backend = LitellmBackend(mock_bedrock_model_config)

        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "content": "Let me help",
                "tool_calls": [
                    {
                        "id": "call_valid",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    },
                    {
                        "id": "call_invalid",
                        "type": "function",
                        "function": {
                            "name": "click_element",
                            "arguments": '{"target": {invalid_json}}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_valid", "content": "Sunny, 72F"},
            {
                "role": "tool",
                "tool_call_id": "call_invalid",
                "content": "Error: Failed to parse input JSON",
            },
        ]

        mock_response = MagicMock(spec=ChatCompletion)
        params = {"temperature": 0.7}

        mock_completion = AsyncMock(return_value=mock_response)
        backend.router.acompletion = mock_completion
        result = await backend.converse(messages, stream=False, params=params)

        # Verify the response is returned
        assert result == mock_response

        # Verify router was called
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args

        # Get the messages that were actually passed to router
        filtered_messages = call_args[1]["messages"]

        # The invalid tool call (and its orphaned result) is filtered out, then the
        # assistant turn carrying both content and a tool_call is split into canonical
        # order: user + assistant{tool_call} + tool{result} + assistant{content}
        assert len(filtered_messages) == 4
        assert filtered_messages[0]["role"] == "user"
        assert filtered_messages[1]["role"] == "assistant"
        assert len(filtered_messages[1]["tool_calls"]) == 1
        assert filtered_messages[1]["tool_calls"][0]["id"] == "call_valid"
        assert "content" not in filtered_messages[1]
        assert filtered_messages[2]["role"] == "tool"
        assert filtered_messages[2]["tool_call_id"] == "call_valid"
        assert filtered_messages[3]["role"] == "assistant"
        assert filtered_messages[3]["content"] == "Let me help"


class TestGetLiteLLMModelString:
    """Test cases for _get_litellm_model_string function"""

    def test_with_inference_profile_and_env_var(self):
        """Test when inference_profile is set and env var exists"""
        with patch.dict(os.environ, {"TEST_PROFILE": "my-profile-value"}):
            result = _get_litellm_model_string(
                upstream_model="test-model",
                api_base=None,
                inference_profile="TEST_PROFILE",
            )
            assert result == "bedrock/converse/my-profile-value"

    def test_with_inference_profile_no_env_var(self):
        """Test when inference_profile is set but env var doesn't exist"""
        with patch.dict(os.environ, {}, clear=True):
            result = _get_litellm_model_string(
                upstream_model="test-model",
                api_base=None,
                inference_profile="NONEXISTENT_PROFILE",
            )
            assert result == "bedrock/test-model"

    def test_with_near_ai_api_base(self):
        """Test when api_base contains 'near.ai'"""
        result = _get_litellm_model_string(
            upstream_model="test-model",
            api_base="https://api.near.ai/v1",
            inference_profile=None,
        )
        assert result == "openai/test-model"

    def test_with_near_ai_api_base_and_inference_profile(self):
        """Test that inference_profile takes precedence over api_base"""
        with patch.dict(os.environ, {"TEST_PROFILE": "my-profile"}):
            result = _get_litellm_model_string(
                upstream_model="test-model",
                api_base="https://api.near.ai/v1",
                inference_profile="TEST_PROFILE",
            )
            assert result == "bedrock/converse/my-profile"

    def test_default_hosted_vllm(self):
        """Test default case returns hosted_vllm format"""
        result = _get_litellm_model_string(
            upstream_model="test-model",
            api_base="https://api.example.com",
            inference_profile=None,
        )
        assert result == "hosted_vllm/test-model"

    def test_bedrock_mantle_openai_responses_model(self):
        """OpenAI frontier mantle models use openai/ + Responses API bridge."""
        result = _get_litellm_model_string(
            upstream_model="openai.gpt-5.5",
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
            inference_profile=None,
            backend="bedrock_mantle",
        )
        assert result == "openai/openai.gpt-5.5"

    def test_bedrock_mantle_non_openai_backend(self):
        """Non-openai mantle models keep bedrock_mantle/ prefix."""
        result = _get_litellm_model_string(
            upstream_model="anthropic.claude-foo",
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
            inference_profile=None,
            backend="bedrock_mantle",
        )
        assert result == "bedrock_mantle/anthropic.claude-foo"


class TestBuildRouterEntry:
    """Test cases for _build_router_entry function"""

    @patch("aichat.serve.backend.litellm.model_settings")
    def test_build_router_entry_standard_config(self, mock_settings):
        """Test building router entry with standard config"""
        model_config = {
            "upstream_model": "test-upstream-model",
            "address": "https://api.example.com",
            "inference_profile": None,
        }
        mock_settings.models = {"test-model-id": model_config}
        result = _build_router_entry("test-model-id")

        assert result is not None
        assert result["model_name"] == "test-model-id"
        assert "litellm_params" in result
        assert result["litellm_params"]["model"] == "hosted_vllm/test-upstream-model"
        assert result["litellm_params"]["api_base"] == "https://api.example.com"

    @patch("aichat.serve.backend.litellm.model_settings")
    def test_build_router_entry_with_inference_profile(self, mock_settings):
        """Test building router entry with inference_profile"""
        with patch.dict(os.environ, {"TEST_PROFILE": "my-profile-value"}):
            model_config = {
                "upstream_model": "test-upstream-model",
                "address": "https://api.example.com",
                "inference_profile": "TEST_PROFILE",
            }
            mock_settings.models = {"test-model-id": model_config}
            result = _build_router_entry("test-model-id")

            assert result is not None
            assert result["model_name"] == "test-model-id"
            assert (
                result["litellm_params"]["model"] == "bedrock/converse/my-profile-value"
            )
            assert result["litellm_params"]["api_base"] == "https://api.example.com"

    @patch("aichat.serve.backend.litellm.model_settings")
    def test_build_router_entry_with_near_ai(self, mock_settings):
        """Test building router entry with near.ai address"""
        model_config = {
            "upstream_model": "test-upstream-model",
            "address": "https://api.near.ai/v1",
            "inference_profile": None,
        }
        mock_settings.models = {"test-model-id": model_config}
        result = _build_router_entry("test-model-id")

        assert result is not None
        assert result["model_name"] == "test-model-id"
        assert result["litellm_params"]["model"] == "openai/test-upstream-model"
        assert result["litellm_params"]["api_base"] == "https://api.near.ai/v1"

    @patch("aichat.serve.backend.litellm.model_settings")
    @patch("aichat.serve.backend.litellm.external_service_settings")
    def test_build_router_entry_with_near_ai_api_key(
        self, mock_ext_settings, mock_settings
    ):
        """Test that Near AI models get API key in router entry"""
        mock_ext_settings.near_api_key = "test-near-key"
        model_config = {
            "upstream_model": "test-upstream-model",
            "address": "https://api.near.ai/v1",
            "inference_profile": None,
        }
        mock_settings.models = {"near-test-model": model_config}
        result = _build_router_entry("near-test-model")

        assert result is not None
        assert result["model_name"] == "near-test-model"
        assert result["litellm_params"]["api_key"] == "test-near-key"
        assert result["litellm_params"]["api_base"] == "https://api.near.ai/v1"

    @patch("aichat.serve.backend.litellm.model_settings")
    def test_build_router_entry_with_inference_profile_no_env_var(self, mock_settings):
        """Test building router entry with inference_profile but no env var"""
        with patch.dict(os.environ, {}, clear=True):
            model_config = {
                "upstream_model": "test-upstream-model",
                "address": "https://api.example.com",
                "inference_profile": "NONEXISTENT_PROFILE",
            }
            mock_settings.models = {"test-model-id": model_config}
            result = _build_router_entry("test-model-id")

            assert result is not None
            assert result["model_name"] == "test-model-id"
            assert result["litellm_params"]["model"] == "bedrock/test-upstream-model"
            assert result["litellm_params"]["api_base"] == "https://api.example.com"

    @patch("aichat.serve.backend.litellm.model_settings")
    def test_build_router_entry_bedrock_mantle(self, mock_settings):
        """Test building router entry for bedrock_mantle backend"""
        model_config = {
            "backend": "bedrock_mantle",
            "upstream_model": "openai.gpt-5.5",
            "address": "https://bedrock-mantle.us-east-2.api.aws/openai/v1",
            "inference_profile": None,
        }
        mock_settings.models = {"bedrock-openai.gpt-5.5": model_config}
        result = _build_router_entry("bedrock-openai.gpt-5.5")

        assert result is not None
        assert result["model_name"] == "bedrock-openai.gpt-5.5"
        assert result["litellm_params"]["model"] == "openai/openai.gpt-5.5"
        assert result["litellm_params"]["additional_drop_params"] == ["output_config"]
        assert (
            result["litellm_params"]["api_base"]
            == "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
        )

    @patch("aichat.serve.backend.litellm.model_settings")
    def test_build_router_entry_missing_address(self, mock_settings):
        """Test building router entry with missing address"""
        model_config = {
            "upstream_model": "test-upstream-model",
            "inference_profile": None,
        }
        mock_settings.models = {"test-model-id": model_config}
        result = _build_router_entry("test-model-id")

        assert result is not None
        assert result["model_name"] == "test-model-id"
        assert result["litellm_params"]["model"] == "hosted_vllm/test-upstream-model"
        assert result["litellm_params"]["api_base"] is None

    @patch("aichat.serve.backend.litellm.model_settings")
    def test_build_router_entry_empty_config(self, mock_settings):
        """Test building router entry with empty config"""
        model_config = {}
        mock_settings.models = {"test-model-id": model_config}
        result = _build_router_entry("test-model-id")

        assert result is not None
        assert result["model_name"] == "test-model-id"
        assert result["litellm_params"]["model"] == "hosted_vllm/None"

    @patch("aichat.serve.backend.litellm.model_settings")
    def test_build_router_entry_with_fallbacks(self, mock_settings):
        """Test building router entry with fallback models"""
        model_config = {
            "upstream_model": "test-upstream-model",
            "address": "https://api.example.com",
            "inference_profile": None,
            "fallback_models": ["claude-haiku", "qwen-14b-instruct"],
        }
        mock_settings.models = {"test-model-id": model_config}
        result = _build_router_entry("test-model-id")

        assert result is not None
        assert result["model_name"] == "test-model-id"
        # fallbacks are not added to router entry - they're handled by get_global_router
        # assert "fallbacks" in result["litellm_params"]
        # assert result["litellm_params"]["fallbacks"] == [
        #     "claude-haiku",
        #     "qwen-14b-instruct",
        # ]

    @patch("aichat.serve.backend.litellm.model_settings")
    def test_build_router_entry_with_weighted_deployment(self, mock_model_settings):
        """Test building router entry - weighted_deployment feature not yet implemented"""
        # Mock the models
        mock_model_settings.models = {
            "qwen-14b-instruct": {
                "upstream_model": "Qwen/Qwen3-14B",
                "address": "http://localhost:8885/v1",
                "inference_profile": None,
                "weighted_deployment": {"model": "qwen-3-235b", "weight": 0.2},
            },
            "qwen-3-235b": {
                "upstream_model": "profile-qwen3_235b",
                "address": None,
                "inference_profile": "INFERENCE_PROFILE_QWEN3_235B",
            },
        }

        with patch.dict(os.environ, {"INFERENCE_PROFILE_QWEN3_235B": "qwen-profile"}):
            result = _build_router_entry("qwen-14b-instruct")

            # weighted_deployment is not currently implemented
            # So it should return a single entry, not a list
            assert result is not None
            assert isinstance(result, dict)  # Single entry, not list
            assert result["model_name"] == "qwen-14b-instruct"
            assert result["litellm_params"]["model"] == "hosted_vllm/Qwen/Qwen3-14B"
            assert result["litellm_params"]["api_base"] == "http://localhost:8885/v1"

    @patch("aichat.serve.backend.litellm.model_settings")
    def test_build_router_entry_exception_handling(self, mock_settings):
        """Test that exceptions are caught and None is returned"""
        # Set up invalid model config that will cause KeyError
        mock_settings.models = {}  # model_id not in models dict
        result = _build_router_entry("nonexistent-model-id")

        assert result is None


class TestGetGlobalRouter:
    """Test cases for get_global_router fallback construction"""

    @patch("aichat.serve.backend.litellm._global_router", None)
    @patch("aichat.serve.backend.litellm.Router")
    @patch("aichat.serve.backend.litellm.model_settings")
    def test_fallbacks_built_as_single_key_dicts(self, mock_settings, mock_router):
        """litellm requires each fallbacks entry to be a single-key dict.

        Regression test: multiple models with fallbacks must not be collapsed
        into one multi-key dict (which raises ValueError in litellm).
        """
        mock_settings.models = {
            "claude-3-sonnet": {
                "backend": "bedrock",
                "type": "llm",
                "upstream_model": "sonnet",
                "address": None,
                "inference_profile": None,
                "fallback_models": ["claude-3-haiku"],
            },
            "claude-opus": {
                "backend": "bedrock",
                "type": "llm",
                "upstream_model": "opus",
                "address": None,
                "inference_profile": None,
                "fallback_models": ["claude-3-sonnet", "claude-3-haiku"],
            },
            "qwen-3-235b": {
                "backend": "bedrock",
                "type": "llm",
                "upstream_model": "qwen",
                "address": None,
                "inference_profile": None,
                "fallback_models": ["qwen-3-235b-bedrock"],
            },
            "claude-3-haiku": {
                "backend": "bedrock",
                "type": "llm",
                "upstream_model": "haiku",
                "address": None,
                "inference_profile": None,
            },
        }

        get_global_router()

        _, kwargs = mock_router.call_args
        fallbacks = kwargs["fallbacks"]

        # Every entry must be a dict with exactly one key.
        assert all(isinstance(entry, dict) and len(entry) == 1 for entry in fallbacks)

        # Flatten and verify the mappings are preserved.
        merged = {k: v for entry in fallbacks for k, v in entry.items()}
        assert merged == {
            "claude-3-sonnet": ["claude-3-haiku"],
            "claude-opus": ["claude-3-sonnet", "claude-3-haiku"],
            "qwen-3-235b": ["qwen-3-235b-bedrock"],
        }
