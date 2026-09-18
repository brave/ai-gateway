from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aichat.protocol.open_ai_protocol import (
    AssistantMessage,
    Capability,
    FilterTabsContentPart,
    ImageContentPart,
    ReduceFocusTopicsContentPart,
    SuggestFocusTopicsContentPart,
    SuggestFocusTopicsWithEmojiContentPart,
    TextContentPart,
    UserMessage,
)
from aichat.serve.androcles import ANDROCLES_TRIAGE_INDICES
from aichat.serve.services.androcles_prefetch import AndroclesPrefetch
from aichat.serve.services.model_selection import (
    classify_task_with_androcles,
    is_long_context,
    select_model_for_request,
    triage_request,
)

_ANDROCLES_MOCK_VEC_LEN = max(ANDROCLES_TRIAGE_INDICES.values()) + 1


class TestSelectModelForRequest:
    @pytest.fixture
    def mock_messages(self):
        """Mock messages for testing"""
        return [
            UserMessage(content="Hello, how are you?"),
            AssistantMessage(content="I'm doing well, thank you!"),
        ]

    @pytest.fixture
    def mock_model_settings(self):
        """Mock app settings"""
        settings = MagicMock()
        settings.content_agent_default_model = "content-agent-model"
        settings.model_triaging = {
            "default": {"premium": "premium-model", "non-premium": "free-model"},
            "vision": {"premium": "vision-premium", "non-premium": "vision-free"},
            "coding": {"premium": "coding-premium", "non-premium": "coding-free"},
            "language": {"premium": "language-premium", "non-premium": "language-free"},
            "long_context": {"premium": "long-premium", "non-premium": "long-free"},
        }
        return settings

    @pytest.mark.asyncio
    async def test_content_agent_capability_unsupported_model_falls_back(
        self, mock_messages, mock_model_settings
    ):
        """Test that content_agent capability falls back to default when model flag is False"""
        mock_cfg = MagicMock()
        mock_cfg.content_agent_support = False

        with (
            patch(
                "aichat.serve.services.model_selection.model_settings",
                mock_model_settings,
            ),
            patch(
                "aichat.serve.services.model_selection.get_model_config",
                return_value=mock_cfg,
            ),
        ):
            result = await select_model_for_request(
                model="some-model",
                messages=mock_messages,
                brave_capability=Capability.content_agent,
                is_premium=True,
            )
            assert result == "content-agent-model"

    @pytest.mark.asyncio
    async def test_content_agent_capability_supported_model_passes_through(
        self, mock_messages, mock_model_settings
    ):
        """Test that content_agent capability passes through when model flag is True"""
        mock_cfg = MagicMock()
        mock_cfg.content_agent_support = True
        mock_model_settings.models = {"supported-model": {}}

        with (
            patch(
                "aichat.serve.services.model_selection.model_settings",
                mock_model_settings,
            ),
            patch(
                "aichat.serve.services.model_selection.get_model_config",
                return_value=mock_cfg,
            ),
        ):
            result = await select_model_for_request(
                model="supported-model",
                messages=mock_messages,
                brave_capability=Capability.content_agent,
                is_premium=True,
            )
            assert result == "supported-model"

    @pytest.mark.asyncio
    async def test_content_agent_capability_unknown_model_falls_back(
        self, mock_messages, mock_model_settings
    ):
        """Test that content_agent capability falls back when get_model_config returns None"""
        with (
            patch(
                "aichat.serve.services.model_selection.model_settings",
                mock_model_settings,
            ),
            patch(
                "aichat.serve.services.model_selection.get_model_config",
                return_value=None,
            ),
        ):
            result = await select_model_for_request(
                model="unknown-model",
                messages=mock_messages,
                brave_capability=Capability.content_agent,
                is_premium=True,
            )
            assert result == "content-agent-model"

    @pytest.mark.asyncio
    async def test_content_agent_capability_automatic_model_falls_back(
        self, mock_messages, mock_model_settings
    ):
        """Test that content_agent with automatic model falls back to default without raising"""
        mock_model_settings.models = {}
        with patch(
            "aichat.serve.services.model_selection.model_settings", mock_model_settings
        ):
            result = await select_model_for_request(
                model="automatic",
                messages=mock_messages,
                brave_capability=Capability.content_agent,
                is_premium=True,
            )
            assert result == "content-agent-model"

    @pytest.mark.asyncio
    async def test_non_automatic_model(self, mock_messages, mock_model_settings):
        """Test that supported non-automatic model returns the same model"""
        # Add specific-model to the supported models
        mock_model_settings.models = {"specific-model": {"free": True}}

        with patch(
            "aichat.serve.services.model_selection.model_settings", mock_model_settings
        ):
            result = await select_model_for_request(
                model="specific-model",
                messages=mock_messages,
                brave_capability=None,
                is_premium=True,
            )
            assert result == "specific-model"

    @pytest.mark.asyncio
    async def test_premium_user_gets_premium_model(
        self, mock_messages, mock_model_settings
    ):
        """Test that premium users get premium models"""
        with (
            patch(
                "aichat.serve.services.model_selection.model_settings",
                mock_model_settings,
            ),
            patch(
                "aichat.serve.services.model_selection.triage_request"
            ) as mock_triage,
        ):
            mock_triage.return_value = {
                "premium": "premium-model",
                "non-premium": "free-model",
            }

            result = await select_model_for_request(
                model="automatic",
                messages=mock_messages,
                brave_capability=None,
                is_premium=True,
            )
            assert result == "premium-model"

    @pytest.mark.asyncio
    async def test_non_premium_user_gets_free_model(
        self, mock_messages, mock_model_settings
    ):
        """Test that non-premium users get free models"""
        with (
            patch(
                "aichat.serve.services.model_selection.model_settings",
                mock_model_settings,
            ),
            patch(
                "aichat.serve.services.model_selection.triage_request"
            ) as mock_triage,
            patch(
                "aichat.serve.services.model_selection.check_and_increment_automatic_mode_daily_count",
                new_callable=AsyncMock,
            ) as mock_daily_count,
        ):
            mock_daily_count.return_value = False
            mock_triage.return_value = {
                "premium": "premium-model",
                "non-premium": "free-model",
            }

            result = await select_model_for_request(
                model="automatic",
                messages=mock_messages,
                brave_capability=None,
                is_premium=False,
            )
            assert result == "free-model"


class TestTriageRequest:
    @pytest.fixture
    def mock_messages(self):
        """Mock messages for testing"""
        return [
            UserMessage(content="Hello, how are you?"),
        ]

    @pytest.fixture
    def mock_model_settings(self):
        """Mock app settings"""
        settings = MagicMock()
        settings.model_triaging = {
            "default": {"premium": "default-premium", "non-premium": "default-free"},
            "vision": {"premium": "vision-premium", "non-premium": "vision-free"},
            "coding": {"premium": "coding-premium", "non-premium": "coding-free"},
            "language": {"premium": "language-premium", "non-premium": "language-free"},
            "long_context": {"premium": "long-premium", "non-premium": "long-free"},
        }
        return settings

    @pytest.mark.asyncio
    async def test_media_type_takes_priority(self, mock_messages, mock_model_settings):
        """Test that media type detection takes priority over other checks"""
        with patch(
            "aichat.serve.services.model_selection.model_settings", mock_model_settings
        ):
            result = await triage_request(
                messages=mock_messages,
                media_type="vision",
            )
            assert result == {"premium": "vision-premium", "non-premium": "vision-free"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content_part",
        [
            FilterTabsContentPart(type="brave-filter-tabs", topic="news", text="tabs"),
            ReduceFocusTopicsContentPart(
                type="brave-reduce-focus-topics", text="topics"
            ),
            SuggestFocusTopicsContentPart(
                type="brave-suggest-focus-topics", text="tabs"
            ),
            SuggestFocusTopicsWithEmojiContentPart(
                type="brave-suggest-focus-topics-emoji", text="tabs"
            ),
        ],
    )
    async def test_tab_focus_routes_to_tab_focus_model(
        self, mock_model_settings, content_part
    ):
        """Test that tab focus content parts route to tab_focus models"""
        mock_model_settings.model_triaging["tab_focus"] = {
            "premium": "tab-focus-premium",
            "non-premium": "tab-focus-free",
        }
        messages = [
            UserMessage(content=[content_part]),
        ]
        with patch(
            "aichat.serve.services.model_selection.model_settings", mock_model_settings
        ):
            result = await triage_request(messages=messages)
            assert result == {
                "premium": "tab-focus-premium",
                "non-premium": "tab-focus-free",
            }

    @pytest.mark.asyncio
    async def test_long_context_detection(self, mock_model_settings):
        """Test that long context messages trigger long context model"""
        long_messages = [
            UserMessage(content="x" * 10000),  # Very long message
        ]

        with (
            patch(
                "aichat.serve.services.model_selection.model_settings",
                mock_model_settings,
            ),
            patch(
                "aichat.serve.services.model_selection.is_long_context"
            ) as mock_long_context,
        ):
            mock_long_context.return_value = True

            result = await triage_request(
                messages=long_messages,
            )
            assert result == {"premium": "long-premium", "non-premium": "long-free"}

    @pytest.mark.asyncio
    async def test_androcles_classification(self, mock_messages, mock_model_settings):
        """Test that Androcles classification works"""
        with (
            patch(
                "aichat.serve.services.model_selection.model_settings",
                mock_model_settings,
            ),
            patch(
                "aichat.serve.services.model_selection.is_long_context"
            ) as mock_long_context,
        ):
            mock_long_context.return_value = False

            with patch(
                "aichat.serve.services.model_selection.classify_task_with_androcles"
            ) as mock_classify:
                mock_classify.return_value = "coding"

                result = await triage_request(
                    messages=mock_messages,
                    last_user_message_content="Write a Python function",
                )
                assert result == {
                    "premium": "coding-premium",
                    "non-premium": "coding-free",
                }

    @pytest.mark.asyncio
    async def test_default_fallback(self, mock_messages, mock_model_settings):
        """Test that default model is returned when no other conditions match"""
        with (
            patch(
                "aichat.serve.services.model_selection.model_settings",
                mock_model_settings,
            ),
            patch(
                "aichat.serve.services.model_selection.is_long_context"
            ) as mock_long_context,
        ):
            mock_long_context.return_value = False

            with patch(
                "aichat.serve.services.model_selection.classify_task_with_androcles"
            ) as mock_classify:
                mock_classify.return_value = None

                result = await triage_request(
                    messages=mock_messages,
                    last_user_message_content="Regular question",
                )
                assert result == {
                    "premium": "default-premium",
                    "non-premium": "default-free",
                }

    @pytest.mark.asyncio
    async def test_androcles_prefetch_with_task_skips_classifier(
        self, mock_messages, mock_model_settings
    ):
        with (
            patch(
                "aichat.serve.services.model_selection.model_settings",
                mock_model_settings,
            ),
            patch(
                "aichat.serve.services.model_selection.is_long_context"
            ) as mock_long_context,
        ):
            mock_long_context.return_value = False

            with patch(
                "aichat.serve.services.model_selection.classify_task_with_androcles"
            ) as mock_classify:
                result = await triage_request(
                    messages=mock_messages,
                    last_user_message_content="Write a Python function",
                    androcles_prefetch=AndroclesPrefetch(task_type="coding"),
                )
                mock_classify.assert_not_called()

                assert result == {
                    "premium": "coding-premium",
                    "non-premium": "coding-free",
                }

    @pytest.mark.asyncio
    async def test_androcles_prefetch_none_task_returns_default_without_classifier(
        self, mock_messages, mock_model_settings
    ):
        with (
            patch(
                "aichat.serve.services.model_selection.model_settings",
                mock_model_settings,
            ),
            patch(
                "aichat.serve.services.model_selection.is_long_context"
            ) as mock_long_context,
        ):
            mock_long_context.return_value = False

            with patch(
                "aichat.serve.services.model_selection.classify_task_with_androcles"
            ) as mock_classify:
                result = await triage_request(
                    messages=mock_messages,
                    last_user_message_content="Regular question",
                    androcles_prefetch=AndroclesPrefetch(task_type=None),
                )
                mock_classify.assert_not_called()

                assert result == {
                    "premium": "default-premium",
                    "non-premium": "default-free",
                }


class TestIsLongContext:
    @pytest.fixture
    def mock_conversation_settings(self):
        """Mock conversation settings"""
        settings = MagicMock()
        settings.conversation_token_limit_for_long_context = 1000
        return settings

    @pytest.mark.asyncio
    async def test_short_context_returns_false(self, mock_conversation_settings):
        """Test that short messages return False"""
        short_messages = [
            UserMessage(content="Hello"),
            AssistantMessage(content="Hi there!"),
        ]

        with patch(
            "aichat.serve.services.model_selection.conversation_settings",
            mock_conversation_settings,
        ):
            result = await is_long_context(short_messages)
            assert result is False

    @pytest.mark.asyncio
    async def test_long_context_returns_true(self, mock_conversation_settings):
        """Test that long messages return True"""
        long_messages = [
            UserMessage(content="x" * 5000),  # Very long message
        ]

        with patch(
            "aichat.serve.services.model_selection.conversation_settings",
            mock_conversation_settings,
        ):
            result = await is_long_context(long_messages)
            assert result is True

    @pytest.mark.asyncio
    async def test_mixed_content_types(self):
        """Test handling of different content types"""
        # Test with messages that exceed the default threshold of 6400 tokens
        # 20000 chars = 6666 tokens, which is > 6400
        simple_messages = [
            UserMessage(content="Short text"),
            UserMessage(content="x" * 20000),  # Long text
        ]

        result = await is_long_context(simple_messages)
        assert result is True

    @pytest.mark.asyncio
    async def test_text_content_parts(self):
        """Test handling of TextContentPart objects"""
        # Test with messages that exceed the default threshold of 6400 tokens
        # 20000 chars + text parts = 6666+ tokens, which is > 6400
        mixed_messages = [
            UserMessage(content="Short text"),
            UserMessage(
                content=[
                    TextContentPart(text="Part 1"),
                    TextContentPart(text="Part 2"),
                ],
            ),
            UserMessage(content="x" * 20000),  # Long text
        ]

        result = await is_long_context(mixed_messages)
        assert result is True

    @pytest.mark.asyncio
    async def test_empty_content_handling(self, mock_conversation_settings):
        """Test handling of empty or None content"""
        empty_messages = [
            UserMessage(content=""),
            AssistantMessage(content=None),
            UserMessage(content="   "),  # Whitespace only
        ]

        with patch(
            "aichat.serve.services.model_selection.conversation_settings",
            mock_conversation_settings,
        ):
            result = await is_long_context(empty_messages)
            assert result is False

    @pytest.mark.asyncio
    async def test_token_estimation_calculation(self):
        """Test that token estimation uses 3:1 ratio"""
        # The default threshold is 6400 tokens
        # 20000 characters / 3 = 6666 tokens, which is > 6400
        messages = [UserMessage(content="x" * 20000)]

        result = await is_long_context(messages)
        assert result is True

    @pytest.mark.asyncio
    async def test_below_threshold(self):
        """Test that messages below threshold return False"""
        # The default threshold is 6400 tokens
        # 10000 characters / 3 = 3333 tokens, which is < 6400
        messages = [UserMessage(content="x" * 10000)]

        result = await is_long_context(messages)
        assert result is False


class TestClassifyTaskWithAndrocles:
    @pytest.mark.asyncio
    async def test_successful_classification(self):
        mock_androcles_result = [0.02] * _ANDROCLES_MOCK_VEC_LEN
        mock_androcles_result[10] = 0.95

        with patch(
            "aichat.serve.services.model_selection.androcles_inference"
        ) as mock_androcles:
            mock_androcles.return_value = mock_androcles_result

            result = await classify_task_with_androcles("Translate this text")
            assert result == "language"

    @pytest.mark.asyncio
    async def test_wrong_output_dimension_returns_none(self):
        mock_androcles_result = [0.1, 0.1, 0.95, 0.1]

        with patch(
            "aichat.serve.services.model_selection.androcles_inference"
        ) as mock_androcles:
            mock_androcles.return_value = mock_androcles_result

            result = await classify_task_with_androcles("Translate this text")
            assert result is None

    @pytest.mark.asyncio
    async def test_coding_task_classification(self):
        mock_androcles_result = [0.02] * _ANDROCLES_MOCK_VEC_LEN
        mock_androcles_result[3] = 0.95

        with patch(
            "aichat.serve.services.model_selection.androcles_inference"
        ) as mock_androcles:
            mock_androcles.return_value = mock_androcles_result

            result = await classify_task_with_androcles("Write a Python function")
            assert result == "coding"

    @pytest.mark.asyncio
    async def test_vision_task_classification(self):
        mock_androcles_result = [0.02] * _ANDROCLES_MOCK_VEC_LEN
        mock_androcles_result[8] = 0.95

        with patch(
            "aichat.serve.services.model_selection.androcles_inference"
        ) as mock_androcles:
            mock_androcles.return_value = mock_androcles_result

            result = await classify_task_with_androcles("Describe this image")
            assert result == "vision"

    @pytest.mark.asyncio
    async def test_below_threshold_returns_none(self):
        mock_androcles_result = [0.5] * _ANDROCLES_MOCK_VEC_LEN

        with patch(
            "aichat.serve.services.model_selection.androcles_inference"
        ) as mock_androcles:
            mock_androcles.return_value = mock_androcles_result

            result = await classify_task_with_androcles("Regular question")
            assert result is None

    @pytest.mark.asyncio
    async def test_androcles_returns_none(self):
        """Test handling when Androcles returns None"""
        with patch(
            "aichat.serve.services.model_selection.androcles_inference"
        ) as mock_androcles:
            mock_androcles.return_value = None

            result = await classify_task_with_androcles("Test content")
            assert result is None

    @pytest.mark.asyncio
    async def test_http_error_handling(self):
        """Test that androcles_inference returns None when HTTP errors occur (handled internally)"""
        with patch(
            "aichat.serve.services.model_selection.androcles_inference"
        ) as mock_androcles:
            # androcles_inference catches exceptions internally and returns None
            mock_androcles.return_value = None

            result = await classify_task_with_androcles("Test content")
            assert result is None

    @pytest.mark.asyncio
    async def test_json_decode_error_handling(self):
        """Test that androcles_inference returns None when JSON decode errors occur (handled internally)"""
        with patch(
            "aichat.serve.services.model_selection.androcles_inference"
        ) as mock_androcles:
            # androcles_inference catches exceptions internally and returns None
            mock_androcles.return_value = None

            result = await classify_task_with_androcles("Test content")
            assert result is None

    @pytest.mark.asyncio
    async def test_timeout_exception_handling(self):
        """Test that androcles_inference returns None when timeout exceptions occur (handled internally)"""
        with patch(
            "aichat.serve.services.model_selection.androcles_inference"
        ) as mock_androcles:
            # androcles_inference catches exceptions internally and returns None
            mock_androcles.return_value = None

            result = await classify_task_with_androcles("Test content")
            assert result is None

    @pytest.mark.asyncio
    async def test_request_error_handling(self):
        """Test that androcles_inference returns None when request errors occur (handled internally)"""
        with patch(
            "aichat.serve.services.model_selection.androcles_inference"
        ) as mock_androcles:
            # androcles_inference catches exceptions internally and returns None
            mock_androcles.return_value = None

            result = await classify_task_with_androcles("Test content")
            assert result is None


class TestIntegrationScenarios:
    """Integration tests that combine multiple functions"""

    @pytest.fixture
    def mock_model_settings(self):
        """Mock app settings for integration tests"""
        settings = MagicMock()
        settings.content_agent_default_model = "content-agent-model"
        settings.model_triaging = {
            "default": {"premium": "default-premium", "non-premium": "default-free"},
            "vision": {"premium": "vision-premium", "non-premium": "vision-free"},
            "coding": {"premium": "coding-premium", "non-premium": "coding-free"},
            "language": {"premium": "language-premium", "non-premium": "language-free"},
            "long_context": {"premium": "long-premium", "non-premium": "long-free"},
        }
        return settings

    @pytest.mark.asyncio
    async def test_coding_request_flow(self, mock_model_settings):
        """Test complete flow for a coding request"""
        coding_messages = [
            UserMessage(content="Write a Python function to sort a list"),
        ]

        mock_androcles_result = [0.02] * _ANDROCLES_MOCK_VEC_LEN
        mock_androcles_result[3] = 0.95

        with (
            patch(
                "aichat.serve.services.model_selection.model_settings",
                mock_model_settings,
            ),
            patch(
                "aichat.serve.services.model_selection.androcles_inference"
            ) as mock_androcles,
            patch(
                "aichat.serve.services.model_selection.check_and_increment_automatic_mode_daily_count",
                new_callable=AsyncMock,
            ) as mock_daily_count,
        ):
            mock_daily_count.return_value = False
            mock_androcles.return_value = mock_androcles_result

            result = await select_model_for_request(
                model="automatic",
                messages=coding_messages,
                brave_capability=None,
                is_premium=False,
                last_user_message_content="Write a Python function to sort a list",
            )
            assert result == "coding-free"

    @pytest.mark.asyncio
    async def test_vision_request_flow(self, mock_model_settings):
        """Test complete flow for a vision request"""
        vision_messages = [
            UserMessage(
                content=[
                    ImageContentPart(
                        image_url={"url": "data:image/jpeg;base64,dGVzdA=="},
                        type="image_url",
                    )
                ],
            ),
        ]

        with patch(
            "aichat.serve.services.model_selection.model_settings", mock_model_settings
        ):
            result = await select_model_for_request(
                model="automatic",
                messages=vision_messages,
                brave_capability=None,
                is_premium=True,
                media_type="vision",
            )
            assert result == "vision-premium"

    @pytest.mark.asyncio
    async def test_long_context_request_flow(self, mock_model_settings):
        """Test complete flow for a long context request"""
        long_messages = [
            UserMessage(content="x" * 5000),  # Very long message
        ]

        mock_conv_settings = MagicMock()
        mock_conv_settings.conversation_token_limit_for_long_context = 1000

        with (
            patch(
                "aichat.serve.services.model_selection.model_settings",
                mock_model_settings,
            ),
            patch(
                "aichat.serve.services.model_selection.conversation_settings",
                mock_conv_settings,
            ),
        ):
            result = await select_model_for_request(
                model="automatic",
                messages=long_messages,
                brave_capability=None,
                is_premium=True,
            )
            assert result == "long-premium"
