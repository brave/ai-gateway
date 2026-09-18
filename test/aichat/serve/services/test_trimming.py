"""
Tests for token trimming service.
"""

from unittest.mock import MagicMock, patch

from aichat.serve.services.trimming import (
    TRIMMABLE_CONTENT_TYPES,
    is_trimmable_content_part,
    maybe_trim_messages,
    trim_messages_to_fit,
    trim_tool_messages,
)
from aichat.serve.utils import calculate_message_tokens


class TestIsTrimmableContentPart:
    """Tests for is_trimmable_content_part function."""

    def test_trimmable_page_text(self):
        """PageText content parts are trimmable."""
        part = {"type": "brave-page-text", "text": "Some page content"}
        assert is_trimmable_content_part(part) is True

    def test_trimmable_page_excerpt(self):
        """PageExcerpt content parts are trimmable."""
        part = {"type": "brave-page-excerpt", "text": "Some excerpt"}
        assert is_trimmable_content_part(part) is True

    def test_trimmable_search_results(self):
        """SearchResults content parts are trimmable."""
        part = {"type": "brave-search-results", "text": "Search results"}
        assert is_trimmable_content_part(part) is True

    def test_trimmable_video_transcript(self):
        """VideoTranscript content parts are trimmable."""
        part = {"type": "brave-video-transcript", "text": "Transcript"}
        assert is_trimmable_content_part(part) is True

    def test_not_trimmable_text(self):
        """Regular text content parts are not trimmable."""
        part = {"type": "text", "text": "User message"}
        assert is_trimmable_content_part(part) is False

    def test_not_trimmable_image(self):
        """Image content parts are not trimmable."""
        part = {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}}
        assert is_trimmable_content_part(part) is False


class TestCalculateMessageTokens:
    """Tests for calculate_message_tokens function."""

    def test_calculate_simple_messages(self):
        """Returns a positive token count for basic messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        tokens = calculate_message_tokens(messages)
        assert tokens > 0

    def test_calculate_multimodal_messages(self):
        """Counts tokens from both text and custom content parts."""
        plain = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        with_page = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "brave-page-text", "text": "Page content here"},
                ],
            }
        ]
        assert calculate_message_tokens(with_page) > calculate_message_tokens(plain)

    def test_empty_messages(self):
        """Returns only overhead tokens for empty message list."""
        assert calculate_message_tokens([]) == 0


class TestTrimMessagesToFit:
    """Tests for trim_messages_to_fit function."""

    def test_no_trimming_when_under_limit(self):
        """No trimming when messages are under token limit."""
        messages = [{"role": "user", "content": "Short"}]

        trimmed, _total, trimmed_count = trim_messages_to_fit(
            messages=messages,
            max_input_tokens=100000,
        )

        assert trimmed_count == 0
        assert trimmed == messages

    def test_trims_page_text_when_over_limit(self):
        """Trims page text content when over token limit."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "brave-page-text", "text": "word " * 5000},
                ],
            }
        ]

        trimmed, _total, trimmed_count = trim_messages_to_fit(
            messages=messages,
            max_input_tokens=50,
        )

        assert trimmed_count > 0
        assert len(trimmed[0]["content"][1]["text"]) < len("word " * 5000)

    def test_trims_multiple_content_parts(self):
        """Trims multiple trimmable content parts in order."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "brave-page-text", "text": "word " * 2000},
                    {"type": "brave-page-excerpt", "text": "word " * 2000},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "brave-search-results", "text": "word " * 2000},
                ],
            },
        ]

        _trimmed, _total, trimmed_count = trim_messages_to_fit(
            messages=messages,
            max_input_tokens=100,
        )

        assert trimmed_count > 0

    def test_no_trimming_when_zero_limit(self):
        """Returns original messages unchanged when max_input_tokens is zero."""
        messages = [{"role": "user", "content": "Hello"}]

        trimmed, _total, trimmed_count = trim_messages_to_fit(
            messages=messages,
            max_input_tokens=0,
        )

        assert trimmed == messages
        assert trimmed_count == 0

    def test_pass_2_trims_tool_messages_when_pass_1_insufficient(self):
        """Tool message content is trimmed when multimodal trimming alone
        cannot bring the conversation within the limit."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "brave-page-text", "text": "page " * 500},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "tool " * 5000,
            },
        ]
        original_tool_content = messages[1]["content"]

        trimmed, _total, trimmed_count = trim_messages_to_fit(
            messages=messages,
            max_input_tokens=100,
        )

        assert trimmed_count > 0
        assert len(trimmed[1]["content"]) < len(original_tool_content)
        assert "tool response truncated" in trimmed[1]["content"]
        assert messages[1]["content"] == original_tool_content

    def test_pass_2_skipped_when_pass_1_sufficient(self):
        """Tool messages are left alone when multimodal trimming covers the excess."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "brave-page-text", "text": "page " * 5000},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "small tool response",
            },
        ]

        trimmed, _total, _trimmed_count = trim_messages_to_fit(
            messages=messages,
            max_input_tokens=200,
        )

        assert trimmed[1]["content"] == "small tool response"


class TestTrimToolMessages:
    """Tests for the standalone trim_tool_messages utility."""

    def test_no_trimming_when_under_budget(self):
        """Returns original list when total tokens fit within budget."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]

        out, trimmed = trim_tool_messages(messages, max_tokens=100_000)

        assert trimmed == 0
        assert out is messages

    def test_trims_tool_message_content(self):
        """Truncates oversized tool message string content."""
        big = "word " * 5000
        messages = [
            {"role": "user", "content": "summarise"},
            {"role": "tool", "tool_call_id": "call_1", "content": big},
        ]

        out, trimmed = trim_tool_messages(messages, max_tokens=100)

        assert trimmed > 0
        assert len(out[1]["content"]) < len(big)
        assert "tool response truncated" in out[1]["content"]
        assert messages[1]["content"] == big

    def test_leaves_non_tool_messages_alone(self):
        """Only role=tool messages are modified, even when other messages
        contain enough content to push the conversation over budget."""
        big_user_content = "word " * 5000
        big_tool_content = "word " * 5000
        messages = [
            {"role": "user", "content": big_user_content},
            {"role": "tool", "tool_call_id": "call_1", "content": big_tool_content},
        ]

        out, trimmed = trim_tool_messages(messages, max_tokens=200)

        assert out[0]["content"] == big_user_content
        assert trimmed > 0
        assert len(out[1]["content"]) < len(big_tool_content)

    def test_skips_when_max_tokens_zero(self):
        """Zero or negative budget is treated as a skip signal."""
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "word " * 1000},
        ]

        out, trimmed = trim_tool_messages(messages, max_tokens=0)

        assert out is messages
        assert trimmed == 0

    def test_trims_earliest_tool_messages_first(self):
        """Iterates oldest to newest; earlier tool messages take the hit first."""
        big = "word " * 3000
        messages = [
            {"role": "tool", "tool_call_id": "old", "content": big},
            {"role": "user", "content": "middle"},
            {"role": "tool", "tool_call_id": "new", "content": big},
        ]

        out, trimmed = trim_tool_messages(messages, max_tokens=4000)

        assert trimmed > 0
        assert len(out[0]["content"]) < len(big)
        assert out[2]["content"] == big


class TestMaybeTrimMessages:
    """Tests for maybe_trim_messages function."""

    def _make_model_config(self):
        cfg = MagicMock()
        cfg.conversation_token_limit = 100000
        cfg.conversation_token_limit_premium = 200000
        cfg.max_tokens = 4096
        return cfg

    def test_trims_when_needed(self):
        """Trims messages when they exceed token limit."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "brave-page-text", "text": "word " * 10000}],
            }
        ]

        with patch(
            "aichat.serve.services.trimming.trim_messages_to_fit",
            return_value=(messages, 50000, 49950),
        ):
            _trimmed, _total, trimmed_count = maybe_trim_messages(
                messages=messages,
                model_config=self._make_model_config(),
                is_premium=False,
            )

        assert trimmed_count > 0

    def test_no_trimming_when_under_limit(self):
        """Returns original messages unchanged when under token limit."""
        messages = [{"role": "user", "content": "Hello"}]

        with patch(
            "aichat.serve.services.trimming.trim_messages_to_fit",
            return_value=(messages, 10, 0),
        ):
            trimmed, _total, trimmed_count = maybe_trim_messages(
                messages=messages,
                model_config=self._make_model_config(),
                is_premium=False,
            )

        assert trimmed == messages
        assert trimmed_count == 0


class TestTrimmableContentTypes:
    """Tests for TRIMMABLE_CONTENT_TYPES constant."""

    def test_contains_expected_types(self):
        """Verify all expected types are in the set."""
        expected = {
            "brave-page-text",
            "brave-page-excerpt",
            "brave-search-results",
            "brave-video-transcript",
            "brave-file-extracted-text",
            "brave-pdf-text",
            "brave-request-summary",
        }
        assert TRIMMABLE_CONTENT_TYPES == expected
