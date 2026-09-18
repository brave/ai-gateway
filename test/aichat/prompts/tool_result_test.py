from unittest.mock import patch

from aichat.prompts.tool_result import tool_result


def test_tool_result_with_text_only():
    """Test that text-only tool messages are preserved."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {"name": "test_tool", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": [{"type": "text", "text": "Tool output text"}],
        },
    ]

    result = tool_result.augment(messages)

    assert len(result) == 2
    tool_msg = result[1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_123"
    assert "Tool output text" in tool_msg["content"][0]["text"]
    assert "<tool_output>" in tool_msg["content"][0]["text"]


def test_tool_result_with_web_sources():
    """Test that websources content is preserved and formatted."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {"name": "brave_web_search", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": [
                {
                    "type": "brave-chat.webSources",
                    "sources": [
                        {"title": "Test Source", "url": "https://test.com"},
                        {"title": "Example Source", "url": "https://example.com"},
                    ],
                    "query": "test query",
                }
            ],
        },
    ]

    with patch("aichat.prompts.tool_result._format_web_sources") as mock_format:
        mock_format.return_value = "Found 2 sources: Test Source, Example Source"
        result = tool_result.augment(messages)

        # Verify websources were formatted
        mock_format.assert_called_once_with(
            [
                {"title": "Test Source", "url": "https://test.com"},
                {"title": "Example Source", "url": "https://example.com"},
            ],
            "test query",
        )

        # Verify result includes formatted websources
        assert len(result) == 2
        tool_msg = result[1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_123"
        assert "Found 2 sources" in tool_msg["content"][0]["text"]
        assert "<tool_output>" in tool_msg["content"][0]["text"]


def test_tool_result_with_text_and_web_sources():
    """Test that both text and websources are preserved together."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {"name": "brave_web_search", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": [
                {"type": "text", "text": "Search completed."},
                {
                    "type": "brave-chat.webSources",
                    "sources": [{"title": "Test Source", "url": "https://test.com"}],
                    "query": "test query",
                },
            ],
        },
    ]

    with patch("aichat.prompts.tool_result._format_web_sources") as mock_format:
        mock_format.return_value = "Found 1 source: Test Source"
        result = tool_result.augment(messages)

        # Verify both text and websources are included
        assert len(result) == 2
        tool_msg = result[1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_123"
        content_text = tool_msg["content"][0]["text"]
        assert "Search completed." in content_text
        assert "Found 1 source" in content_text
        assert "<tool_output>" in content_text
        # Verify they are joined with double newline
        assert "\n\n" in content_text or content_text.count("Search completed.") == 1


def test_tool_result_with_empty_web_sources():
    """Test that empty websources fall back to text field if available."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {"name": "brave_web_search", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": [
                {
                    "type": "brave-chat.webSources",
                    "sources": [],
                    "text": "No sources found",
                }
            ],
        },
    ]

    result = tool_result.augment(messages)

    assert len(result) == 2
    tool_msg = result[1]
    assert tool_msg["role"] == "tool"
    assert "No sources found" in tool_msg["content"][0]["text"]


def test_tool_result_preserves_non_tool_messages():
    """Test that non-tool messages are passed through unchanged."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    result = tool_result.augment(messages)

    assert len(result) == 2
    assert result[0] == messages[0]
    assert result[1] == messages[1]


def test_tool_result_with_string_content():
    """Test that tool messages with string content are passed through."""
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "Simple string content",
        }
    ]

    result = tool_result.augment(messages)

    assert len(result) == 1
    assert result[0] == messages[0]
