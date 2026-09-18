"""Tests for DeepResearchServerHandler."""

import pytest

from aichat.serve.services.mcp.handlers.deep_research import DeepResearchServerHandler


@pytest.fixture
def handler():
    """Create a DeepResearchServerHandler instance."""
    return DeepResearchServerHandler()


class TestServerName:
    """Tests for server_name property."""

    def test_server_name_returns_deep_research(self, handler):
        """Test that server_name returns 'deep_research'."""
        assert handler.server_name == "deep_research"


class TestValidateResult:
    """Tests for validate_result method."""

    def test_valid_final_answer_event(self, handler):
        """Test validation of valid final answer event."""
        result = {
            "event": "answer",
            "final": True,
            "answer": "This is the research answer",
            "citations": [],
        }
        assert handler.validate_result(result) is True

    def test_non_final_answer_event(self, handler):
        """Test validation of non-final answer event."""
        result = {
            "event": "answer",
            "final": False,
            "answer": "Partial answer",
        }
        # Non-final answer events should still pass if they have 'answer' key
        assert handler.validate_result(result) is True

    def test_result_with_content_field(self, handler):
        """Test validation of result with content field."""
        result = {
            "content": "Some research content",
        }
        assert handler.validate_result(result) is True

    def test_result_with_answer_field(self, handler):
        """Test validation of result with answer field."""
        result = {
            "answer": "Some answer text",
        }
        assert handler.validate_result(result) is True

    def test_non_dict_input(self, handler):
        """Test validation of non-dict input returns False."""
        assert handler.validate_result("string input") is False
        assert handler.validate_result(123) is False
        assert handler.validate_result(["list", "input"]) is False
        assert handler.validate_result(None) is False

    def test_empty_dict(self, handler):
        """Test validation of empty dict returns False."""
        assert handler.validate_result({}) is False

    def test_missing_answer_key_in_final_event(self, handler):
        """Test validation when answer key is missing from final event."""
        result = {
            "event": "answer",
            "final": True,
            # Missing 'answer' key
        }
        assert handler.validate_result(result) is False

    def test_final_answer_with_citations(self, handler):
        """Test validation of final answer with citations."""
        result = {
            "event": "answer",
            "final": True,
            "answer": "Answer with citations",
            "citations": [{"url": "https://example.com", "snippet": "Source snippet"}],
        }
        assert handler.validate_result(result) is True


class TestFormatResult:
    """Tests for format_result method."""

    def test_format_result_with_citations(self, handler):
        """Test formatting result with citations."""
        result = {
            "answer": "This is the research answer.",
            "citations": [
                {
                    "url": "https://example.com/1",
                    "snippet": "First source snippet",
                    "number": 1,
                    "favicon": "https://example.com/favicon.ico",
                },
                {
                    "url": "https://example.com/2",
                    "snippet": "Second source snippet",
                    "number": 2,
                },
            ],
        }

        formatted = handler.format_result("deep_research", result)

        assert formatted["type"] == "brave-deep-research-result"
        assert formatted["tool_name"] == "deep_research"
        assert formatted["answer"] == "This is the research answer."
        assert len(formatted["citations"]) == 2
        assert len(formatted["sources"]) == 2
        assert "content" in formatted
        assert "Sources:" in formatted["content"]
        assert "[1] https://example.com/1" in formatted["content"]
        assert "[2] https://example.com/2" in formatted["content"]

    def test_format_result_without_citations(self, handler):
        """Test formatting result without citations."""
        result = {
            "answer": "Answer without citations.",
        }

        formatted = handler.format_result("deep_research", result)

        assert formatted["type"] == "brave-deep-research-result"
        assert formatted["tool_name"] == "deep_research"
        assert formatted["answer"] == "Answer without citations."
        assert formatted["citations"] == []
        assert formatted["sources"] == []
        assert "Sources:" not in formatted["content"]

    def test_format_result_citation_field_extraction(self, handler):
        """Test that citation fields are correctly extracted."""
        result = {
            "answer": "Answer text",
            "citations": [
                {
                    "url": "https://test.com",
                    "snippet": "Test snippet",
                    "number": 5,
                    "favicon": "https://test.com/icon.png",
                },
            ],
        }

        formatted = handler.format_result("deep_research", result)

        source = formatted["sources"][0]
        assert source["url"] == "https://test.com"
        assert source["title"] == "Test snippet"[:100]
        assert source["favicon"] == "https://test.com/icon.png"

    def test_format_result_snippet_truncation(self, handler):
        """Test that long snippets are truncated in formatted output."""
        long_snippet = "x" * 300  # Longer than 200 chars
        result = {
            "answer": "Answer",
            "citations": [
                {
                    "url": "https://test.com",
                    "snippet": long_snippet,
                    "number": 1,
                },
            ],
        }

        formatted = handler.format_result("deep_research", result)

        # In content, snippet should be truncated to 200 chars with ...
        assert f"    {long_snippet[:200]}..." in formatted["content"]

        # In sources title, snippet should be truncated to 100 chars
        assert formatted["sources"][0]["title"] == long_snippet[:100]

    def test_format_result_missing_number_fallback(self, handler):
        """Test that missing citation number falls back to index + 1."""
        result = {
            "answer": "Answer",
            "citations": [
                {
                    "url": "https://first.com",
                    "snippet": "First",
                    # No 'number' key
                },
                {
                    "url": "https://second.com",
                    "snippet": "Second",
                    # No 'number' key
                },
            ],
        }

        formatted = handler.format_result("deep_research", result)

        assert "[1] https://first.com" in formatted["content"]
        assert "[2] https://second.com" in formatted["content"]

    def test_format_result_empty_url_handling(self, handler):
        """Test formatting when citation URL is empty."""
        result = {
            "answer": "Answer",
            "citations": [
                {
                    "url": "",
                    "snippet": "Snippet with no URL",
                    "number": 1,
                },
            ],
        }

        formatted = handler.format_result("deep_research", result)

        # Should still include the citation but with empty URL
        assert formatted["sources"][0]["url"] == ""
        assert "[1] " in formatted["content"]

    def test_format_result_uses_content_fallback(self, handler):
        """Test that content is used when answer is not present."""
        result = {
            "content": "Content as fallback",
        }

        formatted = handler.format_result("deep_research", result)

        assert formatted["answer"] == "Content as fallback"
        assert "Content as fallback" in formatted["content"]

    def test_format_result_missing_snippet_uses_source_n(self, handler):
        """Test that missing snippet uses 'Source N' as title."""
        result = {
            "answer": "Answer",
            "citations": [
                {
                    "url": "https://example.com",
                    "number": 3,
                    # No snippet
                },
            ],
        }

        formatted = handler.format_result("deep_research", result)

        assert formatted["sources"][0]["title"] == "Source 3"


class TestGetToolGuidance:
    """Tests for get_tool_guidance method."""

    def test_returns_dict_with_deep_research_key(self, handler):
        """Test that guidance returns dict with 'deep_research' key."""
        guidance = handler.get_tool_guidance()
        assert isinstance(guidance, dict)
        assert "deep_research" in guidance

    def test_guidance_includes_use_cases(self, handler):
        """Test that guidance includes use cases."""
        guidance = handler.get_tool_guidance()
        deep_research_guidance = guidance["deep_research"]

        # Check for use case indicators (trigger phrases or explicit conditions)
        assert any(
            phrase in deep_research_guidance
            for phrase in (
                "Use this tool when",
                "only when",
                "explicitly requests",
                "ONLY use when",
            )
        )
        assert any(
            phrase in deep_research_guidance.lower()
            for phrase in (
                "complex",
                "research-style",
                "research report",
                "deep research",
            )
        )

    def test_guidance_includes_examples(self, handler):
        """Test that guidance includes examples."""
        guidance = handler.get_tool_guidance()
        deep_research_guidance = guidance["deep_research"]

        # Check for example indicators
        assert "Examples" in deep_research_guidance
        assert "quantum computing" in deep_research_guidance.lower()

    def test_guidance_includes_do_not_use(self, handler):
        """Test that guidance includes when not to use the tool."""
        guidance = handler.get_tool_guidance()
        deep_research_guidance = guidance["deep_research"]

        assert "Do NOT use" in deep_research_guidance or "NOT" in deep_research_guidance


class TestGetAugmentedTools:
    """Tests for get_augmented_tools method."""

    def test_returns_empty_list(self, handler):
        """Test that get_augmented_tools returns empty list."""
        augmented = handler.get_augmented_tools()
        assert augmented == []
        assert isinstance(augmented, list)
