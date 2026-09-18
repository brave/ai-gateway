"""Tests for search handler with page_content and extra_snippets."""

import os

import pytest

from aichat.responses import WebSource

# Set minimal env vars to avoid validation errors
os.environ.setdefault("RATE_LIMITING_ENABLED", "false")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("NEAR_API_KEY", "test_near_key")


@pytest.fixture
def search_handler():
    """Create a search handler instance for testing."""
    # Import inside fixture to avoid config issues during collection
    from aichat.serve.services.mcp.handlers.search import SearchServerHandler

    return SearchServerHandler()


def test_web_source_with_page_content_and_snippets():
    """Test WebSource model with new fields."""
    source = WebSource(
        title="Test Page",
        url="https://example.com",
        favicon="https://example.com/favicon.ico",
        page_content="Full page content here",
        extra_snippets=["snippet 1", "snippet 2"],
    )

    assert source.title == "Test Page"
    assert source.url == "https://example.com"
    assert source.favicon == "https://example.com/favicon.ico"
    assert source.page_content == "Full page content here"
    assert source.extra_snippets == ["snippet 1", "snippet 2"]


def test_web_source_optional_fields():
    """Test WebSource model with optional fields as None."""
    source = WebSource(
        title="Test Page",
        url="https://example.com",
    )

    assert source.title == "Test Page"
    assert source.url == "https://example.com"
    assert source.favicon is None
    assert source.page_content is None
    assert source.extra_snippets is None


def test_format_web_sources_content_basic(search_handler):
    """Test formatting web sources with basic fields."""
    handler = search_handler

    sources = [
        {
            "title": "First Result",
            "url": "https://first.com",
            "snippet": "First snippet",
        },
        {
            "title": "Second Result",
            "url": "https://second.com",
            "snippet": "Second snippet",
        },
    ]

    result = handler.format_web_sources_content(sources)

    assert "Found 2 search results" in result
    assert "First Result" in result
    assert "Second Result" in result
    assert "https://first.com" in result
    assert "https://second.com" in result
    assert "First snippet" in result
    assert "Second snippet" in result


def test_format_web_sources_content_with_page_content(search_handler):
    """Test formatting web sources with page content."""
    handler = search_handler

    sources = [
        {
            "title": "Test Page",
            "url": "https://test.com",
            "snippet": "Brief snippet",
            "page_content": "This is the full page content",
        },
    ]

    result = handler.format_web_sources_content(sources)

    assert "Found 1 search result" in result
    assert "Test Page" in result
    assert "Brief snippet" in result
    assert "Full page content" in result
    assert "This is the full page content" in result


def test_format_web_sources_content_with_extra_snippets(search_handler):
    """Test formatting web sources with extra snippets."""
    handler = search_handler

    sources = [
        {
            "title": "Test Page",
            "url": "https://test.com",
            "snippet": "Main snippet",
            "extra_snippets": [
                "Additional snippet 1",
                "Additional snippet 2",
            ],
        },
    ]

    result = handler.format_web_sources_content(sources)

    assert "Found 1 search result" in result
    assert "Test Page" in result
    assert "Main snippet" in result
    assert "Additional snippets:" in result
    assert "Additional snippet 1" in result
    assert "Additional snippet 2" in result


def test_format_web_sources_content_with_all_fields(search_handler):
    """Test formatting with all fields including page_content and snippets."""
    handler = search_handler

    sources = [
        {
            "title": "Complete Page",
            "url": "https://complete.com",
            "snippet": "Main description",
            "extra_snippets": ["Extra 1", "Extra 2"],
            "page_content": "Full text of the page",
        },
    ]

    result = handler.format_web_sources_content(sources)

    assert "Found 1 search result" in result
    assert "Complete Page" in result
    assert "Main description" in result
    assert "Additional snippets:" in result
    assert "Extra 1" in result
    assert "Extra 2" in result
    assert "Full page content" in result
    assert "Full text of the page" in result


def test_format_web_sources_content_empty_sources(search_handler):
    """Test formatting with empty sources list."""
    handler = search_handler

    result = handler.format_web_sources_content([])

    assert result == "Found 0 search results"


def test_format_web_sources_content_with_query(search_handler):
    """Test formatting web sources with query parameter."""
    handler = search_handler

    sources = [
        {
            "title": "Result",
            "url": "https://result.com",
            "snippet": "Content",
        },
    ]

    result = handler.format_web_sources_content(sources, query="test query")

    # Query is passed but main test is that formatting works
    assert "Found 1 search result" in result
    assert "Result" in result


def test_format_web_sources_content_multiple_sources_summary(search_handler):
    """Test that summary shows first 3 titles."""
    handler = search_handler

    sources = [
        {"title": "First", "url": "https://1.com"},
        {"title": "Second", "url": "https://2.com"},
        {"title": "Third", "url": "https://3.com"},
        {"title": "Fourth", "url": "https://4.com"},
        {"title": "Fifth", "url": "https://5.com"},
    ]

    result = handler.format_web_sources_content(sources)

    assert "Found 5 search results" in result
    assert "First, Second, Third" in result
    assert "and 2 more" in result


def test_format_web_sources_content_handles_missing_snippet(search_handler):
    """Test formatting when snippet is missing."""
    handler = search_handler

    sources = [
        {
            "title": "No Snippet Page",
            "url": "https://nosnippet.com",
            # No snippet or description field
        },
    ]

    result = handler.format_web_sources_content(sources)

    assert "Found 1 search result" in result
    assert "No Snippet Page" in result
    assert "https://nosnippet.com" in result


def test_format_web_sources_content_uses_description_fallback(search_handler):
    """Test that description is used as fallback for snippet."""
    handler = search_handler

    sources = [
        {
            "title": "Test Page",
            "url": "https://test.com",
            "description": "Description text",  # No snippet, has description
        },
    ]

    result = handler.format_web_sources_content(sources)

    assert "Found 1 search result" in result
    assert "Description text" in result


def test_extra_snippets_validation_non_string_items(search_handler):
    """Test that non-string items in extra_snippets are handled."""
    handler = search_handler

    sources = [
        {
            "title": "Test",
            "url": "https://test.com",
            "extra_snippets": ["Valid string", 123, None, "Another valid"],
        },
    ]

    # Should not crash, only string items should be included
    result = handler.format_web_sources_content(sources)

    assert "Valid string" in result
    assert "Another valid" in result
    # Non-string items should be skipped
