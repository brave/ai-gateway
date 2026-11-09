import json
from typing import Any
from urllib.parse import urlparse

from aichat.protocol.open_ai_protocol import WebSourcesContentPart
from aichat.responses import WebSource

_ALLOWED_SOURCE_URL_SCHEMES = {"http", "https"}


def safe_source_url(url: Any) -> str:
    """Return url if it is an http(s) URL with a host, otherwise an empty string."""
    if not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return ""
    if parsed.scheme.lower() not in _ALLOWED_SOURCE_URL_SCHEMES or not parsed.netloc:
        return ""
    return url


def extract_query_from_tool_call(tool_call: Any) -> str | None:
    """Extract query from tool call arguments."""
    try:
        tool_args = json.loads(tool_call.arguments) if tool_call.arguments else {}
        return tool_args.get("query")
    except (json.JSONDecodeError, AttributeError):
        return None


def convert_sources_to_web_source_objects(sources: list[dict]) -> list[WebSource]:
    """Convert dict sources to WebSource objects."""
    return [
        WebSource(
            title=source.get("title", ""),
            url=safe_source_url(source.get("url", "")),
            favicon=source.get("favicon"),
            page_content=source.get("page_content"),
            extra_snippets=source.get("extra_snippets"),
        )
        for source in sources
    ]


def build_web_sources_content_part(
    sources: list[dict],
    tool_call: Any,
    queries: list[str] | str | None = None,
) -> WebSourcesContentPart:
    """Build WebSourcesContentPart from sources and tool call. Uses MCP queries when provided."""
    source_objects = convert_sources_to_web_source_objects(sources)
    query = queries if queries is not None else extract_query_from_tool_call(tool_call)
    return WebSourcesContentPart(
        type="brave-chat.webSources",
        sources=source_objects,
        query=query,
    )


def build_web_sources_output_part(
    sources: list[dict],
    tool_call: Any,
    rich_results: list[dict] | None = None,
    queries: list[str] | str | None = None,
) -> dict:
    """Build web sources output content part dict for streaming. Uses MCP queries when provided."""
    source_objects = convert_sources_to_web_source_objects(sources)
    output_part = {
        "type": "brave-chat.webSources",
        "sources": [
            {
                "title": source.title,
                "url": source.url,
                "favicon": source.favicon,
                "page_content": source.page_content,
                "extra_snippets": source.extra_snippets,
            }
            for source in source_objects
        ],
    }
    query = queries if queries is not None else extract_query_from_tool_call(tool_call)
    if query is not None:
        output_part["query"] = query
    if rich_results:
        output_part["rich_results"] = rich_results

    return output_part
