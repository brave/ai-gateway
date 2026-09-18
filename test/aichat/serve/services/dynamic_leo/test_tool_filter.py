from aichat.serve.services.dynamic_leo.tool_filter import (
    ALWAYS_TAG,
    DYNAMIC_TAG,
    filter_mcp_tools_by_categories,
    should_apply_dynamic_leo_tool_filter,
)
from aichat.serve.services.mcp.types import MCPTool


def _tool(name: str, tags: frozenset[str] | None = None) -> MCPTool:
    return MCPTool(
        name=name,
        description=f"{name} description",
        input_schema={"type": "object", "properties": {}},
        server_name="brave_search",
        server_url="http://localhost",
        tags=tags or frozenset(),
    )


def test_always_tagged_tools_are_included():
    tools = [
        _tool("brave_web_search", frozenset({ALWAYS_TAG})),
        _tool("brave_faqs_search", frozenset({DYNAMIC_TAG, "brave_faqs"})),
    ]
    filtered = filter_mcp_tools_by_categories(tools, frozenset())
    assert [t.name for t in filtered] == ["brave_web_search"]


def test_dynamic_tool_included_when_category_matches():
    tools = [
        _tool("brave_web_search", frozenset({ALWAYS_TAG})),
        _tool("brave_faqs_search", frozenset({DYNAMIC_TAG, "brave_faqs"})),
    ]
    filtered = filter_mcp_tools_by_categories(tools, frozenset({"brave_faqs"}))
    assert [t.name for t in filtered] == ["brave_web_search", "brave_faqs_search"]


def test_dynamic_tool_excluded_when_category_does_not_match():
    tools = [
        _tool("brave_faqs_search", frozenset({DYNAMIC_TAG, "brave_faqs"})),
    ]
    filtered = filter_mcp_tools_by_categories(tools, frozenset({"coding"}))
    assert filtered == []


def test_multi_label_union_includes_all_matching_dynamic_tools():
    tools = [
        _tool("brave_web_search", frozenset({ALWAYS_TAG})),
        _tool("brave_faqs_search", frozenset({DYNAMIC_TAG, "brave_faqs"})),
        _tool("coding_helper", frozenset({DYNAMIC_TAG, "coding"})),
        _tool("news_helper", frozenset({DYNAMIC_TAG, "news"})),
    ]
    filtered = filter_mcp_tools_by_categories(
        tools, frozenset({"brave_faqs", "coding"})
    )
    assert [t.name for t in filtered] == [
        "brave_web_search",
        "brave_faqs_search",
        "coding_helper",
    ]


def test_untagged_tools_included_for_backward_compat():
    tools = [_tool("legacy_tool")]
    filtered = filter_mcp_tools_by_categories(tools, frozenset())
    assert [t.name for t in filtered] == ["legacy_tool"]


def test_dynamic_leo_disabled_returns_all_tools():
    tools = [
        _tool("brave_web_search", frozenset({ALWAYS_TAG})),
        _tool("brave_faqs_search", frozenset({DYNAMIC_TAG, "brave_faqs"})),
    ]
    filtered = filter_mcp_tools_by_categories(
        tools, frozenset(), dynamic_leo_enabled=False
    )
    assert [t.name for t in filtered] == ["brave_web_search", "brave_faqs_search"]


def test_should_apply_dynamic_leo_tool_filter_respects_prompt_caching():
    assert (
        should_apply_dynamic_leo_tool_filter(
            enable_dynamic_leo=True,
            prompt_caching_support=True,
        )
        is False
    )
    assert (
        should_apply_dynamic_leo_tool_filter(
            enable_dynamic_leo=True,
            prompt_caching_support=False,
        )
        is True
    )
    assert (
        should_apply_dynamic_leo_tool_filter(
            enable_dynamic_leo=False,
            prompt_caching_support=False,
        )
        is False
    )


def test_none_matched_categories_excludes_dynamic_tools():
    tools = [
        _tool("brave_web_search", frozenset({ALWAYS_TAG})),
        _tool("brave_faqs_search", frozenset({DYNAMIC_TAG, "brave_faqs"})),
    ]
    filtered = filter_mcp_tools_by_categories(tools, None)
    assert [t.name for t in filtered] == ["brave_web_search"]
