"""Filter MCP tools using FastMCP tags and Dynamic Leo matched categories."""

from __future__ import annotations

from aichat.serve.services.mcp.types import MCPTool

ALWAYS_TAG = "always"
DYNAMIC_TAG = "dynamic"


def should_apply_dynamic_leo_tool_filter(
    *,
    enable_dynamic_leo: bool,
    prompt_caching_support: bool | None,
) -> bool:
    """Whether to filter MCP tools by Dynamic Leo category tags.

    Disabled when prompt caching is enabled so the Bedrock tools prefix stays
    stable across turns (classification may still run for model triage).
    """
    return bool(enable_dynamic_leo and not prompt_caching_support)


def filter_mcp_tools_by_categories(
    tools: list[MCPTool],
    matched_categories: frozenset[str] | None,
    *,
    dynamic_leo_enabled: bool = True,
) -> list[MCPTool]:
    """Filter tools by Dynamic Leo matched category tags.

    Rules when ``dynamic_leo_enabled`` is True:
    - Tools tagged ``always`` are always included.
    - Tools tagged ``dynamic`` are included only when their tags intersect
      ``matched_categories`` (category keys from DYNAMIC_LEO_CONFIG).
    - Untagged tools are included for backward compatibility.

    When ``dynamic_leo_enabled`` is False, all tools are returned unchanged.
    """
    if not dynamic_leo_enabled:
        return list(tools)

    matched = matched_categories or frozenset()
    filtered: list[MCPTool] = []
    for tool in tools:
        tags = tool.tags
        if ALWAYS_TAG in tags:
            filtered.append(tool)
        elif DYNAMIC_TAG in tags:
            if tags & matched:
                filtered.append(tool)
        else:
            filtered.append(tool)
    return filtered
