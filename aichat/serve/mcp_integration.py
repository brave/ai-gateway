import logging

from aichat.protocol.open_ai_protocol import Tool
from aichat.serve.services.mcp.client import MCPClient, has_stdio_mcp_servers
from aichat.serve.services.mcp.executor import MCPToolExecutor
from aichat.serve.services.mcp.handlers.deep_research import DeepResearchServerHandler
from aichat.serve.services.mcp.handlers.default import DefaultMCPServerHandler
from aichat.serve.services.mcp.handlers.search import SearchServerHandler
from aichat.serve.services.mcp.mcp_settings import mcp_settings
from aichat.serve.services.mcp.registry import get_global_registry
from aichat.serve.services.models import ModelConfig

logger = logging.getLogger(__name__)

_initialized = False

_shared_mcp_client: MCPClient | None = None


def _ensure_registry_initialized():
    """Ensure registry has handlers registered."""
    global _initialized
    if not _initialized:
        registry = get_global_registry()
        registry.register_server(SearchServerHandler())

        # Register deep research handler if enabled
        if mcp_settings.deep_research_enabled:
            registry.register_server(DeepResearchServerHandler())
            logger.info("Registered deep research handler")

        registry.register_server(DefaultMCPServerHandler())
        _initialized = True
        logger.info("Initialized MCP registry with handlers")


def get_shared_mcp_client() -> MCPClient | None:
    return _shared_mcp_client


def set_shared_mcp_client(client: MCPClient | None) -> None:
    global _shared_mcp_client
    _shared_mcp_client = client


def _client_for_request(registry) -> MCPClient:
    return get_shared_mcp_client() or MCPClient(registry)


async def warmup_shared_mcp_client() -> MCPClient | None:
    """Start local stdio MCP subprocesses during app lifespan (skipped for HTTP-only configs)."""
    if not mcp_settings.mcp_enabled or not has_stdio_mcp_servers():
        return None
    logger.warning(
        "stdio MCP servers configured: stdio spawns local subprocesses and is for "
        "local development only — do not use in production. Use HTTP transport for "
        "deployed MCP services.",
    )
    _ensure_registry_initialized()
    registry = get_global_registry()
    try:
        client = MCPClient(registry)
        await client.get_tools()
        logger.info("Local stdio MCP subprocesses started")
        return client
    except Exception:
        logger.exception("MCP warmup failed (per-request fallback may still work)")
        return None


async def shutdown_shared_mcp_client(client: MCPClient | None) -> None:
    if client is None:
        return
    try:
        await client.close_all_stdio_transports()
    except Exception as e:
        logger.warning("Error closing MCP stdio transports: %s", e, exc_info=True)
    finally:
        set_shared_mcp_client(None)


async def initialize_mcp_for_request(
    model_config: ModelConfig | None = None,
    matched_categories: frozenset[str] | None = None,
    dynamic_leo_enabled: bool = True,
) -> tuple[list[Tool], MCPToolExecutor | None]:
    """Initialize MCP tools and executor for a request.

    Args:
        model_config: Optional model configuration used to gate per-model tools
                      such as deep_research.
        matched_categories: Dynamic Leo matched category keys used to filter
                            tools tagged ``dynamic``.
        dynamic_leo_enabled: When False, skip Dynamic Leo tag filtering.

    Returns:
        Tuple of (mcp_tools, executor) where:
        - mcp_tools: List of available MCP tools
        - executor: MCPToolExecutor instance for executing tools

    Raises:
        Exception: If initialization fails (caller should handle)
    """
    _ensure_registry_initialized()

    registry = get_global_registry()
    mcp_client = _client_for_request(registry)
    mcp_tools = await mcp_client.get_tools(
        model_config=model_config,
        matched_categories=matched_categories,
        dynamic_leo_enabled=dynamic_leo_enabled,
    )

    executor = MCPToolExecutor(registry, mcp_client)

    logger.info(f"Initialized MCP with {len(mcp_tools)} tools")
    return mcp_tools, executor


async def merge_tools(
    custom_tools: list[Tool] | None,
    mcp_tools: list[Tool],
    include_tools: list[str] | None = None,
    exclude_tools: list[str] | None = None,
) -> list[Tool]:
    """Merge and filter custom and MCP tools.

    Args:
        custom_tools: Custom tools from request (can be None)
        mcp_tools: MCP tools from servers
        include_tools: If specified, only include tools with these names
        exclude_tools: Tool names to exclude

    Returns:
        Merged and filtered tool list
    """
    all_tools = list(custom_tools or [])

    filtered_mcp = mcp_tools

    if include_tools:
        filtered_mcp = [t for t in filtered_mcp if t.function.name in include_tools]
        logger.debug(
            f"Filtered MCP tools to include only: {include_tools} ({len(include_tools)} tools) "
            f"({len(filtered_mcp)} tools)"
        )

    if exclude_tools:
        filtered_mcp = [t for t in filtered_mcp if t.function.name not in exclude_tools]
        logger.debug(f"Excluded {len(exclude_tools)} MCP tools ({exclude_tools})")

    all_tools.extend(filtered_mcp)

    seen_names = set()
    unique_tools = []
    for tool in all_tools:
        name = tool.function.name
        if name not in seen_names:
            seen_names.add(name)
            unique_tools.append(tool)
        else:
            logger.debug(f"Skipping duplicate tool: {name}")

    logger.info(
        f"Merged tools: {len(custom_tools or [])} custom + "
        f"{len(filtered_mcp)} MCP = {len(unique_tools)} total"
    )

    return unique_tools


async def get_tool_guidance() -> dict[str, str]:
    """Get tool guidance strings from registry.

    Returns:
        Dictionary mapping tool names to guidance strings
    """
    try:
        _ensure_registry_initialized()

        registry = get_global_registry()
        mcp_client = _client_for_request(registry)
        _, guidance = await mcp_client.get_tools_with_guidance()

        return guidance
    except Exception as e:
        logger.error(f"Failed to get tool guidance: {e}")
        return {}
