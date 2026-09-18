import functools

from aichat.serve.services.mcp.client import MCPClient
from aichat.serve.services.mcp.types import MCPServerConfig, MCPTool


@functools.cache
def get_mcp_client() -> MCPClient:
    """
    Get a cached singleton instance of MCPClient.
    This ensures the client's internal cache is preserved across requests.
    """
    return MCPClient()


__all__ = [
    "MCPClient",
    "MCPServerConfig",
    "MCPTool",
    "get_mcp_client",
]
