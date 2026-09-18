import json
import logging
from typing import Any

import httpx

from aichat.serve.services.mcp.client import MCPClient
from aichat.serve.services.mcp.context import tier_http_headers_for_model
from aichat.serve.services.mcp.registry import (
    MCPServerRegistry,
    get_global_registry,
)
from aichat.serve.services.mcp.types import MCPServerConfig, MCPTool

logger = logging.getLogger(__name__)


class MCPToolExecutor:
    """Executor for MCP tools that integrates with the existing tool system."""

    def __init__(
        self,
        registry: MCPServerRegistry | None = None,
        mcp_client: MCPClient | None = None,
    ):
        """Initialize MCP tool executor.

        Args:
            registry: Optional server registry, uses global if not provided
            mcp_client: Optional pre-initialized MCP client (reuses sessions)
        """
        self.registry = registry or get_global_registry()
        self._mcp_client: MCPClient | None = mcp_client
        self._tool_cache: list[MCPTool] | None = None

    async def _get_mcp_client(self) -> MCPClient:
        """Lazy load MCP client.

        Returns:
            MCP client instance
        """
        if self._mcp_client is None:
            self._mcp_client = MCPClient(self.registry)
        return self._mcp_client

    async def is_mcp_tool(self, tool_name: str) -> bool:
        """Check if a tool is handled by MCP servers.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is available from an MCP server
        """
        try:
            client = await self._get_mcp_client()

            if self._tool_cache is None:
                tools = await self._fetch_all_tools(client)
                self._tool_cache = tools

            return any(tool.name == tool_name for tool in self._tool_cache)
        except Exception as e:
            logger.error(f"Error checking if {tool_name} is MCP tool: {e}")
            return False

    async def _fetch_all_tools(self, client: MCPClient) -> list[MCPTool]:
        """Fetch all tools from all servers.

        Args:
            client: MCP client instance

        Returns:
            List of all available MCP tools
        """
        all_tools = []
        for server in client.servers:
            tools = await client.fetch_tools_from_server(server)
            all_tools.extend(tools)
        return all_tools

    async def execute_mcp_tool(
        self, tool_name: str, args: dict, model: str = "unknown"
    ) -> dict:
        """Execute an MCP tool call.

        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments as a dictionary
            model: Model name (for logging)

        Returns:
            Formatted result dictionary
        """
        try:

            client = await self._get_mcp_client()

            tool_info = await self._find_tool_server(client, tool_name)

            if not tool_info:
                logger.error(f"Tool {tool_name} not found in any MCP server")
                return {}

            _tool, server_config = tool_info

            logger.info(
                f"Executing MCP tool {tool_name} " f"on server {server_config.name}"
            )

            result = await self._execute_tool_on_server(
                server_config, tool_name, args, model=model
            )

            if isinstance(result, dict) and result.get("type") == "error":
                return result

            formatted_result = self.registry.validate_and_format(
                server_config.name, tool_name, result
            )

            return formatted_result

        except Exception as e:
            error_msg = str(e)
            logger.exception(f"Failed to execute MCP tool {tool_name}")
            error_result = {
                "type": "error",
                "tool_name": tool_name,
                "error": error_msg,
                "content": f"Error executing {tool_name}: {error_msg}",
            }
            logger.debug(f"Returning error result: {error_result}")
            return error_result

    async def _find_tool_server(
        self, client: MCPClient, tool_name: str
    ) -> tuple[MCPTool, MCPServerConfig] | None:
        """Find which server provides the given tool.

        Args:
            client: MCP client instance
            tool_name: Name of the tool to find

        Returns:
            Tuple of (tool, server_config) or None if not found
        """
        if self._tool_cache is None:
            self._tool_cache = await self._fetch_all_tools(client)

        for tool in self._tool_cache:
            if tool.name == tool_name:
                for server in client.servers:
                    if server.name == tool.server_name:
                        return tool, server

        return None

    async def _execute_tool_http(
        self,
        mcp_client: MCPClient,
        server_config: MCPServerConfig,
        tool_name: str,
        arguments: dict,
        model: str,
    ) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        headers = mcp_client._prepare_headers(server_config)
        headers.update(tier_http_headers_for_model(model))

        logger.info(
            f"Calling MCP tool {tool_name} at {server_config.url}/mcp with payload: {payload}"
        )

        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                f"{server_config.url}/mcp",
                json=payload,
                headers=headers,
                timeout=30.0,
            )

        if response.status_code != 200:
            logger.error(f"MCP server returned {response.status_code}: {response.text}")

        response.raise_for_status()

        response_text = response.text
        lines = response_text.strip().split("\n")
        is_sse = any(
            line.strip().startswith("event:") or line.strip().startswith(":")
            for line in lines
        )

        if is_sse:
            data = None
            for line in lines:
                # Skip ping comments (SSE keep-alive messages)
                if line.strip().startswith(":"):
                    continue
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    break
            if data is None:
                raise ValueError("No data found in SSE response")
        else:
            data = response.json()

        if "error" in data:
            error_data = data["error"]
            if isinstance(error_data, dict):
                error_msg = error_data.get("message", str(error_data))
            else:
                error_msg = str(error_data)
            logger.error(f"MCP server error for {tool_name}: {error_msg}")
            raise RuntimeError(f"MCP server error: {error_msg}")

        return data.get("result", {})

    async def _execute_tool_on_server(
        self,
        server_config: MCPServerConfig,
        tool_name: str,
        arguments: dict,
        model: str = "unknown",
    ) -> Any:
        """Low-level tool execution via HTTP or stdio transport.

        Args:
            server_config: MCP server configuration
            tool_name: Name of the tool to execute
            arguments: Tool arguments dictionary

        Returns:
            Raw result from the server

        Raises:
            Exception: If the server returns an error or request fails
        """
        mcp_client = await self._get_mcp_client()

        if server_config.transport == "stdio":
            result = await mcp_client.get_stdio_transport(server_config).call_tool(
                tool_name, arguments
            )
        else:
            result = await self._execute_tool_http(
                mcp_client, server_config, tool_name, arguments, model
            )
        if isinstance(result, dict) and result.get("isError"):
            error_text = ""
            content = result.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                first_item = content[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    error_text = first_item["text"]
                    try:
                        if "{" in error_text:
                            json_start = error_text.find("{")
                            json_part = error_text[json_start:]
                            error_json = json.loads(json_part)
                            if "error" in error_json:
                                error_detail = error_json["error"]
                                if isinstance(error_detail, dict):
                                    error_msg = error_detail.get("detail", error_text)
                                    error_code = error_detail.get("code", "")
                                    if error_code:
                                        error_msg = f"{error_code}: {error_msg}"
                                    error_text = error_msg
                    except (json.JSONDecodeError, ValueError):
                        pass

            error_msg = f"MCP server returned error for {tool_name}: " f"{error_text}"
            return {
                "type": "error",
                "tool_name": tool_name,
                "error": error_msg,
                "content": error_msg,
            }
        return result

    def clear_cache(self) -> None:
        """Clear the tool cache (useful for testing or manual refresh)."""
        self._tool_cache = None
        logger.info("Cleared MCP tool cache")
