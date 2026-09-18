import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AugmentedToolConfig:
    """Configuration for an augmented tool that modifies a base tool."""

    name: str
    base_tool: str
    description: str
    parameter_name: str
    augmentation_fn: Callable[[Any], Any]


class MCPServerHandler(ABC):
    """Abstract base class for MCP server handlers."""

    @abstractmethod
    def validate_result(self, result: Any) -> bool:
        """Check if this handler can process the given result.

        Args:
            result: The result to validate

        Returns:
            True if the result is valid for this handler
        """

    @abstractmethod
    def format_result(self, tool_name: str, result: Any) -> dict:
        """Format the result for this server type.

        Args:
            tool_name: Name of the tool that produced the result
            result: The raw result to format

        Returns:
            Formatted result dictionary
        """

    @property
    @abstractmethod
    def server_name(self) -> str:
        """Return the server name this handler is for."""

    def _parse_mcp_content(self, result: Any) -> list[str]:
        """Extract text content from standard MCP result format.

        MCP results typically have structure:
        {
            "content": [
                {"type": "text", "text": "..."},
                ...
            ]
        }

        Args:
            result: Raw MCP result

        Returns:
            List of extracted text strings
        """
        if not isinstance(result, dict) or "content" not in result:
            return [str(result)]

        texts = []
        for item in result.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                texts.append(item["text"])
        return texts

    def get_tool_guidance(self) -> dict[str, str]:
        """Return tool names mapped to their specific guidance.

        Returns:
            Dictionary mapping tool names to guidance strings
        """
        return {}

    def get_augmented_tools(self) -> list[AugmentedToolConfig]:
        """Return list of augmented tools for this server.

        Returns:
            List of augmented tool configurations
        """
        return []

    def get_tool_message_content(
        self, formatted_result: dict, tool_call: Any
    ) -> str | list:
        """Get the content for a ToolMessage to send to the LLM.

        Used for previous tool messages in the conversation history sent
        by the client, so the LLM can see the data properly

        Args:
            formatted_result: The formatted result dict from format_result()
            tool_call: The tool call object (for accessing arguments if needed)

        Returns:
            Either a string (simple content) or a list of content parts that
            preserves structured data
        """
        return formatted_result.get("content", "")

    def get_output_content_parts(
        self, formatted_result: dict, tool_call: Any
    ) -> list[dict]:
        """Get output content parts for real-time streaming to the client.

        Args:
            formatted_result: The formatted result dict from format_result()
            tool_call: The tool call object (for accessing arguments if needed)

        Returns:
            List of output content part dicts (empty list if none)
        """
        return []


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server including validation and formatting."""

    name: str
    url: str
    enabled: bool
    validate_result: Callable[[Any], bool]
    format_result: Callable[[str, Any], dict]
    description: str | None = None
    tool_guidance: dict[str, str] | None = None

    def is_enabled(self) -> bool:
        """Check if this MCP server is enabled.

        Returns:
            True if enabled
        """
        return self.enabled

    def validate(self, result: Any) -> bool:
        """Validate if this server can handle the given result.

        Args:
            result: The result to validate

        Returns:
            True if validation passes
        """
        try:
            return self.validate_result(result)
        except Exception as e:
            logger.warning(f"Validation failed for {self.name}: {e}")
            return False

    def format(self, tool_name: str, result: Any) -> dict:
        """Format the result using this server's formatting function.

        Args:
            tool_name: Name of the tool
            result: Raw result to format

        Returns:
            Formatted result dictionary
        """
        try:
            return self.format_result(tool_name, result)
        except Exception as e:
            logger.error(f"Formatting failed for {self.name}: {e}")
            return {
                "type": "brave-mcp-result",
                "tool_name": tool_name,
                "content": str(result),
                "raw_result": result,
                "error": f"Formatting failed: {e}",
            }


class MCPServerRegistry:
    """Registry for managing MCP server configurations."""

    def __init__(self):
        """Initialize empty registry."""
        self._handlers: dict[str, MCPServerHandler] = {}
        self._configs: dict[str, MCPServerConfig] = {}

    @property
    def servers(self) -> dict[str, MCPServerConfig]:
        """Get all registered servers (for backwards compatibility).

        Returns:
            Dictionary of server name to configuration
        """
        return self._configs

    @property
    def handlers(self) -> dict[str, MCPServerHandler]:
        """Get all registered handlers.

        Returns:
            Dictionary of server name to handler
        """
        return self._handlers

    def register_server(
        self,
        handler: MCPServerHandler,
        url: str = "",
        enabled: bool = True,
        description: str | None = None,
    ) -> None:
        """Register a server handler.

        Args:
            handler: The server handler to register
            url: Base URL for the MCP server
            enabled: Whether the server is enabled
            description: Optional server description
        """
        server_name = handler.server_name
        self._handlers[server_name] = handler

        if server_name in self._configs:
            logger.warning(f"Server '{server_name}' already registered. Overwriting.")

        config = MCPServerConfig(
            name=server_name,
            url=url,
            enabled=enabled,
            validate_result=handler.validate_result,
            format_result=handler.format_result,
            description=description,
            tool_guidance=handler.get_tool_guidance(),
        )
        self._configs[server_name] = config
        logger.debug(f"Registered MCP server '{server_name}' from URL: {url}")

    def get_server_config(self, name: str) -> MCPServerConfig | None:
        """Get server configuration by name.

        Args:
            name: Server name

        Returns:
            Server configuration or None if not found
        """
        return self._configs.get(name)

    def get_all_enabled_servers(self) -> list[MCPServerConfig]:
        """Get all enabled server configurations.

        Returns:
            List of enabled server configurations
        """
        return [cfg for cfg in self._configs.values() if cfg.is_enabled()]

    def get_all_servers(self) -> list[MCPServerConfig]:
        """Get all server configurations regardless of enabled status.

        Returns:
            List of all server configurations
        """
        return list(self._configs.values())

    def validate_and_format(
        self, server_name: str, tool_name: str, result: Any
    ) -> dict:
        """Validate and format a result using the server's handler.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            result: Raw result to validate and format

        Returns:
            Formatted result dictionary (with error field if validation fails)
        """
        logger.debug(f"server_name: {server_name}")
        logger.debug(f"config: {self._configs}")
        config = self.get_server_config(server_name)
        if not config:
            logger.error(
                f"Server '{server_name}' not found in registry for "
                f"validation/formatting."
            )
            return {
                "type": "brave-mcp-result",
                "tool_name": tool_name,
                "content": str(result),
                "raw_result": result,
                "error": f"Server '{server_name}' not registered.",
            }

        if not config.validate(result):
            logger.warning(
                f"Result validation failed for tool '{tool_name}' on server "
                f"'{server_name}'."
            )
            return {
                "type": "brave-mcp-result",
                "tool_name": tool_name,
                "server_name": server_name,
                "content": str(result),
                "raw_result": result,
                "error": f"Result validation failed for tool '{tool_name}'.",
            }

        formatted_result = config.format(tool_name, result)

        if isinstance(formatted_result, dict):
            formatted_result["server_name"] = server_name
        return formatted_result

    def get_tool_start_message(
        self, server_name: str, tool_name: str, tool_args: dict
    ) -> str:
        """Get the tool start message from the server's handler.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            tool_args: Tool arguments

        Returns:
            Tool start message string
        """
        handler = self._handlers.get(server_name)
        if handler and hasattr(handler, "get_tool_start_message"):
            return handler.get_tool_start_message(tool_name, tool_args)

        return f"Running {tool_name}..."


_global_registry = MCPServerRegistry()


def get_global_registry() -> MCPServerRegistry:
    """Get the global registry instance.

    Returns:
        The global registry instance
    """
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _global_registry
    _global_registry = MCPServerRegistry()
