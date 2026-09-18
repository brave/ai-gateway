"""Type definitions for MCP integration."""

from dataclasses import dataclass
from typing import Any


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server (HTTP ``url`` or stdio ``command``)."""

    name: str
    url: str = ""
    enabled: bool = True
    transport: str = "http"
    command_argv: list[str] | None = None
    cwd: str | None = None
    env: dict[str, str] | None = None

    def stdio_argv(self) -> list[str]:
        """Argv for asyncio.create_subprocess_exec (stdio transport only)."""
        if not self.command_argv:
            raise ValueError(
                f"stdio MCP server {self.name!r} needs 'command' (+ optional 'args') "
                "or a JSON list-valued 'command'"
            )
        return self.command_argv


@dataclass
class MCPTool:
    """Represents a tool from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str
    server_url: str
    tool_id: str | None = None
    tags: frozenset[str] = frozenset()
