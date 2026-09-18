# Configuring MCP Servers

MCP servers are configured via `app_settings.mcp_servers`. Each server should have:

- `name`: Unique server identifier
- `url`: Base URL for the MCP server
- `enabled`: Whether the server is active

Example:

```python
mcp_servers = [
    {
        "name": "brave-search",
        "url": "http://brave-search-mcp:8000",
        "enabled": True,
    }
]
```

## Custom Server Handlers

To add server-specific logic, implement the `MCPServerHandler` interface:

```python
from aichat.serve.services.mcp.registry import (
    MCPServerHandler,
    get_global_registry
)

class MyServerHandler(MCPServerHandler):
    @property
    def server_name(self) -> str:
        return "my-server"

    def validate_result(self, result) -> bool:
        return isinstance(result, dict) and "data" in result

    def format_result(self, tool_name: str, result: dict) -> dict:
        return {
            "type": "my-server-result",
            "tool_name": tool_name,
            "content": result["data"]
        }

# Register at startup
registry = get_global_registry()
registry.register_server(
    MyServerHandler(),
    url="http://my-server:8000",
    enabled=True
)
```
