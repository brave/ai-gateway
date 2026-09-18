import json
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings

MCP_SERVERS_DEFAULT = "[]"


class MCPSettings(BaseSettings):
    mcp_enabled: bool = True
    mcp_servers: list[dict[str, Any]] = json.loads(MCP_SERVERS_DEFAULT)
    deep_research_enabled: bool = False
    deep_research_url: str = "http://localhost:8080"
    deep_research_max_iterations: int = 3
    deep_research_max_queries: int = 30
    deep_research_max_seconds: int = 300

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def _coerce_mcp_servers(cls, v: Any) -> Any:
        if v is None or v == "":
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return []
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return parsed
            return []
        return []


mcp_settings = MCPSettings()
