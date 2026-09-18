import asyncio
import json
import logging
import os
import time
from contextlib import AsyncExitStack
from typing import Any

import httpx

import mcp.types as mcp_types
from aichat.protocol.open_ai_protocol import Tool, ToolFunction
from aichat.serve.services.dynamic_leo.tool_filter import (
    filter_mcp_tools_by_categories,
)
from aichat.serve.services.mcp.mcp_settings import mcp_settings
from aichat.serve.services.mcp.registry import (
    MCPServerRegistry,
    get_global_registry,
)
from aichat.serve.services.mcp.types import MCPServerConfig, MCPTool
from aichat.serve.services.models import ModelConfig
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)


def _mcp_expand_str(value: str) -> str:
    return os.path.expandvars(str(value))


def _mcp_expand_optional(value: Any) -> str | None:
    if value is None:
        return None
    out = _mcp_expand_str(value).strip()
    return out if out else None


def _mcp_transport(server_config: dict[str, Any]) -> str:
    raw = server_config.get("type") or server_config.get("transport")
    if raw is not None:
        t = str(raw).strip().lower().replace("_", "-")
        if t == "stdio":
            return "stdio"
        return "http"
    url_raw = server_config.get("url")
    url_expanded = os.path.expandvars(str(url_raw)).strip() if url_raw else ""
    cmd_raw = server_config.get("command")
    has_command = cmd_raw is not None and (
        (isinstance(cmd_raw, str) and cmd_raw.strip() != "")
        or (isinstance(cmd_raw, list) and len(cmd_raw) > 0)
    )
    if has_command and not url_expanded:
        return "stdio"
    return "http"


def has_stdio_mcp_servers(server_configs: list[dict[str, Any]] | None = None) -> bool:
    """True when MCP_SERVERS includes at least one enabled stdio server."""
    configs = server_configs if server_configs is not None else mcp_settings.mcp_servers
    for server_config in configs:
        if not server_config.get("enabled", True):
            continue
        if _mcp_transport(server_config) == "stdio":
            return True
    return False


def _parse_fastmcp_tags(tool_data: dict[str, Any]) -> frozenset[str]:
    """Extract FastMCP tags from an MCP tools/list tool entry.

    FastMCP exposes tags under ``meta.fastmcp.tags`` in Python objects and
    ``_meta.fastmcp.tags`` in JSON-RPC / by-alias serialization.
    """
    meta = tool_data.get("meta")
    if not isinstance(meta, dict):
        meta = tool_data.get("_meta")
    if not isinstance(meta, dict):
        return frozenset()
    fastmcp_meta = meta.get("fastmcp")
    if not isinstance(fastmcp_meta, dict):
        return frozenset()
    tags = fastmcp_meta.get("tags")
    if not tags:
        return frozenset()
    if isinstance(tags, str):
        return frozenset({tags}) if tags.strip() else frozenset()
    try:
        return frozenset(str(t) for t in tags if t is not None and str(t).strip())
    except TypeError:
        return frozenset()


def _stdio_server_parameters(server: MCPServerConfig) -> StdioServerParameters:
    argv = server.stdio_argv()
    command = argv[0]
    args = list(argv[1:])
    cwd = server.cwd
    if cwd and not os.path.isdir(cwd):
        logger.warning(
            "stdio MCP %s: cwd %r is not usable in this environment "
            "(omitted; subprocess uses server process cwd). "
            "Set cwd/command in MCP_SERVERS to paths that exist here (e.g. in Docker).",
            server.name,
            cwd,
        )
        cwd = None
    merged_env: dict[str, str] = {str(k): str(v) for k, v in os.environ.items()}
    if server.env:
        for key, val in server.env.items():
            merged_env[str(key)] = str(val)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    merged_env.setdefault("FASTMCP_SHOW_SERVER_BANNER", "false")
    return StdioServerParameters(command=command, args=args, env=merged_env, cwd=cwd)


class StdioMCPTransport:
    """MCP stdio transport via the official ``mcp`` SDK (subprocess + ``ClientSession``)."""

    def __init__(self, server: MCPServerConfig) -> None:
        self._server = server
        self._initialized = False
        self._lock = asyncio.Lock()
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def initialize(self) -> bool:
        if self._initialized:
            return True
        if self._server.transport != "stdio":
            return False

        async with self._lock:
            if self._initialized:
                return True
            try:
                params = _stdio_server_parameters(self._server)
            except ValueError as e:
                logger.error(
                    "stdio MCP %s: invalid command configuration: %s",
                    self._server.name,
                    e,
                )
                return False

            stack = AsyncExitStack()
            try:
                await stack.__aenter__()
                read_write = await stack.enter_async_context(stdio_client(params))
                read, write = read_write
                session = await stack.enter_async_context(
                    ClientSession(
                        read,
                        write,
                        client_info=mcp_types.Implementation(
                            name="aichat-mcp-client", version="1.0.0"
                        ),
                    )
                )
                await session.initialize()
            except FileNotFoundError as e:
                await stack.aclose()
                logger.error(
                    "stdio MCP %s: cannot execute %r: %s",
                    self._server.name,
                    params.command,
                    e,
                )
                return False
            except OSError as e:
                await stack.aclose()
                logger.error(
                    "stdio MCP %s: OS error starting process (%r): %s",
                    self._server.name,
                    params.command,
                    e,
                )
                return False
            except Exception:
                await stack.aclose()
                logger.exception(f"Failed to initialize stdio MCP {self._server.name}")
                return False

            self._exit_stack = stack
            self._session = session
            self._initialized = True
            logger.info("stdio MCP server %s initialized", self._server.name)
            return True

    async def list_tools(self) -> list[dict[str, Any]]:
        if not self._initialized and not await self.initialize():
            return []
        assert self._session is not None
        try:
            res = await self._session.list_tools()
        except Exception:
            logger.exception(f"stdio MCP {self._server.name} tools/list error")
            return []
        return [t.model_dump(mode="json", by_alias=True) for t in res.tools]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self._initialized and not await self.initialize():
            raise RuntimeError(f"stdio MCP server {self._server.name} not initialized")
        assert self._session is not None
        try:
            res = await self._session.call_tool(tool_name, arguments)
        except Exception as e:
            logger.error("stdio MCP error for %s: %s", tool_name, e)
            raise RuntimeError(f"MCP server error: {e}") from e
        return res.model_dump(mode="json", by_alias=True)

    async def close(self) -> None:
        stack = self._exit_stack
        self._exit_stack = None
        self._session = None
        self._initialized = False
        if stack is not None:
            await stack.aclose()


class MCPToolsTTLCache:
    def __init__(self, ttl=1800):
        self.cache = {}
        self.ttl = ttl

    def get(self, key):
        now = time.time()
        if key in self.cache:
            expire_time, value = self.cache[key]
            if now < expire_time:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key, value):
        expire_time = time.time() + self.ttl
        self.cache[key] = (expire_time, value)


class MCPClient:
    """Client for fetching tools from MCP servers."""

    def __init__(self, registry: MCPServerRegistry | None = None):
        """
        Initialize MCP client with optional registry.

        Args:
            registry: Optional server registry for enhanced server handling.
                     Uses global registry if not provided.
        """
        self.registry = registry or get_global_registry()

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        self.tool_list_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }

        self._initialized_servers: dict[str, bool] = {}
        self._session_ids: dict[str, str] = {}
        self._stdio_transports: dict[str, StdioMCPTransport] = {}

        self.servers: list[MCPServerConfig] = []
        for server_config in mcp_settings.mcp_servers:
            if not server_config.get("enabled", True):
                continue
            reg_config = self.registry.get_server_config(server_config["name"])
            transport_kind = _mcp_transport(server_config)

            if transport_kind == "stdio":
                url_value = server_config.get("url") or ""
                if url_value:
                    url_value = _mcp_expand_str(url_value)
                else:
                    url_value = f"stdio://{server_config['name']}"

                cmd_raw = server_config.get("command")
                command_argv: list[str] | None = None
                if isinstance(cmd_raw, list):
                    command_argv = [_mcp_expand_str(x) for x in cmd_raw]
                elif cmd_raw is not None:
                    command_argv = [_mcp_expand_str(cmd_raw)] + [
                        _mcp_expand_str(x) for x in (server_config.get("args") or [])
                    ]

                cwd = _mcp_expand_optional(server_config.get("cwd"))

                env_raw = server_config.get("env")
                stdio_env = None
                if isinstance(env_raw, dict) and env_raw:
                    stdio_env = {
                        str(k): _mcp_expand_str(str(v)) for k, v in env_raw.items()
                    }

                self.servers.append(
                    MCPServerConfig(
                        name=server_config["name"],
                        url=url_value,
                        enabled=bool(server_config.get("enabled", True)),
                        transport="stdio",
                        command_argv=command_argv,
                        cwd=cwd,
                        env=stdio_env,
                    )
                )
            else:
                self.servers.append(
                    MCPServerConfig(
                        name=server_config["name"],
                        url=server_config["url"],
                        enabled=bool(server_config.get("enabled", True)),
                    )
                )

            if reg_config:
                reg_config.url = self.servers[-1].url

        logger.info(f"Initialized MCP client with {len(self.servers)} enabled servers")

        self.cache = MCPToolsTTLCache()

    def get_stdio_transport(self, server: MCPServerConfig) -> StdioMCPTransport:
        key = server.name
        if key not in self._stdio_transports:
            self._stdio_transports[key] = StdioMCPTransport(server)
        return self._stdio_transports[key]

    async def _initialize_server(self, server: MCPServerConfig) -> bool:
        """Initialize MCP server connection if not already initialized.

        Note - this is needed as otherwise the request will fail
        """
        if server.transport == "stdio":
            return await self.get_stdio_transport(server).initialize()

        server_key = f"{server.name}:{server.url}"

        if self._initialized_servers.get(server_key, False):
            return True

        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }

            request_payload = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {},
                    },
                    "clientInfo": {
                        "name": "aichat-mcp-client",
                        "version": "1.0.0",
                    },
                },
            }

            logger.debug(f"Initializing MCP server {server.name} at {server.url}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{server.url}/mcp",
                    json=request_payload,
                    headers=headers,
                    timeout=10.0,
                )

            response.raise_for_status()

            if "mcp-session-id" in response.headers:
                self._session_ids[server_key] = response.headers["mcp-session-id"]
                logger.info(f"Got session ID for {server.name}")

            response_text = response.text
            if response_text.startswith("event:"):
                lines = response_text.strip().split("\n")
                json_data = None
                for line in lines:
                    if line.startswith("data: "):
                        json_data = line[6:]
                        break
                if json_data:
                    data = json.loads(json_data)
                else:
                    raise ValueError("No data found in SSE response")
            else:
                data = response.json()

            if "error" in data:
                logger.error(
                    f"MCP server {server.name} initialization failed: {data['error']}"
                )
                return False

            logger.info(f"MCP server {server.name} initialized successfully")
            self._initialized_servers[server_key] = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCP server {server.name}: {e}")
            return False

    def _prepare_headers(self, server: MCPServerConfig) -> dict[str, str]:
        """Prepare HTTP headers for MCP requests including session management."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        server_key = f"{server.name}:{server.url}"
        if server_key in self._session_ids:
            headers["mcp-session-id"] = self._session_ids[server_key]

        return headers

    async def fetch_tools_from_server(self, server: MCPServerConfig) -> list[MCPTool]:
        """
        Fetch available tools from an MCP server.

        Args:
            server: MCP server configuration

        Returns:
            List of tools available from the server
        """
        if not await self._initialize_server(server):
            logger.warning(
                f"Failed to initialize MCP server {server.name}, skipping tool fetch"
            )
            return []

        if server.transport == "stdio":
            try:
                tools_data = await self.get_stdio_transport(server).list_tools()
            except Exception:
                logger.exception(f"Failed to fetch tools from server '{server.name}'")
                return []
        else:
            try:
                headers = self._prepare_headers(server)
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{server.url}/mcp",
                        json=self.tool_list_payload,
                        headers=headers,
                        timeout=10.0,
                    )
                response.raise_for_status()

                response_text = response.text
                if response_text.startswith("event:"):
                    lines = response_text.strip().split("\n")
                    json_data = None
                    for line in lines:
                        if line.startswith("data: "):
                            json_data = line[6:]
                            break
                    if json_data:
                        data = json.loads(json_data)
                    else:
                        raise ValueError("No data found in SSE response")
                else:
                    data = response.json()

                if "error" in data:
                    logger.error(
                        f"MCP server {server.name} returned error: {data['error']}"
                    )
                    return []

                result = data.get("result", {})
                tools_data = result.get("tools", [])

            except (httpx.HTTPError, httpx.TimeoutException):
                logger.exception(f"Failed to fetch tools from server '{server.name}'")
                return []
            except (json.JSONDecodeError, KeyError, ValueError):
                logger.error(
                    f"Failed to parse tools response from server '{server.name}'"
                )
                return []

        tools = []
        for tool_data in tools_data:
            tool = MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
                server_name=server.name,
                server_url=server.url,
                tool_id=tool_data.get("id"),
                tags=_parse_fastmcp_tags(tool_data),
            )
            tools.append(tool)

        logger.info(f"Fetched {len(tools)} tools from server '{server.name}'")
        return tools

    @staticmethod
    def convert_to_openai_format(mcp_tool: MCPTool) -> Tool:
        """
        Convert an MCP tool to OpenAI tool format.

        Args:
            mcp_tool: MCP tool to convert

        Returns:
            OpenAI-compatible Tool object
        """
        description = mcp_tool.description or f"Tool from {mcp_tool.server_name}"
        parameters = mcp_tool.input_schema or {
            "type": "object",
            "properties": {},
        }
        return Tool(
            type="function",
            function=ToolFunction(
                name=mcp_tool.name,
                description=description,
                parameters=parameters,
            ),
        )

    def _get_deep_research_tool(self) -> Tool:
        """Get the deep research tool definition."""
        return Tool(
            type="function",
            function=ToolFunction(
                name="deep_research",
                description=(
                    "Run a long-form, multi-step research workflow that issues many iterative "
                    "web searches and synthesizes a cited report. This is SLOW and EXPENSIVE "
                    "(takes minutes and produces a long report), so only invoke it when the "
                    "user has explicitly opted into a research workflow.\n"
                    "\n"
                    "ONLY use when the user's prompt explicitly requests research-style output, "
                    "for example using phrases like: 'deep research', 'deep dive', 'research "
                    "report', 'comprehensive analysis', 'investigate', 'thorough comparison', "
                    "'literature review', 'write a report on', or 'analyze in depth'.\n"
                    "\n"
                    "Do NOT use for:\n"
                    "- 'Tell me about X', 'Who is X', 'What is X', 'Explain X' style prompts "
                    "(answer directly or use `brave_web_search`).\n"
                    "- Single-fact lookups, definitions, recent news, scores, prices, or "
                    "summaries (use `brave_web_search`).\n"
                    "- Conversational or general-knowledge questions the assistant can answer "
                    "from its own knowledge.\n"
                    "- Any question a single web search could resolve.\n"
                    "\n"
                    "When the user's intent is ambiguous, prefer `brave_web_search` or a direct "
                    "answer. Do not upgrade a casual question to deep research on the user's "
                    "behalf."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The user's explicit research request, preserved as stated. "
                                "Do not expand a short or casual question into a research "
                                "topic — if the user did not ask for research, do not call "
                                "this tool."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            ),
        )

    async def get_tools(
        self,
        model_config: ModelConfig | None = None,
        matched_categories: frozenset[str] | None = None,
        dynamic_leo_enabled: bool = True,
    ) -> list[Tool]:
        """
        Get available tools from configured MCP servers in OpenAI format.

        The base tool catalog (from MCP servers, including FastMCP tags) is
        cached. Per-request Dynamic Leo category filtering and the
        deep_research tool are applied after the cache lookup so different
        requests receive correctly filtered lists without polluting the cache.

        Returns:
            List of OpenAI-compatible tools
        """
        cache_key = "mcp_tools_catalog"
        cached_catalog = self.cache.get(cache_key)

        if cached_catalog is None:
            all_mcp_tools: list[MCPTool] = []
            for server in self.servers:
                tools = await self.fetch_tools_from_server(server)
                if tools:
                    all_mcp_tools.extend(tools)
            cached_catalog = all_mcp_tools
            self.cache.set(cache_key, cached_catalog)
            logger.info(f"Cached {len(cached_catalog)} MCP tools in catalog")
        else:
            logger.info("Returning cached MCP tools catalog")

        filtered_catalog = filter_mcp_tools_by_categories(
            cached_catalog,
            matched_categories,
            dynamic_leo_enabled=dynamic_leo_enabled,
        )
        openai_tools = [
            self.convert_to_openai_format(tool) for tool in filtered_catalog
        ]

        model_supports_deep_research = (
            model_config is not None and model_config.deep_research_support
        )
        if mcp_settings.deep_research_enabled and model_supports_deep_research:
            openai_tools.append(self._get_deep_research_tool())
            logger.info("Added deep_research tool to available tools")

        logger.info(f"Returning {len(openai_tools)} total MCP tools")
        return openai_tools

    async def close_all_stdio_transports(self) -> None:
        """Tear down stdio subprocesses (call from app shutdown)."""
        for transport in list(self._stdio_transports.values()):
            await transport.close()
        self._stdio_transports.clear()

    async def get_tools_with_guidance(self) -> tuple[list[Tool], dict[str, str]]:
        """
        Get all tools with their guidance strings from registry.

        Returns:
            Tuple of (tools, guidance) where guidance is a dict mapping
            tool names to guidance strings
        """
        tools = await self.get_tools(dynamic_leo_enabled=False)

        guidance = {}
        for server in self.servers:
            reg_config = self.registry.get_server_config(server.name)
            if reg_config and reg_config.tool_guidance:
                guidance.update(reg_config.tool_guidance)

        if mcp_settings.deep_research_enabled:
            deep_research_config = self.registry.get_server_config("deep_research")
            if deep_research_config and deep_research_config.tool_guidance:
                guidance.update(deep_research_config.tool_guidance)

        return tools, guidance
