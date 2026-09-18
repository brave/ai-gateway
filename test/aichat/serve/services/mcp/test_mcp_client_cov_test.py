"""Coverage tests for aichat.serve.services.mcp.client."""

import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import mcp.types as mcp_types
import pytest

from aichat.serve.services.mcp.client import (
    MCPClient,
    MCPToolsTTLCache,
    StdioMCPTransport,
    _mcp_expand_str,
    _mcp_transport,
    _parse_fastmcp_tags,
    _stdio_server_parameters,
    has_stdio_mcp_servers,
)
from aichat.serve.services.mcp.types import MCPServerConfig, MCPTool

CLIENT = "aichat.serve.services.mcp.client"


def _cm(return_value):
    """Async context manager mock: __aenter__ -> return_value."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _stdio_server(name="s", cwd=None, env=None):
    return MCPServerConfig(
        name=name,
        url=f"stdio://{name}",
        transport="stdio",
        command_argv=["python", "-m", "some.server"],
        cwd=cwd,
        env=env,
    )


def _http_server(name="h", url="http://h"):
    return MCPServerConfig(name=name, url=url, transport="http")


def settings(servers, deep_research_enabled=False):
    settings = MagicMock()
    settings.mcp_servers = servers
    settings.deep_research_enabled = deep_research_enabled
    return settings


def _client_with_servers(servers, deep_research_enabled=False):
    registry = MagicMock()
    registry.get_server_config = Mock(return_value=None)
    with patch(f"{CLIENT}.mcp_settings", settings(servers, deep_research_enabled)):
        return MCPClient(registry=registry)


def _post_cm(response):
    return _cm(Mock(post=AsyncMock(return_value=response)))


def _http_response(headers=None, text=None, json_data=None, status=200):
    response = MagicMock()
    response.status_code = status
    response.headers = headers or {}
    response.text = text or ""
    response.json = Mock(return_value=json_data if json_data is not None else {})
    response.raise_for_status = Mock()
    return response


# ---------------------------------------------------------------------------
# module helpers
# ---------------------------------------------------------------------------


def test_mcp_transport_explicit_and_inferred():
    assert _mcp_transport({"type": "stdio"}) == "stdio"
    assert _mcp_transport({"type": "STDIO"}) == "stdio"
    assert _mcp_transport({"transport": "streamable_http"}) == "http"
    assert _mcp_transport({"url": "http://x"}) == "http"
    assert _mcp_transport({"command": ["python"]}) == "stdio"
    assert _mcp_transport({"command": "python", "url": "http://x"}) == "http"
    assert _mcp_transport({}) == "http"


def test_mcp_expand_str():
    assert _mcp_expand_str("plain") == "plain"


def test_parse_fastmcp_tags_variants():
    assert _parse_fastmcp_tags(
        {"meta": {"fastmcp": {"tags": ["a", "b"]}}}
    ) == frozenset({"a", "b"})
    assert _parse_fastmcp_tags({"_meta": {"fastmcp": {"tags": "solo"}}}) == frozenset(
        {"solo"}
    )
    assert _parse_fastmcp_tags({"_meta": {"fastmcp": {}}}) == frozenset()
    assert _parse_fastmcp_tags({}) == frozenset()
    assert _parse_fastmcp_tags({"meta": "not-a-dict"}) == frozenset()


def test_has_stdio_mcp_servers():
    assert has_stdio_mcp_servers([{"command": ["python"]}])
    assert not has_stdio_mcp_servers([{"url": "http://x"}])
    assert not has_stdio_mcp_servers([{"command": ["python"], "enabled": False}])


# ---------------------------------------------------------------------------
# _stdio_server_parameters
# ---------------------------------------------------------------------------


def test_stdio_server_parameters_happy(tmp_path):
    server = _stdio_server(cwd=str(tmp_path), env={"MY_ENV": "1"})
    params = _stdio_server_parameters(server)
    assert params.command == "python"
    assert params.args == ["-m", "some.server"]
    assert params.cwd == str(tmp_path)
    assert params.env["PYTHONUNBUFFERED"] == "1"
    assert params.env["FASTMCP_SHOW_SERVER_BANNER"] == "false"
    assert params.env["MY_ENV"] == "1"
    assert params.env["PATH"] == os.environ.get("PATH")


def test_stdio_server_parameters_bad_cwd_warns(tmp_path):
    server = _stdio_server(cwd=str(tmp_path / "missing-dir"))
    params = _stdio_server_parameters(server)
    assert params.cwd is None


def test_stdio_server_parameters_no_cwd():
    params = _stdio_server_parameters(_stdio_server())
    assert params.cwd is None


# ---------------------------------------------------------------------------
# StdioMCPTransport
# ---------------------------------------------------------------------------


def _initialized_transport(session=None):
    transport = StdioMCPTransport(_stdio_server())
    transport._initialized = True
    transport._session = session if session is not None else MagicMock()
    return transport


@pytest.mark.asyncio
async def test_transport_initialize_success():
    transport = StdioMCPTransport(_stdio_server())
    session = MagicMock()
    session.initialize = AsyncMock()
    with (
        patch(f"{CLIENT}.stdio_client", return_value=_cm((Mock(), Mock()))),
        patch(f"{CLIENT}.ClientSession", return_value=_cm(session)),
    ):
        assert await transport.initialize() is True
    assert transport._initialized is True
    assert transport._session is session
    assert await transport.initialize() is True


@pytest.mark.asyncio
async def test_transport_initialize_already_initialized():
    assert await _initialized_transport().initialize() is True


@pytest.mark.asyncio
async def test_transport_initialize_non_stdio_returns_false():
    assert await StdioMCPTransport(_http_server()).initialize() is False


@pytest.mark.asyncio
async def test_transport_initialize_invalid_config_returns_false():
    transport = StdioMCPTransport(_stdio_server())
    with patch(f"{CLIENT}._stdio_server_parameters", side_effect=ValueError("boom")):
        assert await transport.initialize() is False
    assert transport._initialized is False


@pytest.mark.parametrize("exc", [FileNotFoundError, OSError, RuntimeError])
@pytest.mark.asyncio
async def test_transport_initialize_process_failures(exc):
    transport = StdioMCPTransport(_stdio_server())
    with patch(f"{CLIENT}.stdio_client", side_effect=exc("boom")):
        assert await transport.initialize() is False
    assert transport._initialized is False


@pytest.mark.asyncio
async def test_transport_list_tools_success():
    tool = mcp_types.Tool(name="t1", description="d", inputSchema={"type": "object"})
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=Mock(tools=[tool]))
    transport = _initialized_transport(session)
    assert await transport.list_tools() == [tool.model_dump(mode="json", by_alias=True)]


@pytest.mark.asyncio
async def test_transport_list_tools_init_fails_returns_empty():
    transport = StdioMCPTransport(_stdio_server())
    with patch(f"{CLIENT}._stdio_server_parameters", side_effect=ValueError("boom")):
        assert await transport.list_tools() == []


@pytest.mark.asyncio
async def test_transport_list_tools_error_returns_empty():
    session = MagicMock()
    session.list_tools = AsyncMock(side_effect=RuntimeError("boom"))
    transport = _initialized_transport(session)
    assert await transport.list_tools() == []


@pytest.mark.asyncio
async def test_transport_call_tool_success():
    res = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="ok")], isError=False
    )
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=res)
    transport = _initialized_transport(session)
    result = await transport.call_tool("t1", {"a": 1})
    assert result["content"][0]["text"] == "ok"


@pytest.mark.asyncio
async def test_transport_call_tool_not_initialized():
    transport = StdioMCPTransport(_stdio_server())
    with pytest.raises(Exception, match="not initialized"):
        await transport.call_tool("t1", {})


@pytest.mark.asyncio
async def test_transport_call_tool_error_raises():
    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=RuntimeError("bad"))
    transport = _initialized_transport(session)
    with pytest.raises(Exception, match="MCP server error"):
        await transport.call_tool("t1", {})


@pytest.mark.asyncio
async def test_transport_close():
    transport = StdioMCPTransport(_stdio_server())
    stack = MagicMock()
    stack.aclose = AsyncMock()
    transport._exit_stack = stack
    transport._session = MagicMock()
    await transport.close()
    assert transport._session is None
    stack.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------


def test_ttl_cache_get_set():
    cache = MCPToolsTTLCache(ttl=1800)
    assert cache.get("k") is None
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_ttl_cache_expiry():
    cache = MCPToolsTTLCache(ttl=0.01)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    import time

    time.sleep(0.02)
    assert cache.get("k") is None


# ---------------------------------------------------------------------------
# MCPClient construction
# ---------------------------------------------------------------------------


def test_client_init_stdio_and_http_servers():
    servers = [
        {
            "name": "stdio-a",
            "transport": "stdio",
            "command": ["python", "-m", "x"],
            "env": {"SRV": "1"},
        },
        {"name": "disabled", "command": ["python"], "enabled": False},
        {"name": "http-srv", "url": "http://http-srv"},
    ]
    client = _client_with_servers(servers)
    assert len(client.servers) == 2
    stdio_server = client.servers[0]
    assert stdio_server.transport == "stdio"
    assert stdio_server.url == "stdio://stdio-a"
    assert stdio_server.command_argv == ["python", "-m", "x"]
    assert stdio_server.env == {"SRV": "1"}
    http_server = client.servers[1]
    assert http_server.transport == "http"
    assert http_server.url == "http://http-srv"


def test_client_init_registry_config_url_update():
    servers = [{"name": "http-srv", "url": "http://x"}]
    reg_config = Mock(url=None)
    registry = MagicMock()
    registry.get_server_config = Mock(return_value=reg_config)
    with patch(f"{CLIENT}.mcp_settings", settings(servers)):
        MCPClient(registry=registry)
    assert reg_config.url == "http://x"


def test_get_stdio_transport_caches():
    client = _client_with_servers([])
    server = _stdio_server(name="cache-me")
    assert client.get_stdio_transport(server) is client.get_stdio_transport(server)


# ---------------------------------------------------------------------------
# MCPClient._initialize_server / _prepare_headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_server_stdio_delegates():
    client = _client_with_servers([])
    transport = MagicMock()
    transport.initialize = AsyncMock(return_value=True)
    with patch.object(client, "get_stdio_transport", return_value=transport) as factory:
        assert await client._initialize_server(_stdio_server(name="std")) is True
        factory.assert_called_once()
        transport.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_server_http_sse_session_id():
    client = _client_with_servers([{"name": "h", "url": "http://h"}])
    response = _http_response(
        headers={"mcp-session-id": "sid-1"},
        text='event: message\ndata: {"result": {}}\n',
    )
    with patch(f"{CLIENT}.httpx.AsyncClient", return_value=_post_cm(response)):
        assert await client._initialize_server(_http_server()) is True
    assert client._session_ids["h:http://h"] == "sid-1"


@pytest.mark.asyncio
async def test_initialize_server_http_error_dict_false():
    client = _client_with_servers([{"name": "h", "url": "http://h"}])
    response = _http_response(json_data={"error": {"code": -32000}})
    with patch(f"{CLIENT}.httpx.AsyncClient", return_value=_post_cm(response)):
        assert await client._initialize_server(_http_server()) is False


@pytest.mark.asyncio
async def test_initialize_server_http_exception_false():
    client = _client_with_servers([{"name": "h", "url": "http://h"}])
    post = Mock(post=AsyncMock(side_effect=httpx.ConnectError("nope")))
    with patch(f"{CLIENT}.httpx.AsyncClient", return_value=_cm(post)):
        assert await client._initialize_server(_http_server()) is False


@pytest.mark.asyncio
async def test_initialize_server_cached_true():
    client = _client_with_servers([{"name": "h", "url": "http://h"}])
    client._initialized_servers["h:http://h"] = True
    assert await client._initialize_server(_http_server()) is True


def test_prepare_headers_session_id():
    client = _client_with_servers([])
    client._session_ids["h:http://h"] = "sid-1"
    headers = client._prepare_headers(_http_server())
    assert headers["mcp-session-id"] == "sid-1"
    assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# fetch_tools_from_server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_tools_init_failure_returns_empty():
    client = _client_with_servers([])
    with patch.object(client, "_initialize_server", AsyncMock(return_value=False)):
        assert await client.fetch_tools_from_server(_http_server()) == []


@pytest.mark.asyncio
async def test_fetch_tools_stdio_success():
    client = _client_with_servers([])
    server = _stdio_server(name="stdio-srv")
    tools_data = [
        {
            "name": "tool-a",
            "description": "desc",
            "inputSchema": {"type": "object"},
            "_meta": {"fastmcp": {"tags": ["t"]}},
        }
    ]
    transport = MagicMock()
    transport.initialize = AsyncMock(return_value=True)
    transport.list_tools = AsyncMock(return_value=tools_data)
    with patch.object(client, "get_stdio_transport", return_value=transport):
        tools = await client.fetch_tools_from_server(server)
    assert len(tools) == 1
    assert tools[0].name == "tool-a"
    assert tools[0].tags == frozenset({"t"})
    assert tools[0].server_name == "stdio-srv"


@pytest.mark.asyncio
async def test_fetch_tools_stdio_error_returns_empty():
    client = _client_with_servers([])
    transport = MagicMock()
    transport.initialize = AsyncMock(return_value=True)
    transport.list_tools = AsyncMock(side_effect=RuntimeError("boom"))
    with patch.object(client, "get_stdio_transport", return_value=transport):
        assert await client.fetch_tools_from_server(_stdio_server()) == []


@pytest.mark.asyncio
async def test_fetch_tools_http_success():
    client = _client_with_servers([{"name": "h", "url": "http://h"}])
    client._initialized_servers["h:http://h"] = True
    payload = {
        "result": {
            "tools": [
                {
                    "name": "tool-h",
                    "description": "d",
                    "inputSchema": {"type": "object"},
                }
            ]
        }
    }
    with patch(
        f"{CLIENT}.httpx.AsyncClient",
        return_value=_post_cm(_http_response(json_data=payload)),
    ):
        tools = await client.fetch_tools_from_server(_http_server())
    assert len(tools) == 1
    assert tools[0].server_name == "h"
    assert tools[0].server_url == "http://h"


@pytest.mark.asyncio
async def test_fetch_tools_http_error_dict_empty():
    client = _client_with_servers([{"name": "h", "url": "http://h"}])
    client._initialized_servers["h:http://h"] = True
    response = _http_response(json_data={"error": {"message": "nope"}})
    with patch(f"{CLIENT}.httpx.AsyncClient", return_value=_post_cm(response)):
        assert await client.fetch_tools_from_server(_http_server()) == []


@pytest.mark.asyncio
async def test_fetch_tools_http_error_exception_empty():
    client = _client_with_servers([{"name": "h", "url": "http://h"}])
    client._initialized_servers["h:http://h"] = True
    response = _http_response(json_data={})
    response.raise_for_status = Mock(side_effect=httpx.HTTPError("bad"))
    with patch(f"{CLIENT}.httpx.AsyncClient", return_value=_post_cm(response)):
        assert await client.fetch_tools_from_server(_http_server()) == []


@pytest.mark.asyncio
async def test_fetch_tools_http_json_decode_error_empty():
    client = _client_with_servers([{"name": "h", "url": "http://h"}])
    client._initialized_servers["h:http://h"] = True
    response = _http_response(text="event: message\ndata: {broken\n")
    with patch(f"{CLIENT}.httpx.AsyncClient", return_value=_post_cm(response)):
        assert await client.fetch_tools_from_server(_http_server()) == []


# ---------------------------------------------------------------------------
# get_tools / cache / close / guidance
# ---------------------------------------------------------------------------


def _mcp_tool(name="tool-x"):

    return MCPTool(
        name=name,
        description="d",
        input_schema={},
        server_name="h",
        server_url="http://h",
    )


@pytest.mark.asyncio
async def test_get_tools_cache_miss_then_hit():
    client = _client_with_servers(
        [{"name": "h", "url": "http://h"}], deep_research_enabled=False
    )
    with (
        patch(f"{CLIENT}.mcp_settings", settings([], deep_research_enabled=False)),
        patch.object(
            MCPClient,
            "fetch_tools_from_server",
            new=AsyncMock(return_value=[_mcp_tool()]),
        ) as fetch,
        patch(
            f"{CLIENT}.filter_mcp_tools_by_categories",
            side_effect=lambda catalog, cats, dynamic_leo_enabled: catalog,
        ),
    ):
        tools = await client.get_tools()
        assert len(tools) == 1
        assert tools[0].function.name == "tool-x"
        tools2 = await client.get_tools()
        assert len(tools2) == 1
        assert fetch.await_count == 1


@pytest.mark.asyncio
async def test_get_tools_deep_research_appended():
    client = _client_with_servers([], deep_research_enabled=True)
    model_config = Mock(deep_research_support=True)
    with (
        patch(f"{CLIENT}.mcp_settings", settings([], deep_research_enabled=True)),
        patch(
            f"{CLIENT}.filter_mcp_tools_by_categories",
            side_effect=lambda catalog, cats, dynamic_leo_enabled: catalog,
        ),
    ):
        tools = await client.get_tools(model_config=model_config)
    assert any(tool.function.name == "deep_research" for tool in tools)


@pytest.mark.asyncio
async def test_close_all_stdio_transports():
    client = _client_with_servers([])
    transport = MagicMock()
    transport.close = AsyncMock()
    client._stdio_transports["s"] = transport
    await client.close_all_stdio_transports()
    transport.close.assert_awaited_once()
    assert client._stdio_transports == {}


@pytest.mark.asyncio
async def test_get_tools_with_guidance():
    client = _client_with_servers([], deep_research_enabled=True)
    reg_config = Mock(tool_guidance={"tool-x": "be careful"})
    deep_config = Mock(tool_guidance={"deep_research": "research guidance"})
    registry = MagicMock()
    registry.get_server_config = Mock(
        side_effect=lambda name: deep_config if name == "deep_research" else reg_config
    )
    client.registry = registry
    with (
        patch(f"{CLIENT}.mcp_settings", settings([], deep_research_enabled=True)),
        patch(
            f"{CLIENT}.filter_mcp_tools_by_categories",
            side_effect=lambda catalog, cats, dynamic_leo_enabled: catalog,
        ),
    ):
        tools, guidance = await client.get_tools_with_guidance()
    assert guidance == {"deep_research": "research guidance"}
    assert isinstance(tools, list)
