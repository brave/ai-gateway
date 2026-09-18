"""Tests for MCP Tool Executor."""

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import pytest_asyncio

from aichat.serve.services.mcp.executor import MCPToolExecutor
from aichat.serve.services.mcp.registry import (
    MCPServerHandler,
    MCPServerRegistry,
)
from aichat.serve.services.mcp.types import MCPServerConfig


class TestServerHandler(MCPServerHandler):
    """Test handler for executor tests."""

    @property
    def server_name(self) -> str:
        return "test-server"

    def validate_result(self, result) -> bool:
        return isinstance(result, dict)

    def format_result(self, tool_name: str, result: dict) -> dict:
        return {
            "type": "test-result",
            "tool_name": tool_name,
            "content": result.get("content", []),
        }


@pytest.fixture
def registry():
    """Create a registry with test handler."""
    MCPServerRegistry._instance = None
    reg = MCPServerRegistry()
    handler = TestServerHandler()
    reg.register_server(handler, "http://localhost:8000", enabled=True)
    return reg


@pytest_asyncio.fixture
async def httpx_client():
    """Create a mock httpx client."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest_asyncio.fixture
async def executor(httpx_client, registry):
    """Create an executor with mocked client."""
    return MCPToolExecutor(registry)


@pytest.mark.asyncio
async def test_executor_initialization(httpx_client, registry):
    """Test executor initializes correctly."""
    executor = MCPToolExecutor(registry)

    assert executor.registry is registry
    assert executor._mcp_client is None
    assert executor._tool_cache is None


@pytest.mark.asyncio
async def test_lazy_mcp_client_loading(executor):
    """Test that MCP client is lazily loaded."""
    assert executor._mcp_client is None

    client = await executor._get_mcp_client()
    assert client is not None
    assert executor._mcp_client is client

    # Second call should return same instance
    client2 = await executor._get_mcp_client()
    assert client2 is client


@pytest.mark.asyncio
async def test_is_mcp_tool_found(executor, httpx_client):
    """Test checking if a tool is an MCP tool."""
    # Mock the MCP server response
    tools_response = {
        "result": {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "Test tool",
                    "inputSchema": {"type": "object"},
                }
            ]
        }
    }

    # Mock responses: initialize, tools/list
    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = f"event: message\ndata: {json.dumps(init_response)}\n\n"
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    tools_mock.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        # Clear cached MCP client so it's created fresh with patched settings
        executor._mcp_client = None
        executor._tool_cache = None
        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            result = await executor.is_mcp_tool("test_tool")
            assert result is True


@pytest.mark.asyncio
async def test_is_mcp_tool_not_found(executor, httpx_client):
    """Test checking for non-existent tool."""
    tools_response = {
        "result": {
            "tools": [
                {
                    "name": "other_tool",
                    "description": "Other tool",
                    "inputSchema": {"type": "object"},
                }
            ]
        }
    }

    # Mock responses: initialize, tools/list
    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = f"event: message\ndata: {json.dumps(init_response)}\n\n"
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    tools_mock.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        # Clear cached MCP client so it's created fresh with patched settings
        executor._mcp_client = None
        executor._tool_cache = None
        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            result = await executor.is_mcp_tool("nonexistent_tool")
            assert result is False


@pytest.mark.asyncio
async def test_execute_mcp_tool_success(executor, httpx_client):
    """Test successful tool execution."""
    # Mock tools list response
    tools_response = {
        "result": {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "Test tool",
                    "inputSchema": {"type": "object"},
                }
            ]
        }
    }

    # Mock tool execution response
    exec_response = {"result": {"content": [{"type": "text", "text": "Tool result"}]}}

    # Mock responses: initialize, tools/list, tools/call
    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = f"event: message\ndata: {json.dumps(init_response)}\n\n"
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    tools_mock.raise_for_status = Mock()

    exec_mock = Mock()
    exec_mock.text = json.dumps(exec_response)
    exec_mock.json = Mock(return_value=exec_response)
    exec_mock.status_code = 200
    exec_mock.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock, exec_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        # Clear cached MCP client so it's created fresh with patched settings
        executor._mcp_client = None
        executor._tool_cache = None
        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            result = await executor.execute_mcp_tool("test_tool", {"arg": "value"})

            assert result is not None
            assert result["type"] == "test-result"
            assert result["tool_name"] == "test_tool"


@pytest.mark.asyncio
async def test_execute_mcp_tool_sends_brave_context_headers(executor):
    """Brave model tier is sent as HTTP headers on tools/call."""
    tools_response = {
        "result": {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "Test tool",
                    "inputSchema": {"type": "object"},
                }
            ]
        }
    }
    exec_response = {"result": {"content": [{"type": "text", "text": "Tool result"}]}}
    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = f"event: message\ndata: {json.dumps(init_response)}\n\n"
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()
    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    tools_mock.raise_for_status = Mock()
    exec_mock = Mock()
    exec_mock.text = json.dumps(exec_response)
    exec_mock.json = Mock(return_value=exec_response)
    exec_mock.status_code = 200
    exec_mock.raise_for_status = Mock()

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock, exec_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings:
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        executor._mcp_client = None
        executor._tool_cache = None
        with (
            patch(
                "aichat.serve.services.mcp.executor.tier_http_headers_for_model",
                return_value={"X-Brave-Tier": "freemium"},
            ),
            patch("httpx.AsyncClient", return_value=mock_async_client_cm),
        ):
            await executor.execute_mcp_tool(
                "test_tool",
                {"arg": "value"},
                model="near-deepseek-v3-1",
            )

    call_kwargs = mock_async_client.post.call_args_list[-1].kwargs
    headers = call_kwargs["headers"]
    assert headers["X-Brave-Tier"] == "freemium"
    assert "X-Brave-Model-Id" not in headers
    assert "X-Brave-Max-Page-Content" not in headers
    assert "X-Brave-User-Tier" not in headers


@pytest.mark.asyncio
async def test_execute_mcp_tool_not_found(executor, httpx_client, caplog):
    """Test executing non-existent tool."""
    tools_response = {"result": {"tools": []}}

    # Mock responses: initialize, tools/list
    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = f"event: message\ndata: {json.dumps(init_response)}\n\n"
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    tools_mock.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        # Clear cached MCP client so it's created fresh with patched settings
        executor._mcp_client = None
        executor._tool_cache = None
        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            result = await executor.execute_mcp_tool("nonexistent", {})

            assert result == {}
            assert "not found in any MCP server" in caplog.text


@pytest.mark.asyncio
async def test_execute_mcp_tool_server_error(executor, httpx_client, caplog):
    """Test handling of server errors during execution."""
    tools_response = {
        "result": {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "Test tool",
                    "inputSchema": {"type": "object"},
                }
            ]
        }
    }

    error_response = {"error": {"code": -32601, "message": "Method not found"}}

    # Mock responses: initialize, tools/list, tools/call (error)
    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = json.dumps(init_response)
    init_mock.json = Mock(return_value=init_response)
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    tools_mock.raise_for_status = Mock()

    error_mock = Mock()
    error_mock.text = json.dumps(error_response)
    error_mock.json = Mock(return_value=error_response)
    error_mock.status_code = 200
    error_mock.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock, error_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        # Clear cached MCP client so it's created fresh with patched settings
        executor._mcp_client = None
        executor._tool_cache = None
        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            result = await executor.execute_mcp_tool("test_tool", {})

            # Should return error structure, not empty dict
            assert isinstance(result, dict)
            assert result.get("type") == "error"
            assert result.get("tool_name") == "test_tool"
            assert "error" in result
            assert "content" in result
            assert "Failed to execute MCP tool" in caplog.text


@pytest.mark.asyncio
async def test_execute_tool_on_server_sse_format(executor, httpx_client):
    """Test _execute_tool_on_server with SSE response format."""
    exec_response = {"result": {"content": [{"type": "text", "text": "SSE result"}]}}

    mock_response = Mock()
    mock_response.text = f"event: message\ndata: {json.dumps(exec_response)}\n\n"
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    server_config = MCPServerConfig(
        name="test-server",
        url="http://localhost:8000",
        enabled=True,
    )
    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,
        patch("httpx.AsyncClient", return_value=mock_async_client_cm),
    ):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        executor._mcp_client = None
        result = await executor._execute_tool_on_server(
            server_config, "test_tool", {"arg": "value"}
        )

        assert result == {"content": [{"type": "text", "text": "SSE result"}]}


@pytest.mark.asyncio
async def test_execute_tool_on_server_json_format(executor, httpx_client):
    """Test _execute_tool_on_server with regular JSON response."""
    exec_response = {"result": {"content": [{"type": "text", "text": "JSON result"}]}}

    mock_response = Mock()
    mock_response.text = json.dumps(exec_response)
    mock_response.json = Mock(return_value=exec_response)
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    server_config = MCPServerConfig(
        name="test-server",
        url="http://localhost:8000",
        enabled=True,
    )
    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,
        patch("httpx.AsyncClient", return_value=mock_async_client_cm),
    ):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        executor._mcp_client = None
        result = await executor._execute_tool_on_server(
            server_config, "test_tool", {"arg": "value"}
        )

        assert result == {"content": [{"type": "text", "text": "JSON result"}]}


@pytest.mark.asyncio
async def test_execute_tool_on_server_error_response(executor, httpx_client, caplog):
    """Test error handling in _execute_tool_on_server."""
    error_response = {"error": {"code": -32700, "message": "Parse error"}}

    mock_response = Mock()
    mock_response.text = json.dumps(error_response)
    mock_response.json = Mock(return_value=error_response)
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    server_config = MCPServerConfig(
        name="test-server",
        url="http://localhost:8000",
        enabled=True,
    )
    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,
        patch("httpx.AsyncClient", return_value=mock_async_client_cm),
    ):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        executor._mcp_client = None
        with pytest.raises(Exception, match="MCP server error"):
            await executor._execute_tool_on_server(server_config, "test_tool", {})

        assert "MCP server error" in caplog.text


@pytest.mark.asyncio
async def test_execute_tool_on_server_network_error(executor, httpx_client):
    """Test network error handling."""
    # Create mock async client context manager that raises error
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    server_config = MCPServerConfig(
        name="test-server",
        url="http://localhost:8000",
        enabled=True,
    )
    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,
        patch("httpx.AsyncClient", return_value=mock_async_client_cm),
    ):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        executor._mcp_client = None
        with pytest.raises(httpx.HTTPError):
            await executor._execute_tool_on_server(server_config, "test_tool", {})


@pytest.mark.asyncio
async def test_find_tool_server(executor, httpx_client):
    """Test finding which server provides a tool."""
    tools_response = {
        "result": {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "Test tool",
                    "inputSchema": {"type": "object"},
                }
            ]
        }
    }

    # Mock responses: initialize, tools/list
    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = f"event: message\ndata: {json.dumps(init_response)}\n\n"
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    tools_mock.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        # Clear cached MCP client so it's created fresh with patched settings
        executor._mcp_client = None
        executor._tool_cache = None
        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            client = await executor._get_mcp_client()
            tool_info = await executor._find_tool_server(client, "test_tool")

            assert tool_info is not None
            tool, server_config = tool_info
            assert tool.name == "test_tool"
            assert server_config.name == "test-server"


@pytest.mark.asyncio
async def test_find_tool_server_not_found(executor, httpx_client):
    """Test finding non-existent tool returns None."""
    tools_response = {"result": {"tools": []}}

    # Mock responses: initialize, tools/list
    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = f"event: message\ndata: {json.dumps(init_response)}\n\n"
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    tools_mock.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        # Clear cached MCP client so it's created fresh with patched settings
        executor._mcp_client = None
        executor._tool_cache = None
        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            client = await executor._get_mcp_client()
            tool_info = await executor._find_tool_server(client, "nonexistent")

            assert tool_info is None


@pytest.mark.asyncio
async def test_clear_cache(executor, httpx_client):
    """Test clearing the tool cache."""
    # Populate cache
    tools_response = {
        "result": {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "Test tool",
                    "inputSchema": {"type": "object"},
                }
            ]
        }
    }

    # Mock responses: initialize, tools/list
    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = f"event: message\ndata: {json.dumps(init_response)}\n\n"
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    tools_mock.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        # Clear cached MCP client so it's created fresh with patched settings
        executor._mcp_client = None
        executor._tool_cache = None
        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            # Load tool into cache
            await executor.is_mcp_tool("test_tool")
            assert executor._tool_cache is not None

            # Clear cache
            executor.clear_cache()
            assert executor._tool_cache is None


@pytest.mark.asyncio
async def test_fetch_all_tools(executor, httpx_client):
    """Test fetching all tools from all servers."""
    tools_response = {
        "result": {
            "tools": [
                {
                    "name": "tool1",
                    "description": "Tool 1",
                    "inputSchema": {"type": "object"},
                },
                {
                    "name": "tool2",
                    "description": "Tool 2",
                    "inputSchema": {"type": "object"},
                },
            ]
        }
    }

    # Mock responses: initialize, tools/list
    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = f"event: message\ndata: {json.dumps(init_response)}\n\n"
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    tools_mock.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        # Clear cached MCP client so it's created fresh with patched settings
        executor._mcp_client = None
        executor._tool_cache = None
        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            client = await executor._get_mcp_client()
            all_tools = await executor._fetch_all_tools(client)

            assert len(all_tools) == 2
            assert all_tools[0].name == "tool1"
            assert all_tools[1].name == "tool2"


@pytest.mark.asyncio
async def test_execute_tool_on_server_sse_with_ping_messages(executor, httpx_client):
    """Test _execute_tool_on_server with SSE response containing ping messages."""
    exec_response = {"result": {"content": [{"type": "text", "text": "SSE result"}]}}

    # SSE response with ping messages before event/data
    sse_response = (
        ": ping - 2026-01-14 16:36:25.310058+00:00\n"
        "\n"
        ": ping - 2026-01-14 16:36:40.311899+00:00\n"
        "\n"
        "event: message\n"
        f"data: {json.dumps(exec_response)}\n"
        "\n"
    )

    mock_response = Mock()
    mock_response.text = sse_response
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()

    # Create mock async client context manager
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    server_config = MCPServerConfig(
        name="test-server",
        url="http://localhost:8000",
        enabled=True,
    )
    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,
        patch("httpx.AsyncClient", return_value=mock_async_client_cm),
    ):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        executor._mcp_client = None
        result = await executor._execute_tool_on_server(
            server_config, "test_tool", {"arg": "value"}
        )

        assert result == {"content": [{"type": "text", "text": "SSE result"}]}


@pytest.mark.asyncio
async def test_execute_tool_on_server_sse_ping_at_start(executor, httpx_client):
    """Test SSE detection when response starts with ping (no leading whitespace)."""
    exec_response = {"result": {"content": [{"type": "text", "text": "Result"}]}}

    # SSE response starting directly with ping
    sse_response = (
        ": ping - 2026-01-14 16:36:25.310058+00:00\n"
        "event: message\n"
        f"data: {json.dumps(exec_response)}\n"
    )

    mock_response = Mock()
    mock_response.text = sse_response
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    server_config = MCPServerConfig(
        name="test-server",
        url="http://localhost:8000",
        enabled=True,
    )
    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,
        patch("httpx.AsyncClient", return_value=mock_async_client_cm),
    ):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        executor._mcp_client = None
        result = await executor._execute_tool_on_server(server_config, "test_tool", {})

        assert result == {"content": [{"type": "text", "text": "Result"}]}


@pytest.mark.asyncio
async def test_execute_tool_on_server_sse_multiple_pings(executor, httpx_client):
    """Test SSE parsing with multiple ping messages."""
    exec_response = {"result": {"content": [{"type": "text", "text": "Final result"}]}}

    # Multiple ping messages
    sse_response = (
        ": ping - 2026-01-14 16:36:25.310058+00:00\n"
        "\n"
        ": ping - 2026-01-14 16:36:40.311899+00:00\n"
        "\n"
        ": ping - 2026-01-14 16:36:55.313553+00:00\n"
        "\n"
        "event: message\n"
        f"data: {json.dumps(exec_response)}\n"
        "\n"
    )

    mock_response = Mock()
    mock_response.text = sse_response
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    server_config = MCPServerConfig(
        name="test-server",
        url="http://localhost:8000",
        enabled=True,
    )
    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,
        patch("httpx.AsyncClient", return_value=mock_async_client_cm),
    ):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        executor._mcp_client = None
        result = await executor._execute_tool_on_server(server_config, "test_tool", {})

        assert result == {"content": [{"type": "text", "text": "Final result"}]}


@pytest.mark.asyncio
async def test_execute_tool_on_server_sse_ping_with_whitespace(executor, httpx_client):
    """Test SSE parsing when ping lines have leading/trailing whitespace."""
    exec_response = {"result": {"content": [{"type": "text", "text": "Result"}]}}

    # Ping lines with whitespace
    sse_response = (
        "  : ping - 2026-01-14 16:36:25.310058+00:00  \n"
        "\n"
        "event: message\n"
        f"data: {json.dumps(exec_response)}\n"
    )

    mock_response = Mock()
    mock_response.text = sse_response
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    server_config = MCPServerConfig(
        name="test-server",
        url="http://localhost:8000",
        enabled=True,
    )
    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,
        patch("httpx.AsyncClient", return_value=mock_async_client_cm),
    ):
        mock_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        executor._mcp_client = None
        result = await executor._execute_tool_on_server(server_config, "test_tool", {})

        assert result == {"content": [{"type": "text", "text": "Result"}]}
