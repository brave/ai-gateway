"""Coverage tests for MCPToolExecutor paths not exercised by test_executor.py."""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aichat.serve.services.mcp.executor import MCPToolExecutor
from aichat.serve.services.mcp.types import MCPServerConfig, MCPTool


def _tool(server_name="test-server"):
    return MCPTool(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object"},
        server_name=server_name,
        server_url="http://localhost:8000",
    )


def _http_server():
    return MCPServerConfig(
        name="test-server",
        url="http://localhost:8000",
        enabled=True,
    )


def _stdio_server():
    return MCPServerConfig(
        name="stdio-server",
        command_argv=["python", "-m", "server"],
        transport="stdio",
        enabled=True,
    )


def _mock_client(server):
    client = MagicMock()
    client.servers = [server]
    client.fetch_tools_from_server = AsyncMock(return_value=[_tool(server.name)])
    client._prepare_headers = Mock(return_value={})
    return client


def _registry(validate_return=None):
    registry = MagicMock()
    registry.validate_and_format = Mock(return_value=validate_return or {"ok": True})
    return registry


def _http_cm(response):
    http_client = AsyncMock()
    http_client.post = AsyncMock(return_value=response)
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=http_client)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


@pytest.mark.asyncio
async def test_is_mcp_tool_exception_returns_false(caplog):
    """If tool fetching blows up, is_mcp_tool logs and returns False."""
    client = _mock_client(_http_server())
    client.fetch_tools_from_server = AsyncMock(side_effect=RuntimeError("boom"))
    executor = MCPToolExecutor(registry=_registry(), mcp_client=client)

    assert await executor.is_mcp_tool("test_tool") is False
    assert "Error checking if test_tool is MCP tool" in caplog.text


@pytest.mark.asyncio
async def test_execute_mcp_tool_formats_registry_result():
    """Successful execution is formatted by the registry."""
    executor = MCPToolExecutor(
        registry=_registry({"ok": True}), mcp_client=_mock_client(_http_server())
    )

    response = Mock()
    response.text = json.dumps({"result": {"content": [{"type": "text"}]}})
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json = Mock(return_value={"result": {"content": [{"type": "text"}]}})

    with (
        patch(
            "aichat.serve.services.mcp.executor.tier_http_headers_for_model",
            return_value={},
        ),
        patch("httpx.AsyncClient", return_value=_http_cm(response)),
    ):
        result = await executor.execute_mcp_tool("test_tool", {"arg": 1}, "model-x")

    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_execute_mcp_tool_error_result_passthrough():
    """Error dicts returned by _execute_tool_on_server skip registry formatting."""
    executor = MCPToolExecutor(
        registry=_registry(), mcp_client=_mock_client(_http_server())
    )
    error_result = {"type": "error", "tool_name": "test_tool", "error": "bad"}
    with patch.object(
        executor,
        "_execute_tool_on_server",
        AsyncMock(return_value=error_result),
    ):
        result = await executor.execute_mcp_tool("test_tool", {})

    assert result == error_result


@pytest.mark.asyncio
async def test_execute_tool_http_non_200_sse_without_data(caplog):
    """Non-200 SSE response without data lines logs status then raises."""
    response = Mock()
    response.status_code = 500
    response.text = "event: message\n"
    response.raise_for_status = Mock()

    executor = MCPToolExecutor(
        registry=_registry(), mcp_client=_mock_client(_http_server())
    )
    with (
        patch(
            "aichat.serve.services.mcp.executor.tier_http_headers_for_model",
            return_value={},
        ),
        patch("httpx.AsyncClient", return_value=_http_cm(response)),
        pytest.raises(ValueError, match="No data found in SSE response"),
    ):
        await executor._execute_tool_on_server(
            _http_server(), "test_tool", {}, model="unknown"
        )

    assert any("MCP server returned 500" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_execute_tool_http_error_payload_raises():
    """JSON error payloads in the response raise MCP server error."""
    response = Mock()
    response.status_code = 200
    response.text = json.dumps({"error": {"message": "boom"}})
    response.json = Mock(return_value={"error": {"message": "boom"}})
    response.raise_for_status = Mock()

    executor = MCPToolExecutor(
        registry=_registry(), mcp_client=_mock_client(_http_server())
    )
    with (
        patch(
            "aichat.serve.services.mcp.executor.tier_http_headers_for_model",
            return_value={},
        ),
        patch("httpx.AsyncClient", return_value=_http_cm(response)),
        pytest.raises(Exception, match="MCP server error: boom"),
    ):
        await executor._execute_tool_on_server(
            _http_server(), "test_tool", {}, model="unknown"
        )


@pytest.mark.asyncio
async def test_execute_tool_on_server_stdio_transport():
    """stdio transport delegates to the client's stdio session."""
    executor = MCPToolExecutor(registry=_registry())
    mock_transport = Mock()
    mock_transport.call_tool = AsyncMock(
        return_value={"content": [{"type": "text", "text": "done"}], "isError": False}
    )
    executor._mcp_client = MagicMock()
    executor._mcp_client.get_stdio_transport = Mock(return_value=mock_transport)

    result = await executor._execute_tool_on_server(
        _stdio_server(), "test_tool", {"arg": 1}
    )
    assert result == {"content": [{"type": "text", "text": "done"}], "isError": False}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_text,expected_fragment",
    [
        ("plain failure text", "plain failure text"),
        (
            'boom {"error": {"detail": "bad input", "code": "BRV-4"}}',
            "BRV-4: bad input",
        ),
        ('weird {"error": {"detail": 1}}', "test_tool: 1"),
        ("broken {not json", "broken {not json"),
    ],
)
async def test_execute_tool_on_server_is_error_content_parsing(
    error_text, expected_fragment
):
    """isError content strings are surfaced as MCP error results."""
    executor = MCPToolExecutor(
        registry=_registry(), mcp_client=_mock_client(_http_server())
    )
    mock_transport = Mock()
    mock_transport.call_tool = AsyncMock(
        return_value={
            "isError": True,
            "content": [{"type": "text", "text": error_text}],
        }
    )
    executor._mcp_client = MagicMock()
    executor._mcp_client.get_stdio_transport = Mock(return_value=mock_transport)

    result = await executor._execute_tool_on_server(_stdio_server(), "test_tool", {})

    assert result["type"] == "error"
    assert "MCP server returned error for test_tool: " in result["error"]
    assert expected_fragment in result["error"]
