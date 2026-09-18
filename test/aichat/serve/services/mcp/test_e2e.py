"""End-to-end tests for MCP tool execution flow."""

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import pytest_asyncio

from aichat.protocol.open_ai_protocol import Tool, ToolFunction
from aichat.serve.mcp_integration import initialize_mcp_for_request, merge_tools
from aichat.serve.services.mcp.executor import MCPToolExecutor
from aichat.serve.services.mcp.registry import MCPServerHandler, MCPServerRegistry
from aichat.serve.tool_parser import ToolCall, ToolExecutor


class E2ETestHandler(MCPServerHandler):
    """Handler for end-to-end tests."""

    @property
    def server_name(self) -> str:
        return "e2e-server"

    def validate_result(self, result) -> bool:
        return isinstance(result, dict) and "content" in result

    def format_result(self, tool_name: str, result: dict) -> dict:
        return {
            "type": "e2e-mcp-result",
            "tool_name": tool_name,
            "result": result.get("content"),
            "server": "e2e-server",
        }

    def get_tool_guidance(self) -> dict:
        return {"e2e_search": "Search for information in E2E test environment"}


@pytest.fixture(autouse=True)
def reset_state():
    """Reset all state before each test."""
    MCPServerRegistry._instance = None
    yield
    MCPServerRegistry._instance = None


@pytest_asyncio.fixture
async def httpx_client():
    """Create a mock httpx client."""
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def mock_mcp_server_responses():
    """Create mock responses for MCP server."""

    def create_tools_response():
        return {
            "result": {
                "tools": [
                    {
                        "name": "e2e_search",
                        "description": "Search for information",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                            },
                            "required": ["query"],
                        },
                    },
                    {
                        "name": "e2e_calculate",
                        "description": "Perform calculations",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"},
                            },
                            "required": ["expression"],
                        },
                    },
                ]
            }
        }

    def create_execution_response(tool_name, args):
        if tool_name == "e2e_search":
            return {
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Search results for: {args.get('query', '')}",
                        }
                    ]
                }
            }
        elif tool_name == "e2e_calculate":
            return {
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Result: {args.get('expression', '')} = 42",
                        }
                    ]
                }
            }

    return create_tools_response, create_execution_response


@pytest.mark.asyncio
async def test_full_mcp_initialization_flow(httpx_client, mock_mcp_server_responses):
    """Test complete MCP initialization flow."""
    create_tools_resp, _ = mock_mcp_server_responses

    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = json.dumps(init_response)
    init_mock.json = Mock(return_value=init_response)
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(create_tools_resp())}\n\n"
    tools_mock.raise_for_status = Mock()

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "e2e-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        mock_settings.deep_research_enabled = False

        handler = E2ETestHandler()
        registry = MCPServerRegistry()
        registry.register_server(handler, "http://localhost:8000", enabled=True)

        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            mcp_tools, executor = await initialize_mcp_for_request()

            assert len(mcp_tools) == 2
            tool_names = {t.function.name for t in mcp_tools}
            assert "e2e_search" in tool_names
            assert "e2e_calculate" in tool_names

            assert executor is not None
            assert isinstance(executor, MCPToolExecutor)


@pytest.mark.asyncio
async def test_tool_execution_through_tool_parser(
    httpx_client, mock_mcp_server_responses
):
    """Test tool execution through ToolExecutor."""
    create_tools_resp, create_exec_resp = mock_mcp_server_responses

    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = json.dumps(init_response)
    init_mock.json = Mock(return_value=init_response)
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(create_tools_resp())}\n\n"
    tools_mock.raise_for_status = Mock()

    exec_mock = Mock()
    exec_resp = create_exec_resp("e2e_search", {"query": "test"})
    exec_mock.text = json.dumps(exec_resp)
    exec_mock.json = Mock(return_value=exec_resp)
    exec_mock.status_code = 200
    exec_mock.raise_for_status = Mock()

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock, exec_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "e2e-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]

        handler = E2ETestHandler()
        registry = MCPServerRegistry()
        registry.register_server(handler, "http://localhost:8000", enabled=True)

        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            mcp_executor = MCPToolExecutor(registry)
            # Clear cache so it's created fresh with patched settings
            mcp_executor._mcp_client = None
            mcp_executor._tool_cache = None
            tool_executor = ToolExecutor(mcp_executor)

            tool_call = ToolCall(index=0)
            tool_call.id = "call_123"
            tool_call.function_name = "e2e_search"
            tool_call.arguments = json.dumps({"query": "test"})
            tool_call.is_complete = True

            result = await tool_executor.execute_tool_call(tool_call)

            assert result is not None
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["type"] == "e2e-mcp-result"
            assert result[0]["tool_name"] == "e2e_search"


@pytest.mark.asyncio
async def test_tool_merging_with_custom_tools(httpx_client, mock_mcp_server_responses):
    """Test merging custom and MCP tools."""
    create_tools_resp, _ = mock_mcp_server_responses

    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = json.dumps(init_response)
    init_mock.json = Mock(return_value=init_response)
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(create_tools_resp())}\n\n"
    tools_mock.raise_for_status = Mock()

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "e2e-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        mock_settings.deep_research_enabled = False

        handler = E2ETestHandler()
        registry = MCPServerRegistry()
        registry.register_server(handler, "http://localhost:8000", enabled=True)

        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            mcp_tools, _ = await initialize_mcp_for_request()

            custom_tool = Tool(
                type="function",
                function=ToolFunction(
                    name="custom_action",
                    description="Custom action",
                    parameters={"type": "object"},
                ),
            )

            merged = await merge_tools([custom_tool], mcp_tools)

            assert len(merged) == 3  # 1 custom + 2 MCP
            tool_names = {t.function.name for t in merged}
            assert "custom_action" in tool_names
            assert "e2e_search" in tool_names
            assert "e2e_calculate" in tool_names


@pytest.mark.asyncio
async def test_selective_tool_loading(httpx_client, mock_mcp_server_responses):
    """Test loading only specific MCP tools."""
    create_tools_resp, _ = mock_mcp_server_responses

    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = json.dumps(init_response)
    init_mock.json = Mock(return_value=init_response)
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(create_tools_resp())}\n\n"
    tools_mock.raise_for_status = Mock()

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "e2e-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]

        handler = E2ETestHandler()
        registry = MCPServerRegistry()
        registry.register_server(handler, "http://localhost:8000", enabled=True)

        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            mcp_tools, _ = await initialize_mcp_for_request()

            filtered = await merge_tools(None, mcp_tools, include_tools=["e2e_search"])

            assert len(filtered) == 1
            assert filtered[0].function.name == "e2e_search"


@pytest.mark.asyncio
async def test_tool_execution_with_validation_failure(
    httpx_client, mock_mcp_server_responses
):
    """Test that validation failures are handled gracefully."""
    create_tools_resp, _ = mock_mcp_server_responses

    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = json.dumps(init_response)
    init_mock.json = Mock(return_value=init_response)
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(create_tools_resp())}\n\n"
    tools_mock.raise_for_status = Mock()

    # Return result without 'content' field (fails validation)
    exec_resp = {"result": {"data": "invalid"}}
    exec_mock = Mock()
    exec_mock.text = json.dumps(exec_resp)
    exec_mock.json = Mock(return_value=exec_resp)
    exec_mock.status_code = 200
    exec_mock.raise_for_status = Mock()

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock, exec_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "e2e-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]

        handler = E2ETestHandler()
        registry = MCPServerRegistry()
        registry.register_server(handler, "http://localhost:8000", enabled=True)

        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            mcp_executor = MCPToolExecutor(registry)
            # Clear cache so it's created fresh with patched settings
            mcp_executor._mcp_client = None
            mcp_executor._tool_cache = None

            result = await mcp_executor.execute_mcp_tool(
                "e2e_search", {"query": "test"}
            )

            assert result is not None
            assert "error" in result or result != {}


@pytest.mark.asyncio
async def test_multiple_tool_executions(httpx_client, mock_mcp_server_responses):
    """Test executing multiple tools in sequence."""
    create_tools_resp, create_exec_resp = mock_mcp_server_responses

    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = json.dumps(init_response)
    init_mock.json = Mock(return_value=init_response)
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(create_tools_resp())}\n\n"
    tools_mock.raise_for_status = Mock()

    search_resp = create_exec_resp("e2e_search", {"query": "test"})
    calc_resp = create_exec_resp("e2e_calculate", {"expression": "2+2"})

    search_mock = Mock()
    search_mock.text = json.dumps(search_resp)
    search_mock.json = Mock(return_value=search_resp)
    search_mock.status_code = 200
    search_mock.raise_for_status = Mock()

    calc_mock = Mock()
    calc_mock.text = json.dumps(calc_resp)
    calc_mock.json = Mock(return_value=calc_resp)
    calc_mock.status_code = 200
    calc_mock.raise_for_status = Mock()

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(
        side_effect=[init_mock, tools_mock, search_mock, calc_mock]
    )
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "e2e-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]

        handler = E2ETestHandler()
        registry = MCPServerRegistry()
        registry.register_server(handler, "http://localhost:8000", enabled=True)

        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            mcp_executor = MCPToolExecutor(registry)
            # Clear cache so it's created fresh with patched settings
            mcp_executor._mcp_client = None
            mcp_executor._tool_cache = None
            tool_executor = ToolExecutor(mcp_executor)

            call1 = ToolCall(index=0, id="call_1", function_name="e2e_search")
            call1.arguments = json.dumps({"query": "test"})
            call1.is_complete = True

            call2 = ToolCall(index=1, id="call_2", function_name="e2e_calculate")
            call2.arguments = json.dumps({"expression": "2+2"})
            call2.is_complete = True

            results = await tool_executor.execute_tool_calls([call1, call2])

            assert len(results) == 2
            assert "call_1" in results
            assert "call_2" in results
            assert results["call_1"] is not None
            assert results["call_2"] is not None


@pytest.mark.asyncio
async def test_registry_and_executor_integration(
    httpx_client, mock_mcp_server_responses
):
    """Test that registry and executor work together correctly."""
    create_tools_resp, create_exec_resp = mock_mcp_server_responses

    init_response = {"result": {}}
    init_mock = Mock()
    init_mock.text = json.dumps(init_response)
    init_mock.json = Mock(return_value=init_response)
    init_mock.headers = {}
    init_mock.raise_for_status = Mock()

    tools_mock = Mock()
    tools_mock.text = f"event: message\ndata: {json.dumps(create_tools_resp())}\n\n"
    tools_mock.raise_for_status = Mock()

    exec_resp = create_exec_resp("e2e_search", {"query": "integration test"})
    exec_mock = Mock()
    exec_mock.text = json.dumps(exec_resp)
    exec_mock.json = Mock(return_value=exec_resp)
    exec_mock.status_code = 200
    exec_mock.raise_for_status = Mock()

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(side_effect=[init_mock, tools_mock, exec_mock])
    mock_async_client_cm = AsyncMock()
    mock_async_client_cm.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client_cm.__aexit__ = AsyncMock(return_value=None)

    with (patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,):
        mock_settings.mcp_servers = [
            {
                "name": "e2e-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]

        handler = E2ETestHandler()
        registry = MCPServerRegistry()
        registry.register_server(handler, "http://localhost:8000", enabled=True)

        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            executor = MCPToolExecutor(registry)
            # Clear cache so it's created fresh with patched settings
            executor._mcp_client = None
            executor._tool_cache = None

            result = await executor.execute_mcp_tool(
                "e2e_search", {"query": "integration test"}
            )

            assert result["type"] == "e2e-mcp-result"
            assert result["server"] == "e2e-server"
            assert result["tool_name"] == "e2e_search"
