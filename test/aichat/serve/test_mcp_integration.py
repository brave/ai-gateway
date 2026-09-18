"""Tests for MCP integration helpers."""

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import pytest_asyncio

import aichat.serve.mcp_integration as mcp_integration_module
from aichat.protocol.open_ai_protocol import Tool, ToolFunction
from aichat.serve.mcp_integration import (
    _ensure_registry_initialized,
    get_tool_guidance,
    initialize_mcp_for_request,
    merge_tools,
)
from aichat.serve.services.mcp.registry import (
    MCPServerHandler,
    get_global_registry,
    reset_global_registry,
)
from aichat.serve.services.models import ModelConfig


class TestHandler(MCPServerHandler):
    """Test handler with tool guidance."""

    @property
    def server_name(self) -> str:
        return "test-integration-server"

    def validate_result(self, result) -> bool:
        return True

    def format_result(self, tool_name: str, result: dict) -> dict:
        return {"formatted": True, "tool": tool_name}

    def get_tool_guidance(self) -> dict:
        return {
            "search": "Use search to find information",
            "calculate": "Use calculate for math operations",
        }


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test."""
    reset_global_registry()
    # Reset the _initialized flag so handlers can be re-registered
    mcp_integration_module._initialized = False
    yield
    reset_global_registry()
    mcp_integration_module._initialized = False


@pytest_asyncio.fixture
async def httpx_client():
    """Create a mock httpx client."""
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.mark.asyncio
async def test_ensure_registry_initialized():
    """Test registry initialization."""
    _ensure_registry_initialized()

    registry = get_global_registry()
    assert registry is not None
    assert len(registry.servers) > 0
    assert "default" in registry.handlers


@pytest.mark.asyncio
async def test_initialize_mcp_for_request():
    """Test MCP initialization for a request."""
    tools_response = {
        "result": {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "Test tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                }
            ]
        }
    }

    init_response = {
        "jsonrpc": "2.0",
        "id": 0,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "test-server", "version": "1.0.0"},
        },
    }

    mock_tools_response = Mock()
    tools_response_text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    mock_tools_response.text = tools_response_text
    mock_tools_response.raise_for_status = Mock()
    mock_tools_response.headers = {}

    mock_init_response = Mock()
    mock_init_response.text = json.dumps(init_response)
    mock_init_response.raise_for_status = Mock()
    mock_init_response.headers = {}
    mock_init_response.json.return_value = init_response

    async def mock_post(*args, **kwargs):
        """Mock POST responses for initialize and tools/list."""
        json_data = kwargs.get("json", {})

        if json_data.get("method") == "initialize":
            return mock_init_response
        elif json_data.get("method") == "tools/list":
            return mock_tools_response
        return mock_tools_response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(side_effect=mock_post)

    mock_executor = Mock()
    mock_executor.is_mcp_tool = AsyncMock(return_value=False)

    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_mcp_settings,
        patch("aichat.serve.mcp_integration._ensure_registry_initialized"),
        patch("httpx.AsyncClient", return_value=mock_client),
        patch(
            "aichat.serve.mcp_integration.MCPToolExecutor",
            return_value=mock_executor,
        ),
    ):
        mock_mcp_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        mock_mcp_settings.deep_research_enabled = False
        # Reset registry and ensure it's initialized with only test server
        registry = get_global_registry()
        registry._handlers.clear()
        registry._configs.clear()
        registry.register_server(TestHandler(), "http://localhost:8000", enabled=True)

        mcp_tools, executor = await initialize_mcp_for_request()

        assert len(mcp_tools) == 1
        assert mcp_tools[0].function.name == "test_tool"
        assert executor is not None


@pytest.mark.asyncio
async def test_initialize_mcp_for_request_error(caplog):
    """Test MCP initialization handles errors gracefully."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))

    mock_executor = Mock()
    mock_executor.is_mcp_tool = AsyncMock(return_value=False)

    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_mcp_settings,
        patch("aichat.serve.mcp_integration._ensure_registry_initialized"),
        patch("httpx.AsyncClient", return_value=mock_client),
        patch(
            "aichat.serve.mcp_integration.MCPToolExecutor",
            return_value=mock_executor,
        ),
    ):
        mock_mcp_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        mock_mcp_settings.deep_research_enabled = False
        registry = get_global_registry()
        registry._handlers.clear()
        registry._configs.clear()

        mcp_tools, executor = await initialize_mcp_for_request()

        assert mcp_tools == []
        assert executor is not None


@pytest.mark.asyncio
async def test_merge_tools_basic():
    """Test basic tool merging."""
    custom_tool = Tool(
        type="function",
        function=ToolFunction(
            name="custom_tool",
            description="Custom tool",
            parameters={"type": "object"},
        ),
    )

    mcp_tool = Tool(
        type="function",
        function=ToolFunction(
            name="mcp_tool",
            description="MCP tool",
            parameters={"type": "object"},
        ),
    )

    merged = await merge_tools([custom_tool], [mcp_tool])

    assert len(merged) == 2
    assert merged[0].function.name == "custom_tool"
    assert merged[1].function.name == "mcp_tool"


@pytest.mark.asyncio
async def test_merge_tools_no_custom_tools():
    """Test merging with no custom tools."""
    mcp_tool = Tool(
        type="function",
        function=ToolFunction(
            name="mcp_tool",
            description="MCP tool",
            parameters={"type": "object"},
        ),
    )

    merged = await merge_tools(None, [mcp_tool])

    assert len(merged) == 1
    assert merged[0].function.name == "mcp_tool"


@pytest.mark.asyncio
async def test_merge_tools_with_include_only():
    """Test tool merging with include_only filter."""
    mcp_tool1 = Tool(
        type="function",
        function=ToolFunction(
            name="search",
            description="Search tool",
            parameters={"type": "object"},
        ),
    )

    mcp_tool2 = Tool(
        type="function",
        function=ToolFunction(
            name="calculate",
            description="Calculate tool",
            parameters={"type": "object"},
        ),
    )

    merged = await merge_tools(None, [mcp_tool1, mcp_tool2], include_tools=["search"])

    assert len(merged) == 1
    assert merged[0].function.name == "search"


@pytest.mark.asyncio
async def test_merge_tools_with_exclude():
    """Test tool merging with exclude filter."""
    mcp_tool1 = Tool(
        type="function",
        function=ToolFunction(
            name="search",
            description="Search tool",
            parameters={"type": "object"},
        ),
    )

    mcp_tool2 = Tool(
        type="function",
        function=ToolFunction(
            name="calculate",
            description="Calculate tool",
            parameters={"type": "object"},
        ),
    )

    merged = await merge_tools(
        None, [mcp_tool1, mcp_tool2], exclude_tools=["calculate"]
    )

    assert len(merged) == 1
    assert merged[0].function.name == "search"


@pytest.mark.asyncio
async def test_merge_tools_include_and_exclude():
    """Test that include_only takes precedence over exclude."""
    mcp_tool1 = Tool(
        type="function",
        function=ToolFunction(
            name="search",
            description="Search tool",
            parameters={"type": "object"},
        ),
    )

    mcp_tool2 = Tool(
        type="function",
        function=ToolFunction(
            name="calculate",
            description="Calculate tool",
            parameters={"type": "object"},
        ),
    )

    mcp_tool3 = Tool(
        type="function",
        function=ToolFunction(
            name="translate",
            description="Translate tool",
            parameters={"type": "object"},
        ),
    )

    # include_only should filter first, then exclude applies
    merged = await merge_tools(
        None,
        [mcp_tool1, mcp_tool2, mcp_tool3],
        include_tools=["search", "calculate"],
        exclude_tools=["calculate"],
    )

    assert len(merged) == 1
    assert merged[0].function.name == "search"


@pytest.mark.asyncio
async def test_merge_tools_duplicate_names():
    """Test that custom tools take precedence over MCP tools with same name."""
    custom_tool = Tool(
        type="function",
        function=ToolFunction(
            name="search",
            description="Custom search",
            parameters={"type": "object"},
        ),
    )

    mcp_tool = Tool(
        type="function",
        function=ToolFunction(
            name="search",
            description="MCP search",
            parameters={"type": "object"},
        ),
    )

    merged = await merge_tools([custom_tool], [mcp_tool])

    assert len(merged) == 1
    assert merged[0].function.name == "search"
    assert merged[0].function.description == "Custom search"


@pytest.mark.asyncio
async def test_merge_tools_empty_lists():
    """Test merging empty tool lists."""
    merged = await merge_tools(None, [])
    assert merged == []

    merged = await merge_tools([], [])
    assert merged == []


@pytest.mark.asyncio
async def test_get_tool_guidance():
    """Test getting tool guidance from registry."""
    tools_response = {
        "result": {
            "tools": [
                {
                    "name": "search",
                    "description": "Search tool",
                    "inputSchema": {"type": "object"},
                }
            ]
        }
    }

    init_response = {
        "jsonrpc": "2.0",
        "id": 0,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {
                "name": "test-integration-server",
                "version": "1.0.0",
            },
        },
    }

    mock_tools_response = Mock()
    tools_response_text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    mock_tools_response.text = tools_response_text
    mock_tools_response.raise_for_status = Mock()
    mock_tools_response.headers = {}

    mock_init_response = Mock()
    mock_init_response.text = json.dumps(init_response)
    mock_init_response.raise_for_status = Mock()
    mock_init_response.headers = {}
    mock_init_response.json.return_value = init_response

    async def mock_post(*args, **kwargs):
        """Mock POST responses for initialize and tools/list."""
        json_data = kwargs.get("json", {})

        if json_data.get("method") == "initialize":
            return mock_init_response
        elif json_data.get("method") == "tools/list":
            return mock_tools_response
        return mock_tools_response

    mock_httpx_client = AsyncMock()
    mock_httpx_client.post = AsyncMock(side_effect=mock_post)
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=None)

    def create_mock_client(*args, **kwargs):
        return mock_httpx_client

    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_mcp_settings,
        patch("aichat.serve.mcp_integration._ensure_registry_initialized"),
        patch("httpx.AsyncClient", side_effect=create_mock_client),
    ):
        mock_mcp_settings.mcp_servers = [
            {
                "name": "test-integration-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        mock_mcp_settings.deep_research_enabled = False
        # Reset registry and ensure it's initialized with test handler
        registry = get_global_registry()
        registry._handlers.clear()
        registry._configs.clear()
        registry.register_server(TestHandler(), "http://localhost:8000", enabled=True)

        guidance = await get_tool_guidance()

        assert "search" in guidance
        assert guidance["search"] == "Use search to find information"
        assert "calculate" in guidance
        assert guidance["calculate"] == "Use calculate for math operations"


@pytest.mark.asyncio
async def test_get_tool_guidance_error(caplog):
    """Test that get_tool_guidance handles errors gracefully."""
    # Create a mock httpx client that raises an error
    mock_httpx_client = AsyncMock()
    mock_httpx_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=None)

    def create_mock_client(*args, **kwargs):
        return mock_httpx_client

    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_mcp_settings,
        patch("aichat.serve.mcp_integration._ensure_registry_initialized"),
        patch("httpx.AsyncClient", side_effect=create_mock_client),
    ):
        mock_mcp_settings.mcp_servers = [
            {
                "name": "test-server",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        mock_mcp_settings.deep_research_enabled = False
        registry = get_global_registry()
        registry._handlers.clear()
        registry._configs.clear()

        with patch(
            "aichat.serve.services.mcp.client.MCPClient.get_tools_with_guidance",
            side_effect=Exception("Test error"),
        ):
            guidance = await get_tool_guidance()

        assert guidance == {}
        assert "Failed to get tool guidance" in caplog.text


@pytest.mark.asyncio
async def test_merge_tools_preserves_order():
    """Test that tool merging preserves order (custom tools first)."""
    custom_tools = [
        Tool(
            type="function",
            function=ToolFunction(
                name="custom1",
                description="Custom 1",
                parameters={"type": "object"},
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="custom2",
                description="Custom 2",
                parameters={"type": "object"},
            ),
        ),
    ]

    mcp_tools = [
        Tool(
            type="function",
            function=ToolFunction(
                name="mcp1",
                description="MCP 1",
                parameters={"type": "object"},
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="mcp2",
                description="MCP 2",
                parameters={"type": "object"},
            ),
        ),
    ]

    merged = await merge_tools(custom_tools, mcp_tools)

    assert len(merged) == 4
    assert merged[0].function.name == "custom1"
    assert merged[1].function.name == "custom2"
    assert merged[2].function.name == "mcp1"
    assert merged[3].function.name == "mcp2"


@pytest.mark.asyncio
async def test_merge_tools_all_filters():
    """Test comprehensive filtering scenarios."""
    tools = [
        Tool(
            type="function",
            function=ToolFunction(name=f"tool{i}", description=f"Tool {i}"),
        )
        for i in range(5)
    ]

    merged = await merge_tools(None, tools, include_tools=["tool1", "tool3"])
    assert len(merged) == 2
    assert {t.function.name for t in merged} == {"tool1", "tool3"}

    merged = await merge_tools(None, tools, exclude_tools=["tool0", "tool4"])
    assert len(merged) == 3
    assert {t.function.name for t in merged} == {"tool1", "tool2", "tool3"}

    merged = await merge_tools(
        None,
        tools,
        include_tools=["tool1", "tool2", "tool3"],
        exclude_tools=["tool2"],
    )
    assert len(merged) == 2
    assert {t.function.name for t in merged} == {"tool1", "tool3"}


# Deep Research Integration Tests


@pytest.mark.asyncio
async def test_initialize_mcp_for_request_with_model_config():
    """deep_research tool is included when model_config.deep_research_support is True."""
    mock_executor = Mock()

    model_cfg = ModelConfig(
        model_id="test",
        upstream_model="test",
        backend="litellm",
        api_base=None,
        api_key=None,
        inference_profile=None,
        system_prompt_support=None,
        prompt_caching_support=None,
        prompt_caching_enabled=None,
        tool_support=True,
        image_support=False,
        audio_support=False,
        video_support=False,
        file_support=True,
        friendly_name=None,
        maker=None,
        max_tokens=None,
        max_tokens_premium=None,
        conversation_token_limit=None,
        conversation_token_limit_premium=None,
        free=True,
        key=None,
        rate_limit=None,
        rate_limit_interval_seconds=None,
        max_pages=None,
        max_pages_premium=None,
        extra_body=None,
        deep_research_support=True,
    )

    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_mcp_settings,
        patch("aichat.serve.mcp_integration._ensure_registry_initialized"),
        patch(
            "aichat.serve.mcp_integration.MCPToolExecutor",
            return_value=mock_executor,
        ),
    ):
        mock_mcp_settings.mcp_servers = []
        mock_mcp_settings.deep_research_enabled = True

        mcp_tools, executor = await initialize_mcp_for_request(model_config=model_cfg)

        assert any(t.function.name == "deep_research" for t in mcp_tools)
        assert executor is not None


@pytest.mark.asyncio
async def test_initialize_mcp_for_request_no_model_config_skips_deep_research():
    """deep_research tool is not included when no model_config is passed."""
    mock_executor = Mock()

    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_mcp_settings,
        patch("aichat.serve.mcp_integration._ensure_registry_initialized"),
        patch(
            "aichat.serve.mcp_integration.MCPToolExecutor",
            return_value=mock_executor,
        ),
    ):
        mock_mcp_settings.mcp_servers = []
        mock_mcp_settings.deep_research_enabled = True

        mcp_tools, _ = await initialize_mcp_for_request()

        assert not any(t.function.name == "deep_research" for t in mcp_tools)


@pytest.mark.asyncio
async def test_initialize_mcp_for_request_filters_by_matched_categories():
    """Dynamic tools are filtered by matched Dynamic Leo category tags."""
    catalog_tools = {
        "result": {
            "tools": [
                {
                    "name": "brave_web_search",
                    "description": "Web search",
                    "inputSchema": {"type": "object", "properties": {}},
                    "_meta": {"fastmcp": {"tags": ["always"]}},
                },
                {
                    "name": "brave_faqs_search",
                    "description": "FAQ search",
                    "inputSchema": {"type": "object", "properties": {}},
                    "_meta": {"fastmcp": {"tags": ["dynamic", "brave_faqs"]}},
                },
            ]
        }
    }

    init_response = {
        "jsonrpc": "2.0",
        "id": 0,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "test-server", "version": "1.0.0"},
        },
    }

    mock_tools_response = Mock()
    mock_tools_response.text = f"event: message\ndata: {json.dumps(catalog_tools)}\n\n"
    mock_tools_response.raise_for_status = Mock()
    mock_tools_response.headers = {}

    mock_init_response = Mock()
    mock_init_response.text = json.dumps(init_response)
    mock_init_response.raise_for_status = Mock()
    mock_init_response.headers = {}
    mock_init_response.json.return_value = init_response

    async def mock_post(*args, **kwargs):
        json_data = kwargs.get("json", {})
        if json_data.get("method") == "initialize":
            return mock_init_response
        return mock_tools_response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_executor = Mock()

    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_mcp_settings,
        patch("aichat.serve.mcp_integration._ensure_registry_initialized"),
        patch("httpx.AsyncClient", return_value=mock_client),
        patch(
            "aichat.serve.mcp_integration.MCPToolExecutor",
            return_value=mock_executor,
        ),
        patch("aichat.serve.mcp_integration.get_shared_mcp_client", return_value=None),
    ):
        mock_mcp_settings.mcp_servers = [
            {
                "name": "brave_search",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ]
        mock_mcp_settings.deep_research_enabled = False
        registry = get_global_registry()
        registry._handlers.clear()
        registry._configs.clear()
        registry.register_server(TestHandler(), "http://localhost:8000", enabled=True)

        without_faqs, _ = await initialize_mcp_for_request(
            matched_categories=frozenset(),
            dynamic_leo_enabled=True,
        )
        assert [t.function.name for t in without_faqs] == ["brave_web_search"]

        with_faqs, _ = await initialize_mcp_for_request(
            matched_categories=frozenset({"brave_faqs"}),
            dynamic_leo_enabled=True,
        )
        assert [t.function.name for t in with_faqs] == [
            "brave_web_search",
            "brave_faqs_search",
        ]


@pytest.mark.asyncio
async def test_deep_research_handler_registered_when_enabled():
    """Test that deep research handler is registered when enabled."""
    with patch("aichat.serve.mcp_integration.mcp_settings") as mock_mcp_settings:
        mock_mcp_settings.deep_research_enabled = True

        # Reset initialized flag and registry
        mcp_integration_module._initialized = False
        reset_global_registry()

        _ensure_registry_initialized()

        registry = get_global_registry()
        assert "deep_research" in registry.handlers


@pytest.mark.asyncio
async def test_deep_research_handler_not_registered_when_disabled():
    """Test that deep research handler is not registered when disabled."""
    with patch("aichat.serve.mcp_integration.mcp_settings") as mock_mcp_settings:
        mock_mcp_settings.deep_research_enabled = False

        # Reset initialized flag and registry
        mcp_integration_module._initialized = False
        reset_global_registry()

        _ensure_registry_initialized()

        registry = get_global_registry()
        assert "deep_research" not in registry.handlers


@pytest.mark.asyncio
async def test_tool_guidance_includes_deep_research_when_enabled():
    """Test that tool guidance includes deep_research when enabled."""
    # Mock MCP server response
    tools_response = {
        "result": {
            "tools": [
                {
                    "name": "deep_research",
                    "description": "Deep research tool",
                    "inputSchema": {"type": "object"},
                }
            ]
        }
    }

    init_response = {
        "jsonrpc": "2.0",
        "id": 0,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {
                "name": "deep_research",
                "version": "1.0.0",
            },
        },
    }

    mock_tools_response = Mock()
    tools_response_text = f"event: message\ndata: {json.dumps(tools_response)}\n\n"
    mock_tools_response.text = tools_response_text
    mock_tools_response.raise_for_status = Mock()
    mock_tools_response.headers = {}

    mock_init_response = Mock()
    mock_init_response.text = json.dumps(init_response)
    mock_init_response.raise_for_status = Mock()
    mock_init_response.headers = {}
    mock_init_response.json.return_value = init_response

    async def mock_post(*args, **kwargs):
        """Mock POST responses."""
        json_data = kwargs.get("json", {})
        if json_data.get("method") == "initialize":
            return mock_init_response
        return mock_tools_response

    mock_httpx_client = AsyncMock()
    mock_httpx_client.post = AsyncMock(side_effect=mock_post)
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=None)

    def create_mock_client(*args, **kwargs):
        return mock_httpx_client

    with (
        patch("aichat.serve.services.mcp.client.mcp_settings") as mock_mcp_settings,
        patch("aichat.serve.mcp_integration._ensure_registry_initialized"),
        patch("httpx.AsyncClient", side_effect=create_mock_client),
    ):
        mock_mcp_settings.deep_research_enabled = True
        mock_mcp_settings.mcp_servers = [
            {
                "name": "deep_research",
                "url": "http://localhost:3011",
                "enabled": True,
            }
        ]

        # Reset and re-initialize with deep research handler
        registry = get_global_registry()
        registry._handlers.clear()
        registry._configs.clear()

        # Import and register the deep research handler
        from aichat.serve.services.mcp.handlers.deep_research import (
            DeepResearchServerHandler,
        )

        registry.register_server(
            DeepResearchServerHandler(), "http://localhost:3011", enabled=True
        )

        guidance = await get_tool_guidance()

        # Should include deep_research guidance
        assert "deep_research" in guidance
        assert "comprehensive" in guidance["deep_research"].lower()
