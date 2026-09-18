"""Tests for MCP client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aichat.serve.services.mcp import MCPClient, MCPServerConfig, MCPTool
from aichat.serve.services.mcp.client import has_stdio_mcp_servers
from aichat.serve.services.models import ModelConfig


class TestMCPClient:
    @pytest.fixture
    def mcp_client(self):
        """Create an MCPClient instance"""
        return MCPClient()

    @pytest.fixture
    def mock_async_client(self):
        """Create a mock async client context manager"""
        mock_client = AsyncMock()
        mock_async_client_cm = AsyncMock()
        mock_async_client_cm.__aenter__.return_value = mock_client
        mock_async_client_cm.__aexit__.return_value = None
        return mock_client, mock_async_client_cm

    @pytest.fixture
    def sample_server_config(self):
        """Create a sample server configuration"""
        return MCPServerConfig(
            name="test-server",
            url="https://example.com",
            enabled=True,
        )

    def test_client_initialization(self, mcp_client):
        """Test MCPClient initialization"""
        assert mcp_client.headers["Content-Type"] == "application/json"
        assert mcp_client.headers["Accept"] == "application/json, text/event-stream"
        assert mcp_client.tool_list_payload["method"] == "tools/list"

    @pytest.mark.asyncio
    async def test_fetch_tools_from_server_success_json_rpc(
        self, mcp_client, sample_server_config, mock_async_client
    ):
        """Test fetching tools successfully with JSON-RPC response"""
        mock_client, mock_async_client_cm = mock_async_client

        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "brave_search",
                        "description": "Search the web",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                        "id": "tool_1",
                    },
                    {
                        "name": "calculator",
                        "description": "Perform calculations",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                        },
                    },
                ]
            },
        }

        mock_response = MagicMock()
        mock_response.text = json.dumps(response_data)
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            tools = await mcp_client.fetch_tools_from_server(sample_server_config)

        assert mock_client.post.call_count == 2
        init_call = mock_client.post.call_args_list[0]
        assert init_call[0][0] == "https://example.com/mcp"
        assert init_call[1]["json"]["method"] == "initialize"
        tools_call = mock_client.post.call_args_list[1]
        assert tools_call[0][0] == "https://example.com/mcp"
        assert tools_call[1]["json"]["method"] == "tools/list"
        assert tools_call[1]["json"]["jsonrpc"] == "2.0"

        assert len(tools) == 2
        assert tools[0].name == "brave_search"
        assert tools[0].description == "Search the web"
        assert tools[0].server_name == "test-server"
        assert tools[0].tool_id == "tool_1"
        assert tools[1].name == "calculator"
        assert tools[1].tool_id is None

    @pytest.mark.asyncio
    async def test_fetch_tools_from_server_sse_format(
        self, mcp_client, sample_server_config, mock_async_client
    ):
        """Test fetching tools with Server-Sent Events format"""
        mock_client, mock_async_client_cm = mock_async_client

        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "test_tool",
                        "description": "Test tool",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            },
        }

        sse_response = f"event: message\ndata: {json.dumps(response_data)}\n\n"
        mock_response = MagicMock()
        mock_response.text = sse_response
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            tools = await mcp_client.fetch_tools_from_server(sample_server_config)

        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    @pytest.mark.asyncio
    async def test_fetch_tools_from_server_error_response(
        self, mcp_client, sample_server_config, mock_async_client
    ):
        """Test handling of JSON-RPC error response"""
        mock_client, mock_async_client_cm = mock_async_client

        error_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32600, "message": "Invalid Request"},
        }

        mock_response = MagicMock()
        mock_response.text = json.dumps(error_response)
        mock_response.json.return_value = error_response
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            tools = await mcp_client.fetch_tools_from_server(sample_server_config)

        assert tools == []

    @pytest.mark.asyncio
    async def test_fetch_tools_from_server_http_error(
        self, mcp_client, sample_server_config, mock_async_client
    ):
        """Test handling of HTTP errors"""
        mock_client, mock_async_client_cm = mock_async_client

        mock_client.post.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            tools = await mcp_client.fetch_tools_from_server(sample_server_config)

        assert tools == []

    @pytest.mark.asyncio
    async def test_fetch_tools_from_server_timeout(
        self, mcp_client, sample_server_config, mock_async_client
    ):
        """Test handling of timeout errors"""
        mock_client, mock_async_client_cm = mock_async_client

        mock_client.post.side_effect = httpx.TimeoutException("Request timeout")

        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            tools = await mcp_client.fetch_tools_from_server(sample_server_config)

        assert tools == []

    def test_convert_to_openai_format(self, mcp_client):
        """Test converting MCPTool to OpenAI format"""
        from aichat.protocol.open_ai_protocol import Tool

        mcp_tool = MCPTool(
            name="brave_search",
            server_name="search",
            server_url="https://search.example.com/mcp",
            description="Search the web using Brave Search",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

        openai_tool = mcp_client.convert_to_openai_format(mcp_tool)

        assert isinstance(openai_tool, Tool)
        assert openai_tool.type == "function"
        assert openai_tool.function.name == "brave_search"
        assert openai_tool.function.description == "Search the web using Brave Search"
        assert openai_tool.function.parameters == mcp_tool.input_schema

    def test_convert_to_openai_format_no_description(self, mcp_client):
        """Test conversion with empty description"""
        mcp_tool = MCPTool(
            name="test_tool",
            server_name="test-server",
            server_url="https://example.com/mcp",
            description="",
            input_schema={"type": "object", "properties": {}},
        )

        openai_tool = mcp_client.convert_to_openai_format(mcp_tool)

        assert openai_tool.function.description == "Tool from test-server"

    def test_convert_to_openai_format_empty_schema(self, mcp_client):
        """Test conversion with empty input schema falls back to default"""
        mcp_tool = MCPTool(
            name="test_tool",
            server_name="test-server",
            server_url="https://example.com/mcp",
            description="Test tool",
            input_schema={},
        )

        openai_tool = mcp_client.convert_to_openai_format(mcp_tool)

        # Empty dict is falsy, so it falls back to default schema
        assert openai_tool.function.parameters == {
            "type": "object",
            "properties": {},
        }

    @pytest.mark.asyncio
    async def test_get_tools_no_config(self):
        """Test get_tools with no MCP servers configured"""
        with patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings:
            mock_settings.mcp_servers = []
            mock_settings.deep_research_enabled = False

            # Create client after patching settings
            mcp_client = MCPClient()
            tools = await mcp_client.get_tools()

            assert tools == []

    @pytest.mark.asyncio
    async def test_get_tools_success(self, mock_async_client):
        """Test get_tools with successful server response"""
        mock_client, mock_async_client_cm = mock_async_client

        mock_server_config = {
            "name": "search",
            "url": "https://search.example.com",
            "enabled": True,
        }

        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "brave_search",
                        "description": "Search the web",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "calculator",
                        "description": "Calculate",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ]
            },
        }

        mock_response = MagicMock()
        mock_response.text = json.dumps(response_data)
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        with (
            patch("httpx.AsyncClient", return_value=mock_async_client_cm),
            patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings,
        ):
            mock_settings.mcp_servers = [mock_server_config]
            mock_settings.deep_research_enabled = False

            mcp_client = MCPClient()
            tools = await mcp_client.get_tools()

            assert len(tools) == 2
            assert tools[0].function.name == "brave_search"
            assert tools[1].function.name == "calculator"

    @pytest.mark.asyncio
    async def test_get_tools_disabled_server(self):
        """Test that disabled servers are skipped"""
        mock_server_config = {
            "name": "search",
            "url": "https://search.example.com",
            "enabled": False,
        }

        with patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings:
            mock_settings.mcp_servers = [mock_server_config]
            mock_settings.deep_research_enabled = False

            # Create client after patching settings
            mcp_client = MCPClient()
            tools = await mcp_client.get_tools()

            assert tools == []

    @pytest.mark.asyncio
    async def test_get_tools_deep_research_requires_model_flag(self):
        """deep_research tool is only added when both server flag and model flag are set."""
        with patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings:
            mock_settings.mcp_servers = []
            mock_settings.deep_research_enabled = True

            mcp_client = MCPClient()

            # No model_config — deep_research must NOT be added
            tools = await mcp_client.get_tools(model_config=None)
            assert not any(t.function.name == "deep_research" for t in tools)

            # model_config with deep_research_support=False — must NOT be added
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
                deep_research_support=False,
            )
            tools = await mcp_client.get_tools(model_config=model_cfg)
            assert not any(t.function.name == "deep_research" for t in tools)

            # model_config with deep_research_support=True — MUST be added
            model_cfg_with_dr = ModelConfig(
                **{**model_cfg.__dict__, "deep_research_support": True}
            )
            tools = await mcp_client.get_tools(model_config=model_cfg_with_dr)
            assert any(t.function.name == "deep_research" for t in tools)

    @pytest.mark.asyncio
    async def test_get_tools_deep_research_server_flag_off(self):
        """deep_research tool is never added when the server master switch is off."""
        with patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings:
            mock_settings.mcp_servers = []
            mock_settings.deep_research_enabled = False

            mcp_client = MCPClient()

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
            tools = await mcp_client.get_tools(model_config=model_cfg)
            assert not any(t.function.name == "deep_research" for t in tools)

    def test_stdio_config_parsing(self):
        with patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings:
            mock_settings.mcp_servers = [
                {
                    "name": "echo",
                    "transport": "stdio",
                    "command": "python3",
                    "args": ["server.py"],
                    "env": {"FOO": "bar"},
                    "enabled": True,
                },
                {
                    "name": "search",
                    "url": "https://search.example.com",
                },
            ]
            mcp_client = MCPClient()
            assert mcp_client.servers[0].stdio_argv() == ["python3", "server.py"]
            assert mcp_client.servers[0].env == {"FOO": "bar"}
            assert mcp_client.servers[1].transport == "http"
            assert mcp_client.servers[1].enabled is True

    def test_client_infers_stdio_when_command_set_without_url(self):
        with patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings:
            mock_settings.mcp_servers = [
                {
                    "name": "brave_search",
                    "enabled": True,
                    "command": "docker",
                    "args": ["run", "-i", "--rm", "mcp/example"],
                }
            ]
            mcp_client = MCPClient()
            s = mcp_client.servers[0]
            assert s.transport == "stdio"
            assert s.url == "stdio://brave_search"

    def test_has_stdio_mcp_servers(self):
        assert has_stdio_mcp_servers(
            [{"name": "x", "type": "stdio", "command": "python3", "enabled": True}]
        )
        assert not has_stdio_mcp_servers(
            [{"name": "x", "url": "http://example.com", "enabled": True}]
        )


class TestParseFastMCPTags:
    def test_parse_fastmcp_tags_from_meta(self):
        from aichat.serve.services.mcp.client import _parse_fastmcp_tags

        tags = _parse_fastmcp_tags(
            {
                "name": "brave_faqs_search",
                "meta": {"fastmcp": {"tags": ["dynamic", "brave_faqs"]}},
            }
        )
        assert tags == frozenset({"dynamic", "brave_faqs"})

    def test_parse_fastmcp_tags_from_underscore_meta(self):
        from aichat.serve.services.mcp.client import _parse_fastmcp_tags

        tags = _parse_fastmcp_tags(
            {
                "name": "brave_faqs_search",
                "_meta": {"fastmcp": {"tags": ["dynamic", "brave_faqs"]}},
            }
        )
        assert tags == frozenset({"dynamic", "brave_faqs"})

    def test_parse_fastmcp_tags_missing_meta(self):
        from aichat.serve.services.mcp.client import _parse_fastmcp_tags

        assert _parse_fastmcp_tags({"name": "legacy"}) == frozenset()
        assert _parse_fastmcp_tags({"name": "x", "meta": None}) == frozenset()
        assert _parse_fastmcp_tags({"name": "x", "meta": {}}) == frozenset()
        assert _parse_fastmcp_tags({"name": "x", "_meta": {}}) == frozenset()


class TestGetToolsDynamicLeoFilter:
    @pytest.fixture
    def mock_async_client(self):
        mock_client = AsyncMock()
        mock_async_client_cm = AsyncMock()
        mock_async_client_cm.__aenter__.return_value = mock_client
        mock_async_client_cm.__aexit__.return_value = None
        return mock_client, mock_async_client_cm

    @pytest.fixture
    def sample_server_config(self):
        return MCPServerConfig(
            name="test-server",
            url="https://example.com",
            enabled=True,
        )

    @pytest.mark.asyncio
    async def test_fetch_tools_parses_fastmcp_tags(
        self, mock_async_client, sample_server_config
    ):
        mock_client, mock_async_client_cm = mock_async_client
        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "brave_faqs_search",
                        "description": "FAQ search",
                        "inputSchema": {"type": "object", "properties": {}},
                        "_meta": {"fastmcp": {"tags": ["dynamic", "brave_faqs"]}},
                    },
                    {
                        "name": "brave_web_search",
                        "description": "Web search",
                        "inputSchema": {"type": "object", "properties": {}},
                        "_meta": {"fastmcp": {"tags": ["always"]}},
                    },
                ]
            },
        }
        mock_response = MagicMock()
        mock_response.text = json.dumps(response_data)
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        mcp_client = MCPClient()
        with patch("httpx.AsyncClient", return_value=mock_async_client_cm):
            tools = await mcp_client.fetch_tools_from_server(sample_server_config)

        by_name = {t.name: t for t in tools}
        assert by_name["brave_faqs_search"].tags == frozenset({"dynamic", "brave_faqs"})
        assert by_name["brave_web_search"].tags == frozenset({"always"})

    @pytest.mark.asyncio
    async def test_get_tools_filters_dynamic_tools_by_matched_categories(self):
        catalog = [
            MCPTool(
                name="brave_web_search",
                description="web",
                input_schema={},
                server_name="brave_search",
                server_url="http://localhost",
                tags=frozenset({"always"}),
            ),
            MCPTool(
                name="brave_faqs_search",
                description="faqs",
                input_schema={},
                server_name="brave_search",
                server_url="http://localhost",
                tags=frozenset({"dynamic", "brave_faqs"}),
            ),
            MCPTool(
                name="coding_helper",
                description="code",
                input_schema={},
                server_name="brave_search",
                server_url="http://localhost",
                tags=frozenset({"dynamic", "coding"}),
            ),
        ]

        with patch("aichat.serve.services.mcp.client.mcp_settings") as mock_settings:
            mock_settings.mcp_servers = []
            mock_settings.deep_research_enabled = False
            mcp_client = MCPClient()
            mcp_client.cache.set("mcp_tools_catalog", catalog)

            tools_none = await mcp_client.get_tools(matched_categories=frozenset())
            assert [t.function.name for t in tools_none] == ["brave_web_search"]

            tools_faqs = await mcp_client.get_tools(
                matched_categories=frozenset({"brave_faqs"})
            )
            assert [t.function.name for t in tools_faqs] == [
                "brave_web_search",
                "brave_faqs_search",
            ]

            tools_multi = await mcp_client.get_tools(
                matched_categories=frozenset({"brave_faqs", "coding"})
            )
            assert [t.function.name for t in tools_multi] == [
                "brave_web_search",
                "brave_faqs_search",
                "coding_helper",
            ]

            tools_disabled = await mcp_client.get_tools(
                matched_categories=frozenset(),
                dynamic_leo_enabled=False,
            )
            assert [t.function.name for t in tools_disabled] == [
                "brave_web_search",
                "brave_faqs_search",
                "coding_helper",
            ]
