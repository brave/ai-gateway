"""Tests for MCP Server Registry."""

from typing import Any

import pytest

from aichat.serve.services.mcp.registry import (
    AugmentedToolConfig,
    MCPServerConfig,
    MCPServerHandler,
    MCPServerRegistry,
    get_global_registry,
)


class MockServerHandler(MCPServerHandler):
    """Mock handler for testing."""

    @property
    def server_name(self) -> str:
        return "mock-server"

    def validate_result(self, result: Any) -> bool:
        """Validate that result has 'status' key."""
        return isinstance(result, dict) and "status" in result

    def format_result(self, tool_name: str, result: Any) -> dict:
        """Format result with mock server structure."""
        return {
            "type": "mock-result",
            "tool_name": tool_name,
            "status": result.get("status"),
            "data": result.get("data"),
        }

    def get_tool_guidance(self) -> dict[str, str]:
        return {
            "mock_tool": "Use this tool for mocking",
        }


class AnotherMockHandler(MCPServerHandler):
    """Another mock handler for testing multiple servers."""

    @property
    def server_name(self) -> str:
        return "another-server"

    def validate_result(self, result: Any) -> bool:
        return result is not None

    def format_result(self, tool_name: str, result: Any) -> dict:
        return {
            "type": "another-result",
            "tool": tool_name,
            "content": str(result),
        }


class FailingHandler(MCPServerHandler):
    """Handler that raises exceptions for testing error handling."""

    @property
    def server_name(self) -> str:
        return "failing-server"

    def validate_result(self, result: Any) -> bool:
        raise ValueError("Validation error")

    def format_result(self, tool_name: str, result: Any) -> dict:
        raise ValueError("Formatting error")


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    # Clear singleton
    MCPServerRegistry._instance = None
    return MCPServerRegistry()


@pytest.fixture
def mock_handler():
    """Create a mock handler."""
    return MockServerHandler()


@pytest.fixture
def another_handler():
    """Create another mock handler."""
    return AnotherMockHandler()


def test_registry_singleton():
    """Test that global registry is a singleton."""
    from aichat.serve.services.mcp.registry import (
        get_global_registry,
        reset_global_registry,
    )

    # Reset to ensure clean state
    reset_global_registry()
    registry1 = get_global_registry()
    registry2 = get_global_registry()
    assert registry1 is registry2


def test_global_registry():
    """Test global registry getter."""
    registry1 = get_global_registry()
    registry2 = get_global_registry()
    assert registry1 is registry2


def test_register_server(registry, mock_handler):
    """Test registering a server handler."""
    url = "http://localhost:8000"
    description = "Mock MCP server for testing"

    registry.register_server(mock_handler, url, enabled=True, description=description)

    assert "mock-server" in registry.servers
    config = registry.servers["mock-server"]
    assert config.name == "mock-server"
    assert config.url == url
    assert config.enabled is True
    assert config.description == description
    assert config.tool_guidance == {"mock_tool": "Use this tool for mocking"}


def test_register_multiple_servers(registry, mock_handler, another_handler):
    """Test registering multiple server handlers."""
    registry.register_server(mock_handler, "http://localhost:8000", enabled=True)
    registry.register_server(another_handler, "http://localhost:8001", enabled=True)

    assert len(registry.servers) == 2
    assert "mock-server" in registry.servers
    assert "another-server" in registry.servers


def test_overwrite_existing_server(registry, mock_handler, caplog):
    """Test that re-registering a server logs a warning."""
    registry.register_server(mock_handler, "http://localhost:8000", enabled=True)
    registry.register_server(mock_handler, "http://localhost:9000", enabled=True)

    assert len(registry.servers) == 1
    assert registry.servers["mock-server"].url == "http://localhost:9000"
    assert "already registered" in caplog.text


def test_get_server_config(registry, mock_handler):
    """Test retrieving server configuration."""
    registry.register_server(mock_handler, "http://localhost:8000", enabled=True)

    config = registry.get_server_config("mock-server")
    assert config is not None
    assert config.name == "mock-server"
    assert config.url == "http://localhost:8000"

    # Test non-existent server
    assert registry.get_server_config("non-existent") is None


def test_get_all_enabled_servers(registry, mock_handler, another_handler):
    """Test getting all enabled servers."""
    registry.register_server(mock_handler, "http://localhost:8000", enabled=True)
    registry.register_server(another_handler, "http://localhost:8001", enabled=False)

    enabled_servers = registry.get_all_enabled_servers()
    assert len(enabled_servers) == 1
    assert enabled_servers[0].name == "mock-server"


def test_server_config_validation(registry, mock_handler):
    """Test server config validation."""
    registry.register_server(mock_handler, "http://localhost:8000", enabled=True)
    config = registry.get_server_config("mock-server")

    # Valid result
    valid_result = {"status": "success", "data": "test"}
    assert config.validate(valid_result) is True

    # Invalid result
    invalid_result = {"data": "test"}  # Missing 'status'
    assert config.validate(invalid_result) is False


def test_server_config_formatting(registry, mock_handler):
    """Test server config formatting."""
    registry.register_server(mock_handler, "http://localhost:8000", enabled=True)
    config = registry.get_server_config("mock-server")

    result = {"status": "success", "data": "test data"}
    formatted = config.format("test_tool", result)

    assert formatted["type"] == "mock-result"
    assert formatted["tool_name"] == "test_tool"
    assert formatted["status"] == "success"
    assert formatted["data"] == "test data"


def test_validate_and_format_success(registry, mock_handler):
    """Test successful validation and formatting."""
    registry.register_server(mock_handler, "http://localhost:8000", enabled=True)

    result = {"status": "success", "data": "test"}
    formatted = registry.validate_and_format("mock-server", "test_tool", result)

    assert formatted["type"] == "mock-result"
    assert formatted["tool_name"] == "test_tool"
    assert formatted["status"] == "success"


def test_validate_and_format_validation_failure(registry, mock_handler, caplog):
    """Test validation failure during validate_and_format."""
    registry.register_server(mock_handler, "http://localhost:8000", enabled=True)

    # Result without 'status' key fails validation
    result = {"data": "test"}
    formatted = registry.validate_and_format("mock-server", "test_tool", result)

    assert "error" in formatted
    assert "Result validation failed" in formatted["error"]
    assert "validation failed" in caplog.text.lower()


def test_validate_and_format_nonexistent_server(registry, caplog):
    """Test formatting with non-existent server."""
    result = {"status": "success"}
    formatted = registry.validate_and_format("nonexistent", "test_tool", result)

    assert "error" in formatted
    assert "not registered" in formatted["error"]
    assert "not found in registry" in caplog.text


def test_handler_validation_error_handling(registry, caplog):
    """Test that validation errors are caught and logged."""
    failing_handler = FailingHandler()
    registry.register_server(failing_handler, "http://localhost:8000", enabled=True)

    result = {"status": "success"}
    config = registry.get_server_config("failing-server")

    # Validation error should be caught
    assert config.validate(result) is False
    assert "Validation failed" in caplog.text


def test_handler_formatting_error_handling(registry, caplog):
    """Test that formatting errors are caught and return fallback."""
    failing_handler = FailingHandler()
    registry.register_server(failing_handler, "http://localhost:8000", enabled=True)

    result = {"status": "success"}
    config = registry.get_server_config("failing-server")

    # Formatting error should be caught and return fallback
    formatted = config.format("test_tool", result)
    assert "error" in formatted
    assert "Formatting failed" in formatted["error"]
    assert formatted["type"] == "brave-mcp-result"
    assert "Formatting failed" in caplog.text


def test_augmented_tool_config():
    """Test AugmentedToolConfig data class."""

    def augment_fn(value):
        return value.upper()

    config = AugmentedToolConfig(
        name="upper_search",
        base_tool="search",
        description="Search with uppercase query",
        parameter_name="query",
        augmentation_fn=augment_fn,
    )

    assert config.name == "upper_search"
    assert config.base_tool == "search"
    assert config.augmentation_fn("test") == "TEST"


def test_server_config_is_enabled():
    """Test MCPServerConfig.is_enabled() method."""

    def dummy_validate(result):
        return True

    def dummy_format(tool_name, result):
        return {}

    config_enabled = MCPServerConfig(
        name="test",
        url="http://localhost:8000",
        enabled=True,
        validate_result=dummy_validate,
        format_result=dummy_format,
    )
    assert config_enabled.is_enabled() is True

    config_disabled = MCPServerConfig(
        name="test",
        url="http://localhost:8000",
        enabled=False,
        validate_result=dummy_validate,
        format_result=dummy_format,
    )
    assert config_disabled.is_enabled() is False
