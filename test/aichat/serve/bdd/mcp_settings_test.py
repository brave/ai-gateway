"""BDD: MCPSettings server list coercion."""

import json
import os

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

FEATURE = "features/mcp_settings.feature"
scenarios(FEATURE)

from aichat.serve.services.mcp.mcp_settings import MCPSettings


@pytest.fixture
def ctx():
    return {}


@given("no MCP server environment variables")
def given_no_env(monkeypatch):
    monkeypatch.delenv("MCP_SERVERS", raising=False)


@when("default MCP settings are constructed")
def when_defaults(ctx):
    ctx["settings"] = MCPSettings()


@then("mcp is enabled by default")
def then_mcp_enabled(ctx):
    assert ctx["settings"].mcp_enabled is True


@then("deep research is disabled with localhost url")
def then_dr_defaults(ctx):
    assert ctx["settings"].deep_research_enabled is False
    expected = os.environ.get("DEEP_RESEARCH_URL", "http://localhost:8080")
    assert ctx["settings"].deep_research_url == expected


@then("deep research limits are 3 iterations 30 queries and 300 seconds")
def then_dr_limits(ctx):
    assert ctx["settings"].deep_research_max_iterations == 3
    assert ctx["settings"].deep_research_max_queries == 30
    assert ctx["settings"].deep_research_max_seconds == 300


VALUES = {
    "none_value": None,
    "empty_string": "",
    "valid_json_list": [{"name": "s", "url": "https://m.example"}],
    "string_dict": {"name": "x"},
    "json_list_string": '[{"name": "s"}]',
    "json_dict_string": '{"name": "x"}',
    "broken_json_string": "{oops",
    "blank_string": "   ",
    "number": 5,
}


@given(parsers.parse("an MCP servers value {raw_value}"))
def given_value(ctx, raw_value):
    ctx["value"] = VALUES[raw_value]


@when("MCP settings are constructed")
def when_construct(ctx):
    ctx["settings"] = MCPSettings(mcp_servers=ctx["value"])


@then(parsers.parse("the coerced server list is {expected}"))
def then_coerced(ctx, expected):
    servers = ctx["settings"].mcp_servers
    if expected == "empty":
        assert servers == []
    elif expected == "passthrough":
        assert servers == [{"name": "s", "url": "https://m.example"}]
    elif expected == "parsed":
        assert servers == json.loads('[{"name": "s"}]')
    else:
        raise AssertionError(f"unknown expected {expected}")
