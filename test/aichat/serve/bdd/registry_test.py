"""BDD: MCP registry base handler + config error handling.

Dup TODOs (same behaviour covered by unit tests):
# TODO: remove test/aichat/serve/services/mcp/test_registry.py#test_register_server test/aichat/serve/services/mcp/test_registry.py:115
# TODO: remove test/aichat/serve/services/mcp/test_registry.py#test_get_all_enabled_servers test/aichat/serve/services/mcp/test_registry.py:164
# TODO: remove test/aichat/serve/services/mcp/test_registry.py#test_validate_and_format_success test/aichat/serve/services/mcp/test_registry.py:202
# TODO: remove test/aichat/serve/services/mcp/test_registry.py#test_validate_and_format_validation_failure test/aichat/serve/services/mcp/test_registry.py:214
# TODO: remove test/aichat/serve/services/mcp/test_registry.py#test_handler_validation_error_handling test/aichat/serve/services/mcp/test_registry.py:237
# TODO: remove test/aichat/serve/services/mcp/test_registry.py#test_handler_formatting_error_handling test/aichat/serve/services/mcp/test_registry.py:250
"""

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

FEATURE = "features/registry.feature"
scenarios(FEATURE)

from aichat.serve.services.mcp.handlers.default import DefaultMCPServerHandler
from aichat.serve.services.mcp.handlers.search import SearchServerHandler
from aichat.serve.services.mcp.registry import (
    MCPServerConfig,
    MCPServerHandler,
    MCPServerRegistry,
)


class BareHandler(MCPServerHandler):
    """Concrete handler exercising only base-class behaviour."""

    def validate_result(self, result):
        return result is not None

    def format_result(self, tool_name, result):
        return {"content": str(result)}

    @property
    def server_name(self):
        return "bare"


@pytest.fixture
def ctx():
    ctx_data = {"base_handler": BareHandler()}
    return ctx_data


@given("a base handler")
def given_base_handler(ctx):
    ctx["base_handler"] = BareHandler()


@given("an empty MCP registry")
def given_registry(ctx):
    ctx["registry"] = MCPServerRegistry()


@given("a base handler and a result with two text items and one non text")
def given_parse_result(ctx):
    ctx["parse_result"] = {
        "content": [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two"},
            {"type": "image", "url": "x"},
        ]
    }


@when("the MCP content is parsed")
def when_parse(ctx):
    ctx["texts"] = ctx["base_handler"]._parse_mcp_content(ctx["parse_result"])


@then("only the two text items are returned")
def then_texts(ctx):
    assert ctx["texts"] == ["one", "two"]


@given(parsers.parse("a base handler and a result of kind {kind}"))
def given_kind_result(ctx, kind):
    if kind == "no_content":
        ctx["parse_result"] = {"other": 1}
    else:  # not_a_dict
        ctx["parse_result"] = "raw string"


@then("the parsed content is a single string")
def then_single_string(ctx):
    assert ctx["texts"] == [str(ctx["parse_result"])]


@then("base tool guidance is empty")
def then_base_guidance(ctx):
    assert ctx["base_handler"].get_tool_guidance() == {}


@then("there are no augmented tools")
def then_no_augmented(ctx):
    assert ctx["base_handler"].get_augmented_tools() == []


@then("base tool message content echoes the result")
def then_base_tmc(ctx):
    assert (
        ctx["base_handler"].get_tool_message_content({"content": "the result"}, None)
        == "the result"
    )


@then("base output content parts are empty")
def then_base_parts(ctx):
    assert ctx["base_handler"].get_output_content_parts({"content": "x"}, None) == []


def _boom_formatter(tool_name, result):
    raise ValueError("formatter exploded")


def _bad_validator(result):
    raise ValueError("bad validator")


@given("a config whose formatter always raises")
def given_bad_formatter(ctx):
    ctx["config"] = MCPServerConfig(
        name="broken",
        url="",
        enabled=True,
        validate_result=lambda r: True,
        format_result=_boom_formatter,
    )


@when("the config formats a result")
def when_config_format(ctx):
    ctx["formatted"] = ctx["config"].format("t", {"raw": 1})


@then("a formatting failed error payload is returned")
def then_format_error(ctx):
    assert ctx["formatted"]["error"].startswith("Formatting failed:")
    assert ctx["formatted"]["type"] == "brave-mcp-result"


@given("a config whose validator always raises")
def given_bad_validator(ctx):
    ctx["config"] = MCPServerConfig(
        name="broken",
        url="",
        enabled=True,
        validate_result=_bad_validator,
        format_result=lambda t, r: {"ok": True},
    )


@when("the config validates a result")
def when_config_validate(ctx):
    ctx["valid"] = ctx["config"].validate({"raw": 1})


@then("the validation returns False")
def then_validation_false(ctx):
    assert ctx["valid"] is False


@given("a registered server brave_search enabled")
def given_search_server(ctx):
    ctx["registry"].register_server(SearchServerHandler(), url="https://m.example")


@given("a registered disabled server other_server")
def given_disabled_server(ctx):

    ctx["registry"].register_server(DefaultMCPServerHandler(), enabled=False)


@when("all servers are listed")
def when_all_servers(ctx):
    ctx["listed"] = ctx["registry"].get_all_servers()


@then("two servers are returned")
def then_two_servers(ctx):
    assert len(ctx["listed"]) == 2


@when("only enabled servers are listed")
def when_enabled_servers(ctx):
    ctx["listed"] = ctx["registry"].get_all_enabled_servers()


@then("one server is returned")
def then_one_server(ctx):
    assert len(ctx["listed"]) == 1


@given("a registered search handler")
def given_registered_search(ctx):
    ctx["registry"].register_server(SearchServerHandler())


@when(parsers.parse("a start message is requested for {server_name}"))
def when_start_message(ctx, server_name):
    ctx["start_message"] = ctx["registry"].get_tool_start_message(
        server_name, "brave_web_search", {}
    )


@then("the handler message is used")
def then_handler_message(ctx):
    assert ctx["start_message"] == "Searching the web..."


@then("the fallback start message is used")
def then_fallback_message(ctx):
    assert ctx["start_message"] == "Running brave_web_search..."
