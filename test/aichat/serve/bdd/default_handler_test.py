"""BDD: DefaultMCPServerHandler."""

import json

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

FEATURE = "features/default_handler.feature"
scenarios(FEATURE)

from aichat.serve.services.mcp.handlers.default import DefaultMCPServerHandler


@pytest.fixture
def ctx():
    return {}


@given("a default server handler")
def given_handler(ctx):
    ctx["handler"] = DefaultMCPServerHandler()


@when(parsers.parse("the handler validates a result of kind {kind}"))
def when_validate(ctx, kind):
    h = ctx["handler"]
    if kind == "dict":
        result = {"a": 1}
    elif kind == "text":
        result = "hello"
    else:
        result = None
    ctx["valid"] = h.validate_result(result)


@then("the result is valid")
def then_valid(ctx):
    assert ctx["valid"] is True


@then("the result is invalid")
def then_invalid(ctx):
    assert ctx["valid"] is False


def _text_result(text):
    return {"content": [{"type": "text", "text": text}]}


@given("a raw default result with json content containing title and url")
def given_title_url(ctx):
    ctx["raw"] = _text_result(
        json.dumps(
            {
                "title": "Page Title",
                "url": "https://p.example",
                "snippet": "Some snippet",
            }
        )
    )


@when(parsers.parse("the result is formatted for tool {tool_name}"))
def when_format(ctx, tool_name):
    ctx["formatted"] = ctx["handler"].format_result(tool_name, ctx["raw"])


@then(r'the default content is "Page Title\nhttps://p.example\nSome snippet"')
def then_flattened(ctx):
    assert ctx["formatted"]["content"] == (
        "Page Title\nhttps://p.example\nSome snippet"
    )


@then("the raw result is preserved")
def then_raw(ctx):
    assert ctx["formatted"]["raw_result"] is ctx["raw"]
    assert ctx["formatted"]["type"] == "brave-mcp-result"
    assert ctx["formatted"]["tool_name"] == "web_fetch"


@given("a raw default result with json content without title")
def given_json_only(ctx):
    ctx["raw"] = _text_result(json.dumps({"depth": 2, "ok": True}))


@then("the default content is pretty printed json")
def then_pretty(ctx):
    assert ctx["formatted"]["content"] == json.dumps({"depth": 2, "ok": True}, indent=2)


@given("a raw default result with plain text content")
def given_plain(ctx):
    ctx["raw"] = _text_result("plain text here")


@then("the default content equals the text")
def then_plain(ctx):
    assert ctx["formatted"]["content"] == "plain text here"


@given("a raw default result with two text items")
def given_two_texts(ctx):
    ctx["raw"] = _text_result("first")
    ctx["raw"]["content"].append({"type": "text", "text": "second"})


@then("the default content joins the two texts")
def then_join(ctx):
    assert ctx["formatted"]["content"] == "first\n\nsecond"


@given("a raw default result with no text items")
def given_no_texts(ctx):
    ctx["raw"] = {"content": [{"type": "image", "url": "x"}]}


@then("the default content is empty")
def then_empty(ctx):
    assert ctx["formatted"]["content"] == ""


@then(parsers.parse("the server name is {name}"))
def then_server_name(ctx, name):
    assert ctx["handler"].server_name == name


@then("tool guidance is empty")
def then_guidance_empty(ctx):
    assert ctx["handler"].get_tool_guidance() == {}


@then("there are no augmented tools")
def then_no_augmented(ctx):
    assert ctx["handler"].get_augmented_tools() == []
