# BDD: bedrock edge paths (aichat/serve/services/bedrock.py).
# test/aichat/serve/services/bedrock_test.py covers sanitizers, cache points,
# pairing validation happy paths, format_tools happy path. Scenarios below target
# previously uncovered branches (119-128, 145, 201-205, 313-317, 332-335).
# No exact duplicates -> no TODO links.


import logging
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve.services import bedrock

FEATURE = "features/bedrock_edge.feature"
scenarios(FEATURE)


@pytest.fixture
def ctx():
    return {}


@given("the bedrock edge harness")
def given_harness(ctx):
    ctx["result"] = None
    ctx["error"] = None


VARIANTS = {
    "plain text": "hello there",
    "whitespace only": "   ",
    "list with text part": [{"type": "text", "text": "hello"}],
    "list with content part": [{"type": "text", "content": "hello"}],
    "list with truthy object part": ["raw text"],
    "list with empty parts": [{"type": "text", "text": ""}],
    "dict part without text": [{"type": "text"}],
    "numeric content": 42,
}


@when(parsers.parse("the assistant content variant {variant} is checked"))
def when_check_content(ctx, variant):
    ctx["result"] = bedrock._assistant_content_is_nonempty(VARIANTS[variant])


@then(parsers.parse("non-emptiness is {result}"))
def then_nonempty(ctx, result):
    assert ctx["result"] == (result == "true")


@when("an empty message list is split")
def when_split_empty(ctx):
    ctx["result"] = bedrock.split_assistant_content_with_tool_calls([])


@then("the split result is empty")
def then_split_empty(ctx):
    assert ctx["result"] == []


@given("an assistant turn with content and tool calls followed by a tool result")
def given_mixed_turn(ctx):
    ctx["messages"] = [
        {
            "role": "assistant",
            "content": "I will search for that",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "results"},
        {"role": "user", "content": "thanks"},
    ]


@when("the messages are split")
def when_split(ctx):
    ctx["result"] = bedrock.split_assistant_content_with_tool_calls(ctx["messages"])


@then("the tool calls come first without content")
def then_split_calls_first(ctx):
    first = ctx["result"][0]
    assert first["role"] == "assistant"
    assert "content" not in first
    assert first["tool_calls"][0]["id"] == "call_1"


@then("the original content becomes a trailing assistant message")
def then_split_content_last(ctx):
    assistants = [m for m in ctx["result"] if m.get("role") == "assistant"]
    # The trailing assistant carries the original content; the tool-calls
    # assistant precedes it.
    assert assistants[-1] == {"role": "assistant", "content": "I will search for that"}
    assert "tool_calls" in assistants[0]


@given("tool calls where a continuation has non-string arguments")
def given_nonstring_continuation(ctx):
    ctx["tool_calls"] = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": 123},
        },
        {"id": "", "function": {"name": "", "arguments": {"rest": "x"}}},
    ]


@when("the split tool calls are merged")
def when_merge(ctx, caplog):
    with caplog.at_level(logging.WARNING, logger="aichat.serve.services.bedrock"):
        ctx["result"] = bedrock._merge_split_tool_calls(ctx["tool_calls"])
    ctx["merge_records"] = [r.message for r in caplog.records]


@then("the continuation is dropped and the warning is logged")
def then_continuation_dropped(ctx):
    assert any("non-string continuation arguments" in m for m in ctx["merge_records"])
    assert ctx["result"] == [ctx["tool_calls"][0]]


@given("a non-iterable message list for pairing validation")
def given_bad_pairing_input(ctx):
    ctx["messages"] = 42


@when("the tool call result pairing is validated")
def when_validate_pairing(ctx):
    ctx["result"] = bedrock.filter_tool_call_result_pairing(ctx["messages"])


@then("the original messages are returned unchanged")
def then_original_returned(ctx):
    assert ctx["result"] == ctx["messages"]


class _FakeFunction:
    """function_def that is neither dict nor pydantic (keys-mapping style)."""

    def __init__(self):
        self._data = {"name": "search", "parameters": {"type": "object"}}

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]


class _DumpableObject:
    """function_def that is an object exposing model_dump (pydantic-like)."""

    def __init__(self):
        self._data = {"name": "search", "parameters": {"type": "object"}}

    def model_dump(self, *args, **kwargs):
        return dict(self._data)

    def __getitem__(self, key):
        return self._data[key]


def _tool_with_dumped_function(function_def):
    # format_tools_for_bedrock only calls tool.model_dump(), so a plain
    # namespace fake is enough and avoids writing into pydantic field
    # storage via object.__setattr__ on a real Tool instance.
    return SimpleNamespace(
        model_dump=lambda *a, **kw: {"type": "function", "function": function_def}
    )


@given("a tool whose dump contains a plain object function")
def given_object_function(ctx):
    ctx["tool"] = _tool_with_dumped_function(_DumpableObject())


@given("a tool whose dump contains a keys-style function")
def given_keys_function(ctx):
    ctx["tool"] = _tool_with_dumped_function(_FakeFunction())


@when("the tools are formatted for bedrock")
def when_format_tools(ctx):
    ctx["result"] = bedrock.format_tools_for_bedrock([ctx["tool"]])


@then("the formatted tool keeps a function mapping")
def then_formatted(ctx):
    formatted = ctx["result"][0]
    assert formatted["type"] == "function"
    function_def = formatted["function"]
    assert function_def["name"] == "search"
    params = function_def["parameters"]
    assert params["additionalProperties"] is True
    assert params["properties"] == {}
    assert params["type"] == "object"
