"""BDD: MCP tool execution.

Dup TODOs (same behaviour covered by unit tests):
# TODO: remove test/aichat/serve/test_mcp_tool_execution.py#test_tool_error_event_generated_on_failure test/aichat/serve/test_mcp_tool_execution.py:96
# TODO: remove test/aichat/serve/test_mcp_tool_execution.py#test_error_chunk_yielded_on_failure test/aichat/serve/test_mcp_tool_execution.py:451
# TODO: remove test/aichat/serve/test_mcp_tool_execution.py#test_error_chunk_yielded_on_exception test/aichat/serve/test_mcp_tool_execution.py:515
# TODO: remove test/aichat/serve/test_mcp_tool_execution.py#test_error_chunk_format test/aichat/serve/test_mcp_tool_execution.py:581
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

FEATURE = "features/mcp_tool_execution.feature"
scenarios(FEATURE)

from aichat.protocol.open_ai_protocol import (
    ToolCall as OpenAIToolCall,
)
from aichat.protocol.open_ai_protocol import (
    ToolCallFunction,
    ToolMessage,
    UserMessage,
)
from aichat.serve import mcp_tool_execution
from aichat.serve.services.mcp.handlers.search import SearchServerHandler
from aichat.serve.services.mcp.registry import (
    get_global_registry,
    reset_global_registry,
)
from aichat.serve.tool_parser import ToolCall


@pytest.fixture
def ctx():
    return {}


@pytest.fixture(autouse=True)
def cleanup_registry():
    yield
    reset_global_registry()


def _tool_call(name="search", arguments='{"query":"x"}', tid="call_1", index=0):
    return ToolCall(index=index, id=tid, function_name=name, arguments=arguments)


def _drain(agen):
    async def run():
        out = []
        async for item in agen:
            out.append(item)
        return out

    return asyncio.run(run())


def _fake_executor(result):
    """Patch mcp_tool_execution.ToolExecutor with a stub returning result."""

    class _FakeToolExecutor:
        def __init__(self, mcp_executor):
            pass

        async def execute_tool_call(self, tool_call, model=None):
            return result

    return _FakeToolExecutor


def _mcp_executor():
    ex = Mock()
    ex._tool_cache = {"tools": True}
    return ex


@given("an MCP tool execution harness")
def given_harness(ctx):
    reset_global_registry()
    ctx["events"] = None


@when("the tools are executed and streamed")
def when_execute_tools(ctx, monkeypatch):
    monkeypatch.setattr(
        mcp_tool_execution, "ToolExecutor", _fake_executor(ctx["tool_result"])
    )
    ctx["events"] = _drain(
        mcp_tool_execution.execute_tools_and_stream_events(
            [ctx["tool_call"]],
            _mcp_executor(),
            "conv-1",
            "actual-model",
            "request-model",
        )
    )


@given("a tool call for search")
def given_search_call(ctx):
    ctx["tool_call"] = _tool_call()


@given("the executor returns an error list result")
def given_error_list(ctx):
    ctx["tool_result"] = [{"type": "error", "content": "Tool exploded"}]


@then('an error tool message with content "Tool exploded" is yielded')
def then_error_tool_message(ctx):
    msgs = [m for _, m in ctx["events"] if isinstance(m, ToolMessage)]
    assert any(m.content == "Tool exploded" for m in msgs)


@then("a tool error event is yielded")
def then_error_event(ctx):
    events = [e for e, _ in ctx["events"] if e]
    assert any("brave-chat.toolError" in e for e in events)


@then("an error output chunk is yielded")
def then_error_chunk(ctx):
    chunks = [c for _, c in ctx["events"] if isinstance(c, dict)]
    assert chunks, "expected an error output chunk"
    part = chunks[0]["choices"][0]["delta"]["tool_calls"][0]["output_content"]
    assert part == [{"type": "text", "text": "Tool exploded"}]


@given(parsers.parse("a raw tool result of kind {kind}"))
def given_raw_result(ctx, kind):
    results = {
        "falsy": None,
        "dict_content": {"content": "x"},
        "dict_no_content": {"a": 1},
        "plain": "hi",
    }
    ctx["tool_result"] = results[kind]


@when("the tool result is processed")
def when_process_result(ctx):
    ctx["processed"] = mcp_tool_execution._process_tool_result(ctx["tool_result"])


@then(parsers.parse("the processed content is {expected}"))
def then_processed_content(ctx, expected):
    want = {
        "empty": "",
        "content_plus_newline": "x\n",
        "json_dump": json.dumps({"a": 1}) + "\n",
        "string_conversion": "hi\n",
    }[expected.replace(" ", "_")]
    assert ctx["processed"] == want


@when("an output chunk is created with no parts")
def when_output_chunk_no_parts(ctx):
    ctx["output_chunk"] = mcp_tool_execution._create_tool_output_chunk_from_parts(
        _tool_call(), [], "conv-1", "model"
    )


@then("the output chunk is None")
def then_output_chunk_none(ctx):
    assert ctx["output_chunk"] is None


@given("a broken chunk constructor")
def given_broken_constructor(monkeypatch):
    monkeypatch.setattr(
        mcp_tool_execution,
        "ChatCompletionChunk",
        MagicMock(side_effect=Exception("boom")),
    )


@when("an output chunk is created with parts")
def when_output_chunk_parts(ctx):
    ctx["output_chunk"] = mcp_tool_execution._create_tool_output_chunk_from_parts(
        _tool_call(), [{"type": "text", "text": "hi"}], "conv-1", "model"
    )


@when("an error chunk is created")
def when_error_chunk(ctx):
    ctx["error_chunk"] = mcp_tool_execution._create_tool_error_chunk(
        _tool_call(), "err", "conv-1", "model"
    )


@then("the error chunk is None")
def then_error_chunk_none(ctx):
    assert ctx["error_chunk"] is None


@given("no brave search handler registered")
def given_no_search_handler(ctx):
    reset_global_registry()


@when("web sources are formatted")
def when_format_web_sources(ctx):
    ctx.setdefault("sources", [{"title": "Src", "url": "https://src.example"}])
    ctx["formatted"] = mcp_tool_execution._format_web_sources(
        ctx["sources"], ctx.get("query")
    )


@then("the fallback count text is used")
def then_fallback_text(ctx):
    assert ctx["formatted"] == f"Found {len(ctx['sources'])} sources"


@given("a registered search handler")
def given_search_handler(ctx):
    reset_global_registry()
    get_global_registry().register_server(SearchServerHandler())


@given("one client web source")
def given_one_web_source(ctx):
    ctx["sources"] = [{"title": "Example", "url": "https://example.com"}]
    ctx["query"] = "weather"


@then("the formatted handler text is used")
def then_handler_text(ctx):
    assert ctx["formatted"] != f"Found {len(ctx['sources'])} sources"
    assert "Example" in ctx["formatted"]


@when("a tool end event is created for call_9 done_tool")
def when_tool_end_event(ctx):
    ctx["end_event"] = mcp_tool_execution._create_tool_end_event(
        "call_9", "done_tool", "conv-1", "model"
    )


@then("the end event carries tool_end")
def then_end_event(ctx):
    assert "brave-chat.toolEnd" in ctx["end_event"]
    assert "done_tool" in ctx["end_event"]


@given("a tool message with a dict part without text")
def given_dict_part(ctx):
    ctx["tool_message"] = ToolMessage.model_construct(
        role="tool", tool_call_id="t1", content=[{"type": "custom", "x": 1}]
    )


@when("the tool message is simplified")
def when_simplify(ctx):
    ctx["simplified"] = mcp_tool_execution.simplify_tool_message_for_llm(
        ctx["tool_message"]
    )


@then("the part is stringified into the content")
def then_part_stringified(ctx):
    assert str({"type": "custom", "x": 1}) in ctx["simplified"]["content"]


@given("a tool message with an unknown object part")
def given_unknown_part(ctx):
    ctx["tool_message"] = ToolMessage.model_construct(
        role="tool", tool_call_id="t2", content=[object()]
    )


@then("the unknown part is skipped")
def then_unknown_skipped(ctx):
    assert ctx["simplified"]["content"] == "Tool completed"


@given("a tool message with integer content")
def given_integer_content(ctx):
    ctx["tool_message"] = ToolMessage.model_construct(
        role="tool", tool_call_id="t3", content=7
    )


@then("the content is the string form")
def then_string_form(ctx):
    assert ctx["simplified"]["content"] == "7"


@given("a streaming tool call with broken arguments")
def given_broken_args(ctx):
    ctx["streaming_calls"] = [_tool_call(name="search", arguments="{bad")]


@when("the streaming tool is executed and proxied")
def when_streaming_proxied(ctx):
    ctx["events"] = _drain(
        mcp_tool_execution.execute_streaming_tool_and_proxy_events(
            ctx["streaming_calls"][0], "conv-1", "model"
        )
    )


@then("a tool start event is yielded first")
def then_start_first(ctx):
    first_event, first_msg = ctx["events"][0]
    assert first_msg is None
    assert "brave-chat.toolStart" in first_event


@then("an unknown streaming tool error follows")
def then_unknown_tool_error(ctx):
    second_event, _ = ctx["events"][1]
    assert "brave-chat.toolError" in second_event
    assert "Unknown streaming tool: search" in second_event


@given("a streaming deep research call")
def given_streaming_deep_research(ctx, monkeypatch):
    ctx["streaming_calls"] = [
        _tool_call(name="deep_research", arguments='{"query":"q"}')
    ]
    forwarded = [
        ("data: event-one\n\n", None),
        (None, ToolMessage(role="tool", tool_call_id="call_1", content="done")),
    ]

    async def fake_deep_research(*args, **kwargs):
        for item in forwarded:
            yield item

    monkeypatch.setattr(
        mcp_tool_execution, "execute_deep_research_streaming", fake_deep_research
    )
    ctx["forwarded"] = forwarded


@when("the streaming tool proxies events")
def when_proxy_events(ctx):
    ctx["events"] = _drain(
        mcp_tool_execution.execute_streaming_tool_and_proxy_events(
            ctx["streaming_calls"][0], "conv-1", "model"
        )
    )


@then("the deep research events are forwarded")
def then_forwarded(ctx):
    assert ctx["events"][1:] == ctx["forwarded"]


@given("a streaming tool call that returns a result")
def given_streaming_returns_result(ctx, monkeypatch):
    reset_global_registry()
    ctx["streaming_calls"] = [
        ToolCall(index=0, id="call_1", function_name="search", arguments="{}")
    ]
    tool_message = ToolMessage(role="tool", tool_call_id="call_1", content="result")
    _patch_proxy(monkeypatch, ctx, [("data: start\n\n", None), (None, tool_message)])
    ctx["assistant_calls"] = [
        OpenAIToolCall(
            id="call_1",
            type="function",
            function=ToolCallFunction(name="search", arguments="{}"),
        )
    ]


def _patch_proxy(monkeypatch, ctx, events):
    async def fake_proxy(tool_call, conversation_log_id, model_name):
        for item in events:
            yield item

    monkeypatch.setattr(
        mcp_tool_execution, "execute_streaming_tool_and_proxy_events", fake_proxy
    )
    ctx["proxy_events"] = events


def _default_backend_request(ctx):
    if "request" in ctx:
        return
    ctx.setdefault("assistant_calls", [])
    ctx["backend"] = MagicMock()
    ctx["backend"].build_params.return_value = {}
    ctx["backend"].converse = AsyncMock(return_value=MagicMock())

    class _FakeRequest:
        model = "m1"
        stream = False
        selected_language = "en"
        messages = [UserMessage(role="user", content="hello")]

        def model_copy(self, update=None):
            data = {
                "model": self.model,
                "stream": self.stream,
                "selected_language": self.selected_language,
                "messages": self.messages,
            }
            if update:
                data.update(update)
            return SimpleNamespace(**data)

    ctx["request"] = _FakeRequest()
    ctx["prompts_obj"] = None
    ctx["model_config"] = MagicMock()


@given("a backend and request ready for follow-up")
def given_backend_and_request(ctx):
    _default_backend_request(ctx)


@when("the streaming tool calls are handled")
def when_handle_streaming_calls(ctx, monkeypatch):
    _default_backend_request(ctx)
    # Patch lazy process_streaming_response import target.
    from aichat.serve import open_ai_api

    async def fake_process(
        response,
        request,
        tools,
        mcp_executor,
        backend,
        model_config,
        prompts_obj,
        metrics_state=None,
    ):
        yield "data: follow\n\n"

    monkeypatch.setattr(
        open_ai_api, "process_streaming_response", fake_process, raising=False
    )
    ctx["chunks"] = _drain(
        mcp_tool_execution.handle_streaming_tool_calls(
            ctx["streaming_calls"],
            "conv-1",
            "actual-model",
            ctx.get("mcp_calls", []),
            [],
            ctx.get("assistant_calls", []),
            ctx["request"],
            ctx["backend"],
            ctx["model_config"],
            ctx["prompts_obj"],
            [],
        )
    )


@then("a follow-up converse call happens")
def then_followup_converse(ctx):
    assert ctx["backend"].converse.await_count == 1
    new_messages = ctx["backend"].converse.await_args.args[0]
    roles = [m["role"] for m in new_messages]
    assert "assistant" in roles and "tool" in roles


@then("the follow-up response is streamed")
def then_followup_streamed(ctx):
    assert any("data: follow" in c for c in ctx["chunks"])


@given("only deep research streaming calls")
def given_only_deep_research(ctx, monkeypatch):
    reset_global_registry()
    ctx["streaming_calls"] = [
        ToolCall(index=0, id="call_1", function_name="deep_research", arguments="{}")
    ]
    ctx["backend"] = MagicMock()
    ctx["backend"].converse = AsyncMock()

    class _FakeRequest:
        model = "m1"
        stream = False
        selected_language = "en"
        messages = [UserMessage(role="user", content="hello")]

        def model_copy(self, update=None):
            return SimpleNamespace(messages=self.messages)

    ctx["request"] = _FakeRequest()
    ctx["model_config"] = MagicMock()
    ctx["prompts_obj"] = None
    _patch_proxy(monkeypatch, ctx, [("data: dr\n\n", None)])


@given("an outstanding MCP tool call")
def given_outstanding_mcp(ctx):
    ctx["mcp_calls"] = [SimpleNamespace(id="call_2", function_name="search")]


@then("no follow-up converse call happens")
def then_no_followup(ctx):
    assert ctx["backend"].converse.await_count == 0
