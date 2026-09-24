"""BDD: Deep research streaming.

Dup TODOs (same behaviour covered by unit tests):
# TODO: remove test/aichat/serve/services/mcp/test_streaming_executor.py#test_timeout test/aichat/serve/services/mcp/test_streaming_executor.py:147
# TODO: remove test/aichat/serve/services/mcp/test_streaming_executor.py#test_request_error test/aichat/serve/services/mcp/test_streaming_executor.py:173
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

FEATURE = "features/deep_research.feature"
scenarios(FEATURE)

from aichat.serve import mcp_tool_execution
from aichat.serve.services.mcp.handlers.deep_research import (
    service as dr_service,
)
from aichat.serve.services.mcp.handlers.deep_research import (
    streaming_executor,
)
from aichat.serve.services.mcp.handlers.deep_research.handler import (
    DeepResearchServerHandler,
)
from aichat.serve.services.mcp.handlers.deep_research.streaming_executor import (
    DeepResearchStreamingExecutor,
)
from aichat.serve.services.mcp.mcp_settings import mcp_settings
from aichat.serve.services.mcp.registry import (
    get_global_registry,
    reset_global_registry,
)


@pytest.fixture
def ctx():
    return {}


class _FakeStreamCtx:
    def __init__(self, chunks):
        self._response = _FakeResponse(chunks)

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeResponse:
    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    async def aiter_text(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self):
        return b""


@pytest.fixture(autouse=True)
def cleanup_registry():
    yield
    reset_global_registry()


@pytest.fixture(autouse=True)
def dr_enabled(monkeypatch):
    monkeypatch.setattr(mcp_settings, "deep_research_enabled", True)
    yield


def _drain(agen):
    async def run():
        out = []
        async for item in agen:
            out.append(item)
        return out

    return asyncio.run(run())


@given("a deep research harness")
def given_harness(ctx):
    reset_global_registry()
    ctx["events"] = None


@given("deep research is disabled")
def given_disabled(monkeypatch):
    monkeypatch.setattr(mcp_settings, "deep_research_enabled", False)


@when(parsers.parse('the stream is executed for "{query}"'))
def when_execute_stream_query(ctx, query):
    ctx["events"] = _drain(DeepResearchStreamingExecutor().execute_streaming(query))


@then("a disabled error event is yielded")
def then_disabled_error(ctx):
    assert ctx["events"] == [
        {"type": "error", "error": "Deep research is not enabled on this server."}
    ]


@given(parsers.parse("the deep research transport raises {failure}"))
def given_transport_failure(ctx, monkeypatch, failure):
    exc = {
        "timeout": httpx.TimeoutException("t"),
        "connect_error": httpx.ConnectError("c"),
        "generic_request": httpx.RequestError("r"),
    }[failure.replace(" ", "_")]
    monkeypatch.setattr(
        streaming_executor.httpx, "AsyncClient", MagicMock(side_effect=exc)
    )


@when("the stream is executed")
def when_stream_executed(ctx):
    ctx["events"] = _drain(DeepResearchStreamingExecutor().execute_streaming("q"))


@then("an unreachable or timeout error event is yielded")
def then_transport_error(ctx):
    first = ctx["events"][0]
    assert first["type"] == "error"
    assert "timed out" in first["error"] or "unreachable" in first["error"]


@given("a stream with a trailing non json fragment")
def given_trailing_junk(ctx, monkeypatch):
    chunks = ['{"event": "x"}\n', "not-json"]

    class _Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, *args, **kwargs):
            return _FakeStreamCtx(chunks)

    monkeypatch.setattr(streaming_executor.httpx, "AsyncClient", _Client)


@then("no error is raised and events are yielded")
def then_events_yielded(ctx):
    assert ctx["events"] == [{"event": "x"}]


@given("a deep research call without a query")
def given_no_query(ctx):
    ctx["tool_call"] = SimpleNamespace(
        function_name="deep_research", id="call_1", index=0
    )
    ctx["tool_args"] = {}


@when("deep research is executed and streamed")
def when_dr_executed(ctx):
    ctx["events"] = _drain(
        dr_service.execute_deep_research_streaming(
            ctx["tool_call"],
            "call_1",
            ctx["tool_args"],
            "conv-1",
            "model",
            mcp_tool_execution._create_tool_output_chunk_from_parts,
        )
    )


@then("a missing query error event is yielded")
def then_missing_query(ctx):
    event, message = ctx["events"][0]
    assert "Missing required" in event and "query" in event
    assert message is None


def _fake_executor_class(events):
    class _FakeDR:
        def __init__(self, *a, **k):
            pass

        async def execute_streaming(self, query, country="us", language="en"):
            for event in events:
                yield event

    return _FakeDR


@given("a registry containing the deep research handler")
def given_dr_handler(ctx):
    reset_global_registry()
    get_global_registry().register_server(DeepResearchServerHandler())


@given("streamed events ending with a final answer and citations")
def given_final_events_citations(ctx, monkeypatch):
    ctx["executor_events"] = [
        {"event": "queries", "queries": ["q1"]},
        {
            "event": "answer",
            "final": True,
            "answer": "The answer [1].",
            "citations": [{"number": 1, "url": "https://example.com/a", "title": "A"}],
        },
    ]
    monkeypatch.setattr(
        dr_service,
        "DeepResearchStreamingExecutor",
        _fake_executor_class(ctx["executor_events"]),
    )
    ctx["tool_call"] = SimpleNamespace(
        function_name="deep_research", id="call_1", index=0
    )
    ctx["tool_args"] = {"query": "q"}


@then("a web sources chunk precedes the completion")
def then_web_sources_first(ctx):
    strings = [e for e, _ in ctx["events"] if e]
    ws_idx = next(i for i, s in enumerate(strings) if "brave-chat.webSources" in s)
    ctx["completion"] = strings[ws_idx + 1]
    assert "---" in ctx["completion"]


@then("the completion chunk contains spaced citations")
def then_spaced_citations(ctx):
    assert "The answer [1]." in ctx["completion"]


@then("the tool message holds the formatted answer")
def then_tool_message_answer(ctx):
    _, message = ctx["events"][-1]
    assert message.role == "tool"
    text = message.content[0].text
    assert text.startswith("The answer")


@given("a registry without the deep research handler")
def given_no_dr_handler(ctx):
    reset_global_registry()


@given("streamed events ending with a final answer")
def given_final_events(ctx, monkeypatch):
    ctx["executor_events"] = [
        {
            "event": "answer",
            "final": True,
            "answer": "The answer [1].",
            "citations": [{"number": 1, "url": "https://example.com/a", "title": "A"}],
        },
    ]
    monkeypatch.setattr(
        dr_service,
        "DeepResearchStreamingExecutor",
        _fake_executor_class(ctx["executor_events"]),
    )
    ctx["tool_call"] = SimpleNamespace(
        function_name="deep_research", id="call_1", index=0
    )
    ctx["tool_args"] = {"query": "q"}


@then("the completion chunk is yielded")
def then_completion_yielded(ctx):
    completion = next(e for e, _ in ctx["events"] if e)
    assert "The answer [1]." in completion
    assert "---" in completion


@then("the tool message uses plaintext citations")
def then_plaintext_citations(ctx):
    _, message = ctx["events"][-1]
    assert "Sources:" in message.content
    assert "[1] https://example.com/a" in message.content


@given("streamed events without a final answer")
def given_no_final_events(ctx, monkeypatch):
    ctx["executor_events"] = [{"event": "queries", "queries": []}]
    monkeypatch.setattr(
        dr_service,
        "DeepResearchStreamingExecutor",
        _fake_executor_class(ctx["executor_events"]),
    )
    ctx["tool_call"] = SimpleNamespace(
        function_name="deep_research", id="call_1", index=0
    )
    ctx["tool_args"] = {"query": "q"}


@then("a deep research placeholder output chunk is yielded")
def then_placeholder_chunk(ctx):
    strings = [e for e, _ in ctx["events"] if e]
    assert any(
        "Deep research did not produce a result." in s and "output_content" in s
        for s in strings
    )


@then("the tool message says no final answer")
def then_no_final_answer_message(ctx):
    _, message = ctx["events"][-1]
    assert (
        message.content == "Deep research completed but no final answer was produced."
    )


@given("a formatted deep research result with no sources")
def given_empty_result(ctx):
    ctx["formatted_result"] = {"sources": [], "content": ""}
    ctx["tool_call"] = SimpleNamespace(
        function_name="deep_research", id="call_1", index=0
    )


@when("the deep research tool message content is built")
def when_build_content(ctx):
    ctx["built_content"] = DeepResearchServerHandler().get_tool_message_content(
        ctx["formatted_result"], ctx["tool_call"]
    )


@then('the deep research content is "Deep research completed"')
def then_deep_research_completed(ctx):
    assert ctx["built_content"] == "Deep research completed"
