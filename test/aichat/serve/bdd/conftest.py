"""Shared pytest-bdd step definitions used by multiple bindings."""

from prometheus_client import REGISTRY
from pytest_bdd import parsers, then, when

from test.aichat.serve.bdd.helpers import run_stream_pipeline


@then(parsers.parse('the empty response reason is "{reason}"'))
def empty_reason_then(ctx, reason):
    """Shared step: empty-stream metric incremented for the given reason.

    Both chat_completions.feature and chat_helpers.feature use this step;
    the metric capture happens in the When steps of each binding file
    (ctx["empty_metric_before"]). A missing capture fails loudly here
    instead of silently comparing against 0.0.
    """
    before = ctx["empty_metric_before"]
    value = REGISTRY.get_sample_value(
        "empty_streaming_response_total",
        # The pipeline under test here always requests model "test-model".
        {"model": "test-model", "reason": reason},
    )
    assert value is not None
    assert value == before + 1


@when("the stream pipeline processes no chunks at all")
def when_empty_stream_shared(ctx, monkeypatch):
    ctx["empty_metric_before"] = (
        REGISTRY.get_sample_value(
            "empty_streaming_response_total",
            {"model": "test-model", "reason": "no_chunks"},
        )
        or 0.0
    )
    ctx["events"] = run_stream_pipeline(ctx, monkeypatch, chunks=[])


@then("the stream still ends with a DONE sentinel")
def then_stream_done_shared(ctx):
    assert ctx["events"][-1].strip() == "data: [DONE]"


@then("the follow-up conversation is executed with the tool result")
def then_followup_shared(ctx):
    backend = ctx["backend"]
    backend.converse.assert_awaited_once()
    llm_messages = backend.converse.await_args.args[0]
    tool_messages = [m for m in llm_messages if m.get("role") == "tool"]
    assert tool_messages, llm_messages
    assert tool_messages[0]["content"] == ctx["tool_result"]
