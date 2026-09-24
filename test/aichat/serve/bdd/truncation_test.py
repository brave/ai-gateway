import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import truncation_continuation as tc
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.server_settings import server_settings
from test.aichat.serve.bdd.message_preprocessing_test import _model_config

FEATURE = Path(__file__).parent / "features" / "truncation.feature"
scenarios(str(FEATURE))


@pytest.fixture
def ctx():
    return {}


# --- pure text helpers -----------------------------------------------------------


@given(parsers.parse('the text "{text}"'))
def given_text(text, ctx):
    ctx["text"] = "" if text == "<empty>" else text


@when("the longest character run is measured")
def when_longest_run(ctx):
    ctx["run"] = tc.longest_char_run(ctx["text"])


@then(parsers.parse("the longest run length is {run:d}"))
def then_longest_run(run, ctx):
    assert ctx["run"] == run


@when(parsers.parse("degenerate repetition is checked with min run {min_run:d}"))
def when_degenerate(min_run, ctx):
    ctx["verdict"] = tc.has_degenerate_repetition(ctx["text"], min_run=min_run)


@then(parsers.parse("the repetition verdict is {verdict}"))
def then_repetition(verdict, ctx):
    assert ctx["verdict"] == (verdict == "True")


@given(parsers.parse('a finish reason "{finish_reason}"'))
def given_finish_reason(finish_reason, ctx):
    ctx["finish_reason"] = finish_reason


@when("the truncated reply check runs")
def when_truncated_reply(ctx):
    ctx["verdict"] = tc.should_continue_truncated_reply(
        finish_reason=ctx["finish_reason"], text=ctx["text"]
    )


@then(parsers.parse("the accidental stop verdict is {verdict}"))
def then_accidental(verdict, ctx):
    assert ctx["verdict"] == (verdict == "True")


@when("the truncation action is resolved")
def when_resolve_action(ctx):
    ctx["action"] = tc.resolve_truncation_action(
        finish_reason=ctx["finish_reason"], text=ctx["text"]
    )


@then(parsers.parse("the action is {action}"))
def then_action(action, ctx):
    assert ctx["action"] == action


# --- config gate ------------------------------------------------------------------


@given(parsers.parse("a model config with truncation_continuation {flag}"))
def given_model_config_flag(flag, ctx):
    ctx["model_config"] = _model_config(
        truncation_continuation=flag == "True", tool_support=True
    )


@when("the truncation fixup gate is evaluated")
def when_fixup_gate(ctx):
    ctx["enabled"] = tc.truncation_fixup_enabled(ctx["model_config"])


@then(parsers.parse("the fixup is enabled {enabled}"))
def then_fixup(enabled, ctx):
    assert ctx["enabled"] == (enabled == "True")


# --- continuation message building -------------------------------------------------


@given(parsers.parse('conversation messages "{role} {content}"'))
def given_conversation(role, content, ctx):
    ctx["conversation"] = [{"role": role, "content": content}]


@given(parsers.parse('a partial assistant reply "{partial}"'))
def given_partial(partial, ctx):
    ctx["partial"] = partial


@when("the continuation messages are built")
def when_build_continuation(ctx):
    ctx["built"] = tc.build_truncation_continuation_messages(
        ctx["conversation"], ctx["partial"]
    )


@then("the last message asks to continue exactly where it stopped")
def then_last_message(ctx):
    assert ctx["built"][-1]["role"] == "user"
    assert "Continue exactly where you stopped" in ctx["built"][-1]["content"]


@then("the assistant partial is preserved in the second-to-last message")
def then_partial_preserved(ctx):
    assert ctx["built"][-2] == {"role": "assistant", "content": ctx["partial"]}


# --- response helpers ---------------------------------------------------------------


@given(parsers.parse("a completion response {description}"))
def given_completion_response(description, ctx):
    if description == "an object response":
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hi", tool_calls=None),
                    finish_reason="tool_calls",
                )
            ]
        )
    elif description in ("an error dict",):
        response = {"type": "error", "content": "nope"}
    elif description in (
        "a response carrying tool calls",
        "carrying tool calls",
        "a dict with tools",
    ):
        response = {
            "choices": [
                {
                    "message": {"content": "hi", "tool_calls": [{"id": 1}]},
                    "finish_reason": "tool_calls",
                }
            ]
        }
    elif description in ("a response without content", "without content"):
        response = {
            "choices": [{"message": {"content": None}, "finish_reason": "stop"}]
        }
    elif description in (
        "a complete non-truncated response",
        "complete non-truncated response",
    ):
        response = {
            "choices": [
                {
                    "message": {"content": "A complete sentence."},
                    "finish_reason": "stop",
                }
            ]
        }
    elif description in (
        "a response that degenerately repeats",
        "that degenerately repeats",
    ):
        response = {
            "choices": [{"message": {"content": "a" * 15}, "finish_reason": "stop"}]
        }
    elif description in (
        "a response that is cut off mid-sentence",
        "that is cut off mid-sentence",
    ):
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Let me think about this some more and then"
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    else:  # a plain dict
        response = {
            "choices": [{"message": {"content": "partial"}, "finish_reason": "stop"}]
        }
    ctx["response"] = response


@when("the response helpers are exercised")
def when_response_helpers(ctx):
    ctx["tool_calls"] = tc._response_has_tool_calls(ctx["response"])
    ctx["content"], ctx["finish_reason"] = tc._completion_content_and_finish_reason(
        ctx["response"]
    )


@then(
    parsers.parse(
        'the tool call detection is {tool_calls} and content is "{content}" and finish reason is "{finish_reason}"'
    )
)
def then_helpers(tool_calls, content, finish_reason, ctx):
    assert ctx["tool_calls"] == (tool_calls == "True")
    assert ctx["content"] == content
    assert ctx["finish_reason"] == finish_reason


@then(parsers.parse('the response content can be overwritten with "{new_content}"'))
def then_overwrite(new_content, ctx):
    tc._set_completion_content(ctx["response"], new_content)
    assert tc._completion_content_and_finish_reason(ctx["response"])[0] == new_content


# --- maybe_non_stream_truncation_recovery ----------------------------------------------


@given(parsers.parse("truncation continuation is {flag}"))
def given_truncation_flag(flag, ctx):
    enabled = flag in ("True", "enabled")
    cfg = _model_config(truncation_continuation=enabled, tool_support=True)
    ctx["model_config"] = cfg
    ctx["plan_model_config"] = cfg


@given("the truncation depth budget is exhausted")
def given_depth_exhausted(monkeypatch):
    monkeypatch.setattr(server_settings, "truncation_continuation_max_depth", 0)


@given(parsers.parse('a fake recovery backend retrying "{content}"'))
def given_retry_backend(content, ctx):
    backend = MagicMock()
    backend.converse = AsyncMock(
        return_value={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}]
        }
    )
    ctx["backend"] = backend


@given(parsers.parse('a fake recovery backend continuing "{continuation}"'))
def given_continue_backend(continuation, ctx):
    backend = MagicMock()
    backend.converse = AsyncMock(
        return_value={
            "choices": [{"message": {"content": continuation}, "finish_reason": "stop"}]
        }
    )
    ctx["backend"] = backend


@given("a fake recovery backend failing its follow-up call")
def given_failing_backend(ctx):
    backend = MagicMock()
    backend.converse = AsyncMock(return_value={"type": "error", "code": 50000})
    ctx["backend"] = backend


@when("non-streaming truncation recovery runs")
def when_non_stream_recovery(ctx):
    request = SimpleNamespace(
        model=ctx.get("model", "test-model"),
        messages=[{"role": "user", "content": "hi"}],
    )
    ctx["result"] = asyncio.run(
        tc.maybe_non_stream_truncation_recovery(
            response=ctx["response"],
            request=request,
            llm_messages=[{"role": "user", "content": "hi"}],
            tools=[],
            params={},
            backend=ctx.get("backend", MagicMock()),
            model_config=ctx["model_config"],
            depth=0,
        )
    )


@then("the response is returned unchanged")
def then_unchanged(ctx):
    assert ctx["result"] is ctx["response"]


@then("the backend was asked to converse once")
def then_converse_once(ctx):
    ctx["backend"].converse.assert_awaited_once()


@then(parsers.parse('the returned content is "{content}"'))
def then_returned_content(content, ctx):
    assert ctx["result"]["choices"][0]["message"]["content"] == content


@then("the merged content contains the partial text and the continuation")
def then_merged(ctx):
    content = ctx["result"]["choices"][0]["message"]["content"]
    assert "Let me think about this some more and then" in content
    assert "and that is all" in content


# --- plan / record ---------------------------------------------------------------------


@given(
    parsers.parse(
        'assistant text "{text}" with finish reason "{finish_reason}" and tool calls {tool_calls}'
    )
)
def given_stream_state(text, finish_reason, tool_calls, ctx):
    ctx["assistant_text"] = text
    ctx["stream_finish_reason"] = None if finish_reason == "none" else finish_reason
    ctx["stream_tool_calls"] = tool_calls == "True"


@when("the stream truncation recovery is planned")
def when_plan_stream(ctx):
    ctx["planned"] = tc.plan_stream_truncation_recovery(
        assistant_text=ctx["assistant_text"],
        tool_calls_in_response=ctx["stream_tool_calls"],
        last_finish_reason=ctx["stream_finish_reason"],
        model_config=ctx["plan_model_config"],
        backend=MagicMock(),
        metrics_state={},
    )


@then(parsers.parse("the planned action is {action}"))
def then_planned(action, ctx):
    assert ctx["planned"] == action


@given(parsers.parse('the recorded recovery action "{action}" for model "{model}"'))
def given_record(action, model, ctx):

    ctx["record_model"] = model
    ctx["record_action"] = action
    tc.record_stream_truncation_recovery(model=model, action=action, depth=1)


@then("the skipped-degenerate-stream metric was incremented")
def then_skip_metric(ctx):
    assert ctx["record_action"] == "retry"


@then("the continue metric was incremented")
def then_continue_metric(ctx):
    assert ctx["record_action"] == "continue"


# --- stream_truncation_continuation ------------------------------------------------------


@given('a streaming request for model "near-test"')
def given_near_stream_request(ctx):
    from aichat.protocol.open_ai_protocol import Request as OpenAIRequest
    from aichat.protocol.open_ai_protocol import UserMessage

    ctx["stream_request"] = OpenAIRequest(
        model="near-test",
        messages=[UserMessage(content="hi")],
        stream=True,
        tools=None,
    )


@given("a fake continuation backend")
def given_fake_continuation_backend(ctx):
    backend = MagicMock()
    backend.build_params.return_value = {}

    async def stream():
        yield 'data: {"chunk":1}\n\n'

    backend.converse = AsyncMock(return_value=stream())
    ctx["backend"] = backend


@when("the stream continuation runs")
def when_stream_continuation(ctx):
    metrics_seen = []

    async def fake_process(*args, **kwargs):
        metrics_seen.append(kwargs["metrics_state"]["truncation_continuation_depth"])
        async for event in args[0]:
            yield event

    async def run():
        events = []
        async for event in tc.stream_truncation_continuation(
            request=ctx["stream_request"],
            partial_assistant_text="partial text",
            client_tools=[],
            mcp_executor=None,
            backend=ctx["backend"],
            model_config=_model_config(truncation_continuation=True, tool_support=True),
            prompts_obj=None,
            requested_model=None,
            metrics_state={"truncation_continuation_depth": 0},
            process_streaming_response=fake_process,
        ):
            events.append(event)
        return events

    ctx["forwarded"] = asyncio.run(run())
    ctx["metrics_seen"] = metrics_seen


@then("the continuation backend received the NEAR API key")
def then_near_key(ctx):
    args = ctx["backend"].converse.await_args.args
    assert args[2]["api_key"] == external_service_settings.near_api_key


@then("the continuation depth advanced to 1")
def then_depth(ctx):
    assert ctx["metrics_seen"] == [1]


@then("the follow-up events are forwarded")
def then_forwarded(ctx):
    assert ctx["forwarded"] == ['data: {"chunk":1}\n\n']
