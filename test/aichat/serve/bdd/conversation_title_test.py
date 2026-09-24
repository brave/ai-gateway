# BDD: conversation title completion (aichat/serve/services/conversation_title.py).
# TODO: remove test_complete_conversation_title_chat_returns_error_when_no_title_text \
#   test/aichat/serve/services/conversation_title_test.py:47

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.prompts.conversation_title import TYPE as TITLE_TYPE
from aichat.protocol.open_ai_protocol import (
    ConversationTitleContentPart,
    UserMessage,
)
from aichat.protocol.open_ai_protocol import (
    Request as OpenAIRequest,
)
from aichat.serve.services import conversation_title as ct
from test.aichat.serve.bdd.message_preprocessing_test import _model_config

FEATURE = "features/conversation_title.feature"
scenarios(FEATURE)


class _FakeBackend:
    def __init__(self, reply):
        self.reply = reply

    def build_params(self, *args, **kwargs):
        return {}

    async def converse(self, messages, stream=False, params=None):
        return self.reply


def _completion(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _metric_value() -> float:
    return REGISTRY.get_sample_value("conversation_title_invalid_request_total") or 0.0


@pytest.fixture
def ctx():
    return {}


@given("the conversation title harness")
def given_harness(ctx):
    ctx["request"] = None
    ctx["result"] = None
    ctx["http_error"] = None
    ctx["metric_before"] = _metric_value()


@given("a title request with an empty title part")
def given_empty_title(ctx):
    ctx["request"] = OpenAIRequest(
        model="m1",
        messages=[
            UserMessage(
                content=[ConversationTitleContentPart(type=TITLE_TYPE, text="")]
            )
        ],
        stream=True,
    )


@given("a title request with a filled title part")
def given_filled_title(ctx):
    ctx["request"] = OpenAIRequest(
        model="m1",
        messages=[
            UserMessage(
                content=[
                    ConversationTitleContentPart(type=TITLE_TYPE, text="The chat topic")
                ]
            )
        ],
        stream=False,
    )


@given("the title backend returns a completion")
def given_backend_completion(ctx, monkeypatch):
    ctx["backend"] = _FakeBackend(_completion("Some title reply"))
    monkeypatch.setattr(ct, "get_backend", lambda model: ctx["backend"])
    monkeypatch.setattr(ct, "get_model_config", MagicMock(return_value=_model_config()))


@given(parsers.parse("the title backend returns an error dict with code {code:d}"))
def given_backend_error(ctx, monkeypatch, code):
    ctx["backend"] = _FakeBackend(
        {"type": "error", "code": code, "content": "upstream exploded"}
    )
    monkeypatch.setattr(ct, "get_backend", lambda model: ctx["backend"])
    monkeypatch.setattr(ct, "get_model_config", MagicMock(return_value=_model_config()))


@when("the conversation title completion runs")
def when_run(ctx, monkeypatch):
    try:
        ctx["result"] = asyncio.run(
            ct.complete_conversation_title_chat(
                request=ctx["request"],
                prompts=MagicMock(),
                process_streaming_response=AsyncMock(),
            )
        )
    except HTTPException as e:
        ctx["http_error"] = e


@when("the conversation title completion runs without streaming")
def when_run_nonstream(ctx, monkeypatch):
    ctx["request"].stream = False
    when_run(ctx, monkeypatch)


@when("the conversation title completion runs with streaming")
def when_run_stream(ctx, monkeypatch):
    ctx["request"].stream = True
    ctx["chunks"] = []

    def fake_stream(*args, **kwargs):
        async def gen():
            yield 'data: {"title": true}\n\n'

        return gen()

    try:
        ctx["result"] = asyncio.run(
            ct.complete_conversation_title_chat(
                request=ctx["request"],
                prompts=MagicMock(),
                process_streaming_response=fake_stream,
            )
        )
    except HTTPException as e:
        ctx["http_error"] = e


@then("the response is a 400 error response")
def then_400(ctx):
    assert ctx["result"].status_code == 400


@then("the invalid title request metric increased")
def then_metric(ctx):
    assert _metric_value() >= ctx["metric_before"] + 1


@then("the returned response carries the backend content")
def then_nonstream(ctx):
    assert ctx["result"].choices[0].message.content == "Some title reply"


@then(parsers.parse("an HTTP error with status {status:d} is raised"))
def then_http_error(ctx, status):
    assert ctx["http_error"] is not None
    assert ctx["http_error"].status_code == status


@then("the streamed title chunks carry the backend content")
def then_streamed(ctx):
    from fastapi.responses import StreamingResponse

    assert isinstance(ctx["result"], StreamingResponse)

    async def drain():
        return "".join([chunk async for chunk in ctx["result"].body_iterator])

    body = asyncio.run(drain())
    assert '"title": true' in body


# --- title part detection ---


@given(parsers.parse("a message list where {layout}"))
def given_layout(ctx, layout):
    title_obj = ConversationTitleContentPart(type=TITLE_TYPE, text="hi")
    if layout == "user_with_title":
        ctx["messages"] = [UserMessage(content=[title_obj])]
    elif layout == "assistant_last":
        from aichat.protocol.open_ai_protocol import AssistantMessage

        ctx["messages"] = [UserMessage(content="hi"), AssistantMessage(content="yo")]
    elif layout == "string_content":
        ctx["messages"] = [UserMessage(content="plain string")]
    elif layout == "dict_title_part":
        ctx["messages"] = [
            SimpleNamespace(role="user", content=[{"type": TITLE_TYPE, "text": "hi"}])
        ]
    elif layout == "assistant_then_user":
        ctx["messages"] = [
            SimpleNamespace(role="assistant", content=[{"type": "other", "text": "x"}]),
            UserMessage(content=[title_obj]),
        ]
    else:
        raise ValueError(f"unknown layout {layout}")


@when("the last message is checked for a title part")
def when_check_presence(ctx):
    ctx["present"] = ct.last_message_includes_conversation_title(ctx["messages"])


@then(parsers.parse("the title presence is {present}"))
def then_presence(ctx, present):
    assert ctx["present"] == (present == "true")


@when("the title part text is inspected")
def when_inspect_text(ctx):
    ctx["no_text"] = ct._title_part_has_no_text(ctx["messages"])


@then("the title part has usable text")
def then_usable(ctx):
    assert ctx["no_text"] is False
