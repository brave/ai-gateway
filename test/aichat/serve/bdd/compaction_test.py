# BDD: conversation compaction (aichat/serve/services/compaction.py).
# Existing unit tests (test/aichat/serve/services/test_compaction_cov_test.py) cover
# check_compaction_needed, can_compact, calculate_max_input_tokens,
# select_messages_for_compaction basics, _call_compaction_model RuntimeError variants.
# Scenarios below target previously uncovered paths (150-152, 269, 347-348, 386/393,
# 457, 496-497, 505-506, 515, 627, 632, 687-702, 764, 791-792). No exact duplicates.

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pytest_bdd import given, scenarios, then, when

from aichat.serve.services import compaction as compaction_module
from aichat.serve.services.compaction import (
    _extract_summary_text,
    _resummarize_compacted_messages,
    create_summary_message,
    generate_compaction_summary,
    maybe_compact_conversation,
    select_messages_for_compaction,
)
from aichat.serve.services.compaction_settings import compaction_settings
from test.aichat.serve.bdd.helpers import (
    dataclass_model_config as _model_config,
)
from test.aichat.serve.bdd.helpers import (
    new_patch as _new_patch,
)

FEATURE = "features/compaction.feature"
scenarios(FEATURE)

SUMMARY_TEXT = "Older conversation summarized here"


class _OkBackend:
    def __init__(self, content=SUMMARY_TEXT):
        self.content = content

    def build_params(self, **kwargs):
        return {}

    async def converse(self, messages, stream=False, params=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class _FailingBackend:
    def build_params(self, **kwargs):
        return {}

    async def converse(self, messages, stream=False, params=None):
        return {"type": "error", "code": 50001, "content": "jaguar down"}


def _conversation(with_tool_call=True):
    """6 old turns + tool result + 4 preserved turns; the assistant carrying
    tool_calls sits inside the compacted (old) segment, not the preserved tail."""
    messages = [{"role": "system", "content": "You are helpful."}]
    for i in range(3):
        messages.append({"role": "user", "content": f"question {i}"})
        messages.append({"role": "assistant", "content": f"answer {i}"})
    if with_tool_call:
        messages.append(
            {
                "role": "assistant",
                "content": "using tools",
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": "call_old", "content": "result"}
        )
    messages.append(
        {"role": "user", "content": [{"type": "text", "text": "preserved question"}]}
    )
    messages.append({"role": "assistant", "content": "preserved answer 1"})
    messages.append({"role": "user", "content": "preserved question 2"})
    messages.append({"role": "assistant", "content": "preserved answer 2"})
    return messages


@pytest.fixture
def ctx():
    return {"undo": [], "messages": [], "result": None, "meta": None}


@pytest.fixture(autouse=True)
def _compaction_cleanup(ctx):
    yield
    # Undo in LIFO order: a scenario can stack several patches on the same
    # attribute (e.g. two givens both patch get_backend); undoing in
    # insertion order would restore an intermediate patched value.
    for mp in reversed(ctx.get("undo", [])):
        mp.undo()


@given("the compaction harness")
def given_harness(ctx):
    pass


@given("a conversation with more turns than preserved and a tool call")
def given_conv_toolcall(ctx):
    ctx["messages"] = _conversation()
    mp = _new_patch(ctx)
    mp.setattr(compaction_module, "get_backend", lambda model: _OkBackend())


@given("a conversation with more turns than preserved turns")
def given_conv_plain(ctx):
    ctx["messages"] = _conversation(with_tool_call=False)
    mp = _new_patch(ctx)
    mp.setattr(compaction_module, "get_backend", lambda model: _OkBackend())
    # Deterministic token totals for threshold/chunking decisions.
    mp.setattr(
        compaction_module, "calculate_message_tokens", MagicMock(return_value=5000)
    )


@given("a short conversation with only preserved turns")
def given_short_conv(ctx):
    ctx["messages"] = [
        {"role": "user", "content": "only question"},
        {"role": "assistant", "content": "only answer"},
    ]
    _new_patch(ctx).setattr(
        compaction_module, "get_backend", lambda model: _OkBackend()
    )


@given("a tiny chunk token limit")
def given_tiny_chunks(ctx):
    mp = _new_patch(ctx)
    mp.setattr(compaction_settings, "compaction_chunk_max_tokens", 40)


@given("a jaguar backend that always fails")
def given_failing_backend(ctx):
    mp = _new_patch(ctx)
    # Pin both models so the fallback!=primary assertion does not depend on
    # ambient settings env overrides in CI.
    mp.setattr(compaction_settings, "compaction_model", "jaguar-primary")
    mp.setattr(compaction_settings, "compaction_fallback_model", "claude-3-haiku")

    def _get_backend(model):
        # Record every model requested so the fallback converse is observable.
        ctx.setdefault("backend_models", []).append(model)
        return (
            _FailingBackend()
            if model == compaction_settings.compaction_model
            else _OkBackend()
        )

    mp.setattr(compaction_module, "get_backend", _get_backend)


@given("compacted messages containing an old summary")
def given_compacted_with_summary(ctx):
    mp = _new_patch(ctx)
    mp.setattr(
        compaction_module,
        "get_backend",
        lambda model: _OkBackend(content="Fresh shorter summary content"),
    )
    ctx["compacted"] = [
        {"role": "system", "content": "You are helpful."},
        create_summary_message("Older summary content"),
        {"role": "user", "content": "recent question"},
    ]


@given("a summary message without a closing context tag")
def given_open_context_msg(ctx):
    ctx["summary_msg"] = {"role": "system", "content": "<context>\nno closing tag here"}


@when("the conversation is compacted with skip checks")
def when_compact_skip(ctx, caplog):
    with caplog.at_level(logging.INFO, logger="aichat.serve.services.compaction"):
        ctx["result"], ctx["meta"] = asyncio.run(
            maybe_compact_conversation(
                messages=ctx["messages"],
                model="m1",
                model_config=_model_config(
                    max_tokens=2000, conversation_token_limit=8000
                ),
                is_premium=False,
                skip_checks=True,
            )
        )


@when("the conversation is compacted with checks")
def when_compact_checked(ctx, caplog):
    # Pin the trigger threshold: ambient env overrides for the settings
    # singleton must not decide whether this scenario compacts.
    mp = _new_patch(ctx)
    mp.setattr(compaction_settings, "compaction_threshold", 0.60)

    with caplog.at_level(logging.INFO, logger="aichat.serve.services.compaction"):
        ctx["result"], ctx["meta"] = asyncio.run(
            maybe_compact_conversation(
                messages=ctx["messages"],
                model="m1",
                model_config=_model_config(
                    max_tokens=2000, conversation_token_limit=8000
                ),
                is_premium=False,
                skip_checks=False,
            )
        )


@when("a single chunk summary is generated")
def when_single_chunk_summary(ctx):
    ctx["summary"] = asyncio.run(generate_compaction_summary(ctx["messages"][:4]))


@when("the compacted messages are re-summarized")
def when_resummarize(ctx):
    ctx["result"], ctx["new_summary"] = asyncio.run(
        _resummarize_compacted_messages(ctx["compacted"], "Older summary content")
    )


@when("the summary text is extracted")
def when_extract_summary(ctx):
    ctx["extracted"] = _extract_summary_text(ctx["summary_msg"])


@when("messages are selected for compaction")
def when_select(ctx):
    ctx["selected"] = select_messages_for_compaction(ctx["messages"])


@then("a compaction summary is produced")
def then_summary(ctx):
    assert ctx["meta"] is not None
    assert SUMMARY_TEXT in ctx["meta"].summary
    assert any("<context>" in m["content"] for m in ctx["result"])


@then("the summary message follows the preserved system prompt")
def then_summary_first(ctx):
    # The original system prompt is preserved; the summary follows it directly.
    first = ctx["result"][1]
    assert first["role"] == "system"
    assert first["content"].startswith(
        "The text in <context> tags is what you already know from earlier"
    )
    assert SUMMARY_TEXT in first["content"]


@then("compaction metadata reports the indices")
def then_meta_indices(ctx):
    assert len(ctx["meta"].compacted_message_indices) > 0
    assert ctx["meta"].tokens_before > 0


@then("the compaction trigger was logged")
def then_trigger_logged(ctx, caplog):
    assert any("Compaction triggered" in r.getMessage() for r in caplog.records)


@then("the original messages are returned with no metadata")
def then_no_compaction(ctx):
    assert ctx["result"] == ctx["messages"]
    assert ctx["meta"] is None


@then("a multi-chunk split is logged")
def then_multi_chunk_logged(ctx, caplog):
    assert any("Splitting compaction input" in r.getMessage() for r in caplog.records)


@then("section labels cover start and end")
def then_section_labels(ctx):
    contents = [m["content"] for m in ctx["result"] if m.get("role") == "system"]
    notes = "\n".join(contents)
    # _section_label pins "start" for the first chunk and "end" for the last;
    # a multi-chunk compaction must produce both notes.
    assert "This is a compaction of the start of the conversation." in notes
    assert "This is a compaction of the end of the conversation." in notes


@then("a compaction summary is produced via the fallback model")
def then_fallback_summary(ctx):
    fallback = compaction_settings.compaction_fallback_model
    requested = ctx["backend_models"]
    primary = compaction_settings.compaction_model
    assert primary in requested
    assert fallback in requested
    # A fallback that equals the primary would make this scenario vacuous.
    assert fallback != primary
    assert ctx["meta"] is not None
    assert SUMMARY_TEXT in ctx["meta"].summary


@then("the summary text is returned")
def then_single_summary(ctx):
    assert ctx["summary"] == SUMMARY_TEXT


@then("the rebuilt list has a single fresh summary message")
def then_rebuilt(ctx):
    messages, summary = ctx["result"], ctx["new_summary"]
    summary_msgs = [m for m in messages if compaction_module._is_summary_message(m)]
    assert len(summary_msgs) == 1
    assert summary == "Fresh shorter summary content"
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == "recent question"


@then("the extracted text equals the raw content")
def then_extracted_raw(ctx):
    assert ctx["extracted"] == "<context>\nno closing tag here"


@then("nothing is selected for compaction")
def then_nothing_selected(ctx):
    to_compact, to_preserve, indices = ctx["selected"]
    assert to_compact == []
    assert indices == []
    assert len(to_preserve) == len(ctx["messages"])
