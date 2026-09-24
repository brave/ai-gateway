# BDD: token trimming edge paths (aichat/serve/services/trimming.py).
# Existing unit tests (test/aichat/serve/services/test_trimming.py) cover
# is_trimmable_content_part, trim_tool_messages, trim_messages_to_fit, maybe_trim_messages
# happy paths; the scenarios below target previously uncovered branches
# (113, 130-132, 161, 165-192, 220-225, 284, 292-301, 321-323). No exact-duplicate
# scenarios -> no TODO links required.

import logging
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.prompts.page_text import TYPE as PAGE_TEXT_TYPE
from aichat.prompts.search_results import TYPE as SEARCH_RESULTS_TYPE
from aichat.serve.services import trimming
from aichat.serve.services.compaction_settings import compaction_settings

FEATURE = "features/trimming.feature"
scenarios(FEATURE)

TRIM_NOTE = "[... content truncated to fit conversation limit]"


@pytest.fixture
def ctx():
    return {"undo": [], "messages": [], "result": None}


@pytest.fixture(autouse=True)
def _trim_cleanup(ctx):
    yield
    for mp in reversed(ctx.get("undo", [])):
        mp.undo()


def _new_patch(ctx):
    mp = pytest.MonkeyPatch()
    ctx["undo"].append(mp)
    return mp


@given("the trimming harness")
def given_harness(ctx):
    pass


@given(
    parsers.parse(
        "messages whose real token count is {real:d} but the budget is {budget:d}"
    )
)
def given_token_counts(ctx, real, budget):
    mp = _new_patch(ctx)
    ctx["budget"] = budget
    mp.setattr(trimming, "calculate_message_tokens", MagicMock(return_value=real))


@given(parsers.parse("a trimmable part of type {part_type} carrying the field {field}"))
def given_trimmable_part(ctx, part_type, field):
    ptype = PAGE_TEXT_TYPE if part_type == "brave-page-text" else SEARCH_RESULTS_TYPE
    part = {"type": ptype}
    if field == "content":
        part["content"] = "word " * 800
    elif field == "text":
        part["text"] = "word " * 800
    elif field == "empty":
        part["content"] = ""
    ctx["part"] = part
    ctx["messages"] = [{"role": "user", "content": [part]}]


@given("a tool message whose content is a list")
def given_tool_list_content(ctx):
    ctx["messages"] = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": "structured result"}],
        },
        {"role": "assistant", "content": "ok"},
    ]


@given("a tool message with string content")
def given_tool_str_content(ctx):
    ctx["messages"] = [
        {"role": "tool", "tool_call_id": "call_1", "content": "tool result " * 400},
        {"role": "assistant", "content": "ok"},
    ]


@given("text truncation always fails")
def given_truncate_fails(ctx):
    mp = _new_patch(ctx)
    mp.setattr(
        trimming,
        "truncate_text",
        MagicMock(side_effect=RuntimeError("tokenizer exploded")),
    )


@given("two trimmable parts where the first consumes the whole excess")
def given_two_parts(ctx):
    first = {"type": PAGE_TEXT_TYPE, "content": "word " * 5000}
    second = {"type": PAGE_TEXT_TYPE, "content": "word " * 400}
    ctx["messages"] = [{"role": "user", "content": [first, second]}]


@given("a trimmable part of type brave-page-text carrying neither field")
def given_no_field_part(ctx):
    ctx["messages"] = [{"role": "user", "content": [{"type": PAGE_TEXT_TYPE}]}]


@given("a trimmable part of type brave-page-text carrying an empty content field")
def given_empty_field_part(ctx):
    ctx["messages"] = [
        {"role": "user", "content": [{"type": PAGE_TEXT_TYPE, "content": ""}]}
    ]


@given(parsers.parse("a plain text part limited to {limit:d} tokens"))
def given_plain_text_cap(ctx, limit):
    mp = _new_patch(ctx)
    mp.setattr(compaction_settings, "max_text_part_tokens", limit)
    ctx["messages"] = [
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": [{"type": "text", "text": "long " * 500}]},
    ]


@when("the messages are trimmed to fit")
def when_trim_to_fit(ctx):
    ctx["result"] = trimming.trim_messages_to_fit(ctx["messages"], ctx["budget"])


@when("the tool messages are trimmed directly")
def when_trim_tool_direct(ctx):
    ctx["result"] = trimming.trim_tool_messages(ctx["messages"], ctx["budget"])


@when("maybe trim runs for a model")
def when_maybe_trim(ctx):
    mp = _new_patch(ctx)
    mp.setattr(
        trimming,
        "calculate_max_input_tokens",
        MagicMock(return_value=(ctx["budget"], 8000)),
    )
    ctx["result"] = trimming.maybe_trim_messages(
        ctx["messages"], MagicMock(), False, model="trim-model"
    )


@then("the part content is truncated with a trimming note")
def then_part_truncated(ctx):
    trimmed_messages, _, _ = ctx["result"]
    part = trimmed_messages[0]["content"][0]
    body = part.get("content") or part.get("text") or ""
    assert TRIM_NOTE in body


@then(parsers.parse("{trimmed} tokens were removed"))
def then_removed(ctx, trimmed):
    result = ctx["result"]
    trimmed_tokens = result[2] if len(result) == 3 else result[1]
    if trimmed == ">0":
        assert trimmed_tokens > 0
    elif trimmed == "no":
        assert trimmed_tokens == 0
    else:
        assert trimmed_tokens == int(trimmed)


@then("the tool message content is unchanged")
def then_tool_unchanged(ctx):
    trimmed_messages, _, trimmed_tokens = ctx["result"]
    assert trimmed_tokens == 0
    assert isinstance(trimmed_messages[0]["content"], list)


@then("no tokens were removed")
def then_no_trim(ctx):
    result = ctx["result"]
    trimmed = result[1] if len(result) == 2 else result[2]
    assert trimmed == 0


@then("the second trimmable part is unchanged")
def then_second_unchanged(ctx):
    trimmed_messages, _, _ = ctx["result"]
    parts = trimmed_messages[0]["content"]
    assert parts[1]["content"] == "word " * 400


@then("the part content is unchanged")
def then_part_unchanged(ctx):
    trimmed_messages, _, _ = ctx["result"]
    part = trimmed_messages[0]["content"][0]
    assert TRIM_NOTE not in (part.get("content") or part.get("text") or "")


@then("the plain text part is truncated with a trimming note")
def then_text_part_truncated(ctx):
    trimmed_messages, _, trimmed_tokens = ctx["result"]
    part = trimmed_messages[1]["content"][0]
    assert TRIM_NOTE in part["text"]
    assert trimmed_tokens > 0


@then("a text part failure is logged")
def then_text_failure_logged(ctx, caplog):

    with caplog.at_level(logging.WARNING, logger="aichat.serve.services.trimming"):
        trimming.trim_messages_to_fit(ctx["messages"], ctx["budget"])
    assert any("Failed to trim text part" in r.message for r in caplog.records)


@then("an insufficient trimming warning is logged with a role breakdown")
def then_insufficient_logged(ctx, caplog):

    with caplog.at_level(logging.WARNING, logger="aichat.serve.services.trimming"):
        trimming.trim_messages_to_fit(ctx["messages"], ctx["budget"])
    messages = [r.message for r in caplog.records]
    assert any("Token trimming insufficient" in m for m in messages)
    assert any("Per-role breakdown" in m for m in messages)


@then("trimming metrics are recorded")
def then_metrics(ctx):
    value = REGISTRY.get_sample_value("token_trimming_total", {"model": "trim-model"})
    assert value is not None and value >= 1


@then("trimmed tokens are reported")
def then_reported(ctx):
    _, _, trimmed_tokens = ctx["result"]
    assert trimmed_tokens > 0


@then("the original messages are returned with zero trimmed tokens")
def then_noop(ctx):
    trimmed_messages, total, trimmed_tokens = ctx["result"]
    assert trimmed_messages is ctx["messages"]
    assert trimmed_tokens == 0
    assert total > 0
