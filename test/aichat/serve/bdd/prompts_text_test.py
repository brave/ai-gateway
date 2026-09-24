"""BDD steps for prompt text content augmenters (file extracted text, pdf text,
user memory alignment trace, leo system training cutoff, Prompt ABC).

New coverage — no unit-test duplication found in test/aichat/prompts/
(user_memory unit tests cover augment/_format_memory, not trace text;
file_extracted_text and pdf_text_content have no unit test files).
"""

from datetime import datetime

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.prompts import leo_system as _leo_module
from aichat.prompts.file_extracted_text import file_extracted_text
from aichat.prompts.leo_system import leo_system as leo_system_prompt
from aichat.prompts.pdf_text_content import pdf_text_content
from aichat.prompts.prompt import Prompt
from aichat.prompts.user_memory import UserMemoryContentPart

FEATURE = "features/prompts_text.feature"
scenarios(FEATURE)


@pytest.fixture
def ctx() -> dict:
    return {}


@given("the prompt text harness")
def given_harness(ctx: dict) -> None:
    ctx["messages"] = None
    ctx["result"] = None
    ctx["value"] = None


@given(
    parsers.parse('a user message with a brave-file-extracted-text part "{payload}"')
)
def given_fet_str(ctx: dict, payload: str) -> None:
    ctx["messages"] = [
        {
            "role": "user",
            "content": [{"type": "brave-file-extracted-text", "content": payload}],
        }
    ]


@given(
    parsers.parse(
        'a user message with an extracted-text part holding list content "{first}" and "{second}"'
    )
)
def given_fet_list(ctx: dict) -> None:
    ctx["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "brave-file-extracted-text",
                    "content": [
                        {"type": "text", "text": "Alpha"},
                        {"type": "text", "text": "Beta"},
                        {"type": "image", "data": "ignored"},
                    ],
                }
            ],
        }
    ]


@given(parsers.parse('a user message with a text part "{keep}" and an assistant reply'))
def given_user_plus_assistant(ctx: dict, keep: str) -> None:
    ctx["messages"] = [
        {"role": "user", "content": [{"type": "text", "text": keep}]},
        {"role": "assistant", "content": "hi"},
    ]


@given(parsers.parse('a user message with a text part "{keep}"'))
def given_plain_user(ctx: dict, keep: str) -> None:
    ctx["messages"] = [{"role": "user", "content": [{"type": "text", "text": keep}]}]


@given(parsers.parse('a user message with a brave-pdf-text part "{payload}"'))
def given_pdf_str(ctx: dict, payload: str) -> None:
    ctx["messages"] = [
        {"role": "user", "content": [{"type": "brave-pdf-text", "text": payload}]}
    ]


@given("a leo messages list without a system message")
def given_leo_messages(ctx: dict) -> None:
    ctx["messages"] = [{"role": "user", "content": "hi"}]


@given(parsers.parse('a user memory content part with scalar memory "{pair}"'))
def given_scalar_memory(ctx: dict, pair: str) -> None:
    key, _, value = pair.partition(": ")
    ctx["part"] = UserMemoryContentPart.model_construct(
        type="brave-user-memory", memory={key: value}
    )


@given(parsers.parse('a user memory content part with list memory "{pair}"'))
def given_list_memory(ctx: dict, pair: str) -> None:
    key, _, items = pair.partition(": ")
    ctx["part"] = UserMemoryContentPart.model_construct(
        type="brave-user-memory", memory={key: items.split(", ")}
    )


@given("a user memory content part with no memory attribute")
def given_no_memory(ctx: dict) -> None:
    ctx["part"] = UserMemoryContentPart.model_construct(type="brave-user-memory")


@when("file extracted text augment runs")
def when_fet_augment(ctx: dict) -> None:
    ctx["result"] = file_extracted_text.augment(ctx["messages"])


@when("pdf text content augment runs")
def when_pdf_augment(ctx: dict) -> None:
    ctx["result"] = pdf_text_content.augment(ctx["messages"])


@when(parsers.parse('leo system augment runs with training cutoff "{cutoff}"'))
def when_leo_augment(ctx: dict, cutoff: str) -> None:
    ctx["result"] = leo_system_prompt.augment(
        ctx["messages"], model_config={"training_cutoff": cutoff}
    )


@when(parsers.parse("months since is computed for {cutoff} and {now}"))
def when_months_since(ctx: dict, cutoff: str, now: str) -> None:
    ctx["value"] = _leo_module._months_since(cutoff, datetime.strptime(now, "%Y-%m-%d"))


@when("the abstract Prompt augment is called directly")
def when_abstract_augment(ctx: dict) -> None:
    try:
        Prompt.augment(None, [])
        ctx["raised"] = None
    except NotImplementedError as exc:
        ctx["raised"] = exc


@when("the alignment trace text is computed")
def when_trace_text(ctx: dict) -> None:
    ctx["value"] = ctx["part"].get_alignment_trace_text()


@then("the message content is a single text part")
def then_single_text_part(ctx: dict) -> None:
    content = ctx["result"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"


@then(parsers.parse('the text part starts with "{prefix}"'))
def then_text_starts(ctx: dict, prefix: str) -> None:
    assert ctx["result"][0]["content"][0]["text"].startswith(prefix)


@then(parsers.parse('the text part contains "{snippet}"'))
def then_text_contains(ctx: dict, snippet: str) -> None:
    assert snippet in ctx["result"][0]["content"][0]["text"]


@then(parsers.parse('the text part is "{expected}"'))
def then_text_is(ctx: dict, expected: str) -> None:
    assert ctx["result"][0]["content"][0]["text"] == expected.replace("\\n", "\n")


@then(parsers.parse("the message count is {count}"))
def then_message_count(ctx: dict, count: str) -> None:
    assert len(ctx["result"]) == int(count)


@then(parsers.parse('the first message content keeps "{keep}"'))
def then_keeps(ctx: dict, keep: str) -> None:
    assert ctx["result"][0]["content"][0] == {"type": "text", "text": keep}


@then(parsers.parse('the system message contains "{snippet}"'))
def then_system_contains(ctx: dict, snippet: str) -> None:
    assert ctx["result"][0]["role"] == "system"
    assert snippet in ctx["result"][0]["content"]


@then(parsers.parse("the result is {expected}"))
def then_result(ctx: dict, expected: str) -> None:
    assert ctx["value"] == int(expected)


@then("a NotImplementedError is raised")
def then_raised(ctx: dict) -> None:
    assert isinstance(ctx["raised"], NotImplementedError)


@then(parsers.parse('the trace starts with "{prefix}"'))
def then_trace_starts(ctx: dict, prefix: str) -> None:
    assert ctx["value"].startswith(prefix)


@then(parsers.parse('the trace contains "{snippet}"'))
def then_trace_contains(ctx: dict, snippet: str) -> None:
    assert snippet in ctx["value"]


@then("the trace is none")
def then_trace_none(ctx: dict) -> None:
    assert ctx["value"] is None
