from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.protocol.open_ai_protocol import (
    AssistantMessage,
    File,
    FileContentPart,
    FileUrl,
    FileUrlContentPart,
    ImageContentPart,
    ImageUrl,
    TextContentPart,
    ToolCall,
    ToolCallFunction,
    ToolMessage,
    UserMessage,
)
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.message_preprocessing import preprocess_messages
from test.aichat.serve.bdd.helpers import (
    dataclass_model_config as _model_config,
)

FEATURE = Path(__file__).parent / "features" / "message_preprocessing.feature"
scenarios(str(FEATURE))


@pytest.fixture
def ctx():
    return {}


def _image():
    return ImageContentPart(
        type="image_url", image_url=ImageUrl(url="data:image/png;base64,AAA")
    )


def _file_part(name, data="data:application/pdf;base64,AAA"):
    return FileContentPart(file=File(filename=name, file_data=data))


@given(parsers.parse("a model that supports images with image limit {limit:d}"))
def given_image_limit(limit, ctx, monkeypatch):
    ctx["model_config"] = _model_config()
    monkeypatch.setattr(external_service_settings, "global_image_limit", limit)
    monkeypatch.setattr(external_service_settings, "bedrock_image_limit", limit)


@given(parsers.parse('messages carrying images "{initial}"'))
def given_image_messages(initial, ctx):
    if initial == "over_limit":
        ctx["messages"] = [
            UserMessage(content=[_image(), _image(), _image()]),
        ]
    else:
        ctx["messages"] = [UserMessage(content=[_image(), _image()])]


@given(parsers.parse("a bedrock model with document limit {limit:d}"))
def given_bedrock_doc_limit(limit, ctx, monkeypatch):
    ctx["model_config"] = _model_config(backend="bedrock")
    monkeypatch.setattr(external_service_settings, "bedrock_document_limit", limit)
    monkeypatch.setattr(external_service_settings, "bedrock_image_limit", 10)


@given(parsers.parse("a non-bedrock model with document limit none"))
def given_non_bedrock_no_doc_limit(ctx):
    ctx["model_config"] = _model_config()


@given(parsers.parse("messages carrying two documents and one image"))
def given_two_docs_one_image(ctx):
    ctx["messages"] = [
        UserMessage(
            content=[
                _file_part("old.pdf", "data:application/pdf;base64,AAA"),
                _file_part("new.pdf", "data:application/pdf;base64,BBB"),
                _image(),
            ]
        )
    ]


@given("a bedrock model")
def given_bedrock_model(ctx):
    ctx["model_config"] = _model_config(backend="bedrock")


@given("any model")
def given_any_model(ctx):
    ctx["model_config"] = _model_config()


@given("two attachments with identical content and different names")
def given_duplicate_content(ctx):
    ctx["messages"] = [
        UserMessage(
            content=[
                _file_part("a.pdf", "data:application/pdf;base64,SAME"),
                _file_part("b.pdf", "data:application/pdf;base64,SAME"),
                _file_part("c.pdf", "data:application/pdf;base64,OTHER"),
            ]
        )
    ]


@given('two attachments named "report.pdf" with different content')
def given_duplicate_names(ctx):
    ctx["messages"] = [
        UserMessage(
            content=[
                _file_part("report.pdf", "data:application/pdf;base64,AAA"),
                _file_part("report.pdf", "data:application/pdf;base64,BBB"),
            ]
        )
    ]


@given("a model without tool support")
def given_no_tool_support(ctx):
    ctx["model_config"] = _model_config(tool_support=False)


@given("a conversation with tool result and assistant tool calls")
def given_tool_conversation(ctx):
    ctx["messages"] = [
        UserMessage(content="what is the weather"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=ToolCallFunction(name="get_weather", arguments="{}"),
                )
            ],
        ),
        ToolMessage(role="tool", tool_call_id="call_1", content="sunny"),
        AssistantMessage(content="It is sunny."),
    ]


@given("a model without file support")
def given_no_file_support(ctx):
    ctx["model_config"] = _model_config(file_support=False)


@given("a message with an inline file")
def given_file_message(ctx):
    ctx["messages"] = [
        UserMessage(
            content=[
                _file_part("report.pdf"),
                TextContentPart(type="text", text="summarize"),
            ]
        )
    ]


# NOTE: FileUrlContentPart parts crash _strip_file_parts in production
# (aichat/serve/message_preprocessing.py accesses part.file which
# FileUrlContentPart does not define). The inline-file scenario above is
# limited to inline files; the dedicated xfail scenario below pins the bug.


@given("a message with a file url part")
def given_file_url_message(ctx):
    ctx["messages"] = [
        UserMessage(
            content=[
                FileUrlContentPart(
                    type="file_url",
                    file_url=FileUrl(url="data:application/pdf;base64,AAAA"),
                ),
                TextContentPart(type="text", text="summarize"),
            ]
        )
    ]


@when("the file url part is preprocessed")
def when_file_url_preprocessed(ctx):
    try:
        ctx["result"] = preprocess_messages(ctx["messages"], ctx["model_config"])
        ctx["file_url_error"] = None
    except Exception as exc:
        ctx["file_url_error"] = exc


@then("preprocessing fails with the known file url strip bug")
def then_file_url_strip_bug(ctx):
    err = ctx["file_url_error"]
    if err is None:
        pytest.fail(
            "prod bug fixed: rewrite this scenario to assert text-note replacement"
        )
    assert isinstance(err, AttributeError)
    pytest.xfail(f"prod bug: _strip_file_parts assumes part.file ({err})")


@given("a model without image support")
def given_no_image_support(ctx):
    ctx["model_config"] = _model_config(image_support=False)


@given("a message with an image")
def given_image_message(ctx):
    ctx["messages"] = [
        UserMessage(
            content=[_image(), TextContentPart(type="text", text="what is this")]
        )
    ]


@when("the messages are preprocessed")
def when_preprocessed(ctx):
    ctx["result"] = preprocess_messages(ctx["messages"], ctx["model_config"])


def _part_kinds(messages, kind):
    return [
        part
        for message in messages
        if isinstance(message.content, list)
        for part in message.content
        if isinstance(part, kind)
    ]


@then(parsers.parse('the surviving image layout is "{surviving}"'))
def then_image_layout(surviving, ctx):
    images = _part_kinds(ctx["result"], ImageContentPart)
    if surviving == "over_limit":
        assert len(images) == 3
    else:
        assert len(images) == 2


@then("only the newest document survives")
def then_newest_document(ctx):
    docs = _part_kinds(ctx["result"], FileContentPart)
    assert len(docs) == 1
    assert docs[0].file.filename == "new.pdf"


@then("both documents survive")
def then_both_documents(ctx):
    docs = _part_kinds(ctx["result"], FileContentPart)
    assert len(docs) == 2


@then("only the last attachment with that content survives")
def then_last_attachment(ctx):
    docs = _part_kinds(ctx["result"], FileContentPart)
    names = [d.file.filename for d in docs]
    assert names == ["b.pdf", "c.pdf"]


@then("the first attachment keeps its name")
def then_first_name_kept(ctx):
    docs = _part_kinds(ctx["result"], FileContentPart)
    assert docs[0].file.filename == "report.pdf"


@then(parsers.parse('the second attachment is renamed like "report.pdf-1-<hash>"'))
def then_second_renamed(ctx):
    docs = _part_kinds(ctx["result"], FileContentPart)
    second = docs[1].file.filename
    assert second.startswith("report.pdf-1-"), second
    assert len(second) == len("report.pdf-1-") + 8


@then("tool results are removed and assistant tool calls are cleared")
def then_tools_stripped(ctx):
    result = ctx["result"]
    assert not [m for m in result if isinstance(m, ToolMessage)]
    assistants = [m for m in result if isinstance(m, AssistantMessage)]
    tool_call_assistants = [m for m in assistants if m.tool_calls]
    assert tool_call_assistants == []


@then("assistant-only messages without content are removed")
def then_empty_assistants_removed(ctx):
    assistants = [
        m for m in ctx["result"] if isinstance(m, AssistantMessage) and not m.content
    ]
    assert assistants == []


@then("the file parts are replaced with explanatory text notes")
def then_file_notes(ctx):
    files = _part_kinds(ctx["result"], FileContentPart) + _part_kinds(
        ctx["result"], FileUrlContentPart
    )
    assert files == []
    texts = _part_kinds(ctx["result"], TextContentPart)
    assert any("file attachments" in (t.text or "") for t in texts)


@then("the image is replaced with an explanatory text note")
def then_image_note(ctx):
    assert _part_kinds(ctx["result"], ImageContentPart) == []
    texts = _part_kinds(ctx["result"], TextContentPart)
    assert any("image attachments" in (t.text or "") for t in texts)
