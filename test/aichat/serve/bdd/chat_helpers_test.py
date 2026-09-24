import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.protocol.open_ai_protocol import (
    Capability,
    File,
    FileContentPart,
    FileUrl,
    FileUrlContentPart,
    ImageContentPart,
    ImageUrl,
    InputAudio,
    InputAudioContentPart,
    TextContentPart,
    Tool,
    ToolFunction,
    UserMessage,
    VideoContentPart,
    VideoUrl,
)
from aichat.protocol.open_ai_protocol import Request as OpenAIRequest
from aichat.responses import InlineSearch, PromptInjectionScanResult
from aichat.serve import open_ai_api
from test.aichat.serve.bdd.chat_completions_test import (
    _client_tool_call_chunk,
    _content_chunk,
    _finish_tool_calls_chunk,
    _run_stream_pipeline,
)

FEATURE = Path(__file__).parent / "features" / "chat_helpers.feature"
scenarios(str(FEATURE))


@pytest.fixture
def ctx():
    return {}


def _data_payloads(events):
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: {"):
                payloads.append(json.loads(line[len("data: ") :]))
    return payloads


# --- last user message content -------------------------------------------------


@given(parsers.parse("a message list with last user content {description}"))
def given_message_list(description, ctx):
    if description == "a_plain_string":
        messages = [UserMessage(content="older"), UserMessage(content="hi there")]
    elif description == "a_list_of_text_parts":
        messages = [
            UserMessage(
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"},
                ]
            )
        ]
    elif description == "a_list_of_part_objects":
        messages = [
            UserMessage(content=[TextContentPart(type="text", text="part text")])
        ]
    elif description == "an_assistant_message":
        messages = [
            UserMessage(content="ignored"),
            SimpleNamespace(role="assistant", content="assistant text"),
        ]
    elif description == "only_empty_list":
        messages = [UserMessage(content=[])]
    else:  # no_messages
        messages = []
    ctx["messages"] = messages


@when("the last user message content is extracted")
def when_extract_last_user(ctx):
    ctx["extracted"] = open_ai_api.get_last_user_message_content(ctx["messages"])


@then(parsers.parse("the extracted content is {expected}"))
def then_extracted(expected, ctx):
    expected_value = None if expected == "none" else expected
    assert ctx["extracted"] == expected_value


# --- media detection ------------------------------------------------------------


@given(parsers.parse("a message list containing {media} content parts"))
def given_media_messages(media, ctx):
    if media == "an_image":
        part = ImageContentPart(
            type="image_url", image_url=ImageUrl(url="data:image/png;base64,AAA")
        )
    elif media == "a_file":
        part = FileContentPart(
            type="file",
            file=File(filename="doc.pdf", file_data="data:application/pdf;base64,AAA"),
        )
    elif media == "a_file_url":
        part = FileUrlContentPart(
            type="file_url", file_url=FileUrl(url="data:application/pdf;base64,AAA")
        )
    elif media == "a_video":
        part = VideoContentPart(
            type="video_url", video_url=VideoUrl(url="data:video/mp4;base64,AAA")
        )
    elif media == "an_audio":
        part = InputAudioContentPart(input_audio=InputAudio(data="AAA", format="wav"))
    else:  # text_only
        part = TextContentPart(type="text", text="just text")
    ctx["messages"] = [
        UserMessage(content=[part, TextContentPart(type="text", text="q")])
    ]


@when("the media content is detected")
def when_detect_media(ctx):
    ctx["media_type"] = open_ai_api.detect_media_content(ctx["messages"])


@then(parsers.parse("the detected media type is {media_type}"))
def then_media_type(media_type, ctx):
    expected = None if media_type == "none" else media_type
    assert ctx["media_type"] == expected


# --- metrics state ---------------------------------------------------------------


@when("the metrics state is created with start time 5.0")
def when_metrics_state(ctx):
    ctx["metrics_state"] = open_ai_api._new_metrics_state(start_time=5.0)


@then("the metrics state starts at 5.0 with first token and tool use unrecorded")
def then_metrics_state(ctx):
    state = ctx["metrics_state"]
    assert state["start_time"] == 5.0
    assert state["first_token_recorded"] is False
    assert state["tools_use_recorded"] is False
    assert state["truncation_continuation_depth"] == 0


# --- mid stream fallback detection ----------------------------------------------


@given(parsers.parse("an error chain whose root is named {root_name}"))
def given_error_chain(root_name, ctx):
    root_cls = type(root_name, (Exception,), {})
    wrapped = RuntimeError("mid")
    wrapped.__context__ = root_cls("root failure")
    ctx["error"] = wrapped


@when("the mid-stream fallback check runs")
def when_midstream_check(ctx):
    ctx["fallback"] = open_ai_api._is_mid_stream_fallback_error(ctx["error"])


@then(parsers.parse("the fallback detection result is {result}"))
def then_fallback_result(result, ctx):
    assert ctx["fallback"] == (result == "True")


# --- create_openai_response -------------------------------------------------------


# TODO: remove open_ai_api_test.py#test_unknown_response_type_returns_none test/aichat/serve/open_ai_api_test.py:1279
@when("an unregistered response payload is translated")
def when_translate_unknown(ctx):
    ctx["translation"] = open_ai_api.create_openai_response(
        SimpleNamespace(model_dump=lambda exclude_none=True: {"type": "unknown"})
    )


@then("no SSE translation is produced")
def then_no_translation(ctx):
    assert ctx["translation"] is None


# --- tool preparation --------------------------------------------------------------


@given(
    parsers.parse(
        "MCP integration is {mcp_enabled} and the model tool support is {tool_support}"
    )
)
def given_mcp_mode(mcp_enabled, tool_support, monkeypatch, ctx):
    ctx["tool_support"] = tool_support == "True"
    monkeypatch.setattr(open_ai_api.mcp_settings, "mcp_enabled", mcp_enabled == "True")
    monkeypatch.setattr(
        open_ai_api,
        "initialize_mcp_for_request",
        AsyncMock(return_value=([], None)),
    )


@given(parsers.parse('MCP initialization returns tools "{names}"'))
def given_mcp_tools(names, monkeypatch):
    def _tool(name):
        return Tool(
            type="function",
            function=ToolFunction(name=name, description="d", parameters={}),
        )

    tools = [_tool(n.strip()) for n in names.split(",")] if names.strip() else []
    monkeypatch.setattr(
        open_ai_api,
        "initialize_mcp_for_request",
        AsyncMock(return_value=(tools, MagicMock())),
    )


@given(parsers.parse('the client sends tool "{name}"'))
def given_client_tool(name, ctx):
    ctx["client_tools"] = [
        Tool(
            type="function",
            function=ToolFunction(name=name, description="d", parameters={}),
        )
    ]


@given(parsers.parse('the request excludes MCP tools "{exclude}"'))
def given_excludes(exclude, ctx):
    ctx["exclude"] = (
        None if exclude == "none" else [e.strip() for e in exclude.split(",")]
    )


@given(parsers.parse('the request includes MCP tools "{include}"'))
def given_includes(include, ctx):
    ctx["include"] = (
        None if include == "none" else [i.strip() for i in include.split(",")]
    )


@given(parsers.parse("the client advertises deep research {deep_research}"))
def given_deep_research(deep_research, ctx):
    ctx["deep_research"] = deep_research == "True"


@when("tools are prepared for the request")
def when_prepare_tools(ctx):
    model_config = MagicMock()
    model_config.tool_support = ctx.get("tool_support", True)
    _, tool_names, _ = asyncio.run(
        open_ai_api.prepare_tools_for_request(
            client_tools=ctx.get("client_tools"),
            model_config=model_config,
            mcp_tools_exclude=ctx.get("exclude"),
            mcp_tools_include=ctx.get("include"),
            brave_capability=(
                [Capability.deep_research] if ctx.get("deep_research") else None
            ),
        )
    )
    ctx["tool_names"] = tool_names


@then(parsers.parse('the prepared tool names are "{tool_names}"'))
def then_tool_names(tool_names, ctx):
    expected = [n.strip() for n in tool_names.split(",")] if tool_names.strip() else []
    assert ctx["tool_names"] == expected


@given("MCP initialization fails with a network error")
def given_mcp_network_error(monkeypatch):
    monkeypatch.setattr(
        open_ai_api,
        "initialize_mcp_for_request",
        AsyncMock(side_effect=httpx.ConnectError("boom")),
    )


@given("MCP initialization fails with malformed JSON")
def given_mcp_json_error(monkeypatch):
    monkeypatch.setattr(
        open_ai_api,
        "initialize_mcp_for_request",
        AsyncMock(side_effect=json.JSONDecodeError("bad", "doc", 0)),
    )


# --- augment_messages ---------------------------------------------------------------


@given(parsers.parse("a free token limit of {limit:d}"))
def given_free_limit(limit, ctx):
    ctx["token_limit_free"] = limit


@given(parsers.parse("a premium token limit of {limit:d}"))
def given_premium_limit(limit, ctx):
    ctx["token_limit_premium"] = limit


@given(parsers.parse('a prompt that appends "{marker}" to the messages'))
def given_marker_prompt(marker, monkeypatch):
    class MarkerPrompt:
        def augment(self, messages, **kwargs):
            augmented = [dict(m) for m in messages]
            augmented.append({"role": "system", "content": marker})
            return augmented

    monkeypatch.setattr(
        open_ai_api, "prompts", SimpleNamespace(prompts=[MarkerPrompt()])
    )


@given(parsers.parse('a single user message "{content}"'))
def given_user_message(content, ctx):
    ctx["augment_messages"] = [UserMessage(content=content)]


@when(parsers.parse("the messages are augmented {access}"))
def when_augmented(access, ctx, monkeypatch):
    captured = {}

    async def fake_pdf_limits(messages, token_limit):
        captured["token_limit"] = token_limit
        return messages

    monkeypatch.setattr(open_ai_api, "process_messages_for_pdf_limits", fake_pdf_limits)
    ctx["pdf_capture"] = captured
    is_premium = access == "with premium access"
    model_config = MagicMock()
    model_config.conversation_token_limit = ctx.get("token_limit_free")
    model_config.conversation_token_limit_premium = ctx.get("token_limit_premium")
    ctx["augmented"] = asyncio.run(
        open_ai_api.augment_messages(
            messages=ctx["augment_messages"],
            tools=[],
            model_config=model_config,
            is_premium=is_premium,
        )
    )


@then(parsers.parse('the augmented messages contain "{marker}"'))
def then_augmented_contains(marker, ctx):
    assert any(m.get("content") == marker for m in ctx["augmented"]), ctx["augmented"]


@then(parsers.parse("the PDF limit check received the {tier} token limit"))
def then_pdf_limit(tier, ctx):
    expected = (
        ctx["token_limit_premium"] if tier == "premium" else ctx["token_limit_free"]
    )
    assert ctx["pdf_capture"]["token_limit"] == expected


# --- pipeline scenarios ---------------------------------------------------------------


@given(parsers.parse('a streaming chat request for model "{model}"'))
def given_stream_req(model, ctx):
    ctx["model"] = model
    ctx["request"] = OpenAIRequest(
        model=model,
        messages=[UserMessage(content="Hello")],
        stream=True,
        tools=None,
    )


@given(
    parsers.parse(
        'a streaming chat request for model "{model}" with deep research capability'
    )
)
def given_stream_dr(model, ctx):
    ctx["model"] = model
    ctx["request"] = OpenAIRequest(
        model=model,
        messages=[UserMessage(content="Hello")],
        stream=True,
        tools=None,
        brave_capability=[Capability.deep_research],
    )


@when("the stream pipeline processes no chunks at all")
def when_empty_stream(ctx, monkeypatch):
    ctx["events"] = _run_stream_pipeline(ctx, monkeypatch, chunks=[])


@when("the stream pipeline processes an abandoned tool call")
def when_abandoned_tool_call(ctx, monkeypatch):
    ctx["events"] = _run_stream_pipeline(
        ctx, monkeypatch, chunks=[_client_tool_call_chunk()]
    )


@when(parsers.parse('the stream pipeline processes a chunk with content "{content}"'))
def when_content_chunk(content, ctx, monkeypatch):
    chunk = ChatCompletionChunk(
        id="log-1",
        object="chat.completion.chunk",
        created=1,
        model="test-model",
        choices=[
            Choice(index=0, delta=ChoiceDelta(content=content), finish_reason=None)
        ],
    )
    ctx["events"] = _run_stream_pipeline(ctx, monkeypatch, chunks=[chunk])


@given("a pending prompt injection scan that completes with a result")
def given_pi_scan(monkeypatch):
    emitter = MagicMock()
    emitter.scan_started_event.return_value = PromptInjectionScanResult(
        probability=1, reasoning="benign"
    )
    emitter.try_get_result.return_value = None
    emitter.await_result = AsyncMock(
        return_value=PromptInjectionScanResult(probability=5, reasoning="suspicious")
    )
    monkeypatch.setattr(
        open_ai_api, "StreamingInjectionScanEmitter", MagicMock(return_value=emitter)
    )


@given("an inline search completing during the stream")
def given_inline_search(monkeypatch):
    inline = InlineSearch(query="q", results=[])
    helper = MagicMock()
    helper.pop_completed_searches.side_effect = [[inline], []]
    helper.handle_received_completion = MagicMock()
    helper.get_inline_search_chunks = AsyncMock(return_value=[inline])
    monkeypatch.setattr(
        open_ai_api, "InlineSearchHelper", MagicMock(return_value=helper)
    )


@when(
    parsers.parse(
        'the stream pipeline processes a chunk with model "{chunk_model}" and requested model "{requested}"'
    )
)
def when_chunk_requested_model(chunk_model, requested, ctx, monkeypatch):
    chunk = ChatCompletionChunk(
        id="log-1",
        object="chat.completion.chunk",
        created=1,
        model=chunk_model,
        choices=[Choice(index=0, delta=ChoiceDelta(content="hi"), finish_reason=None)],
    )
    ctx["events"] = _run_stream_pipeline(
        ctx, monkeypatch, chunks=[chunk], requested_model=requested
    )


@then(parsers.parse('the streamed chunk model equals "{expected}"'))
def then_chunk_model(expected, ctx):
    payloads = _data_payloads(ctx["events"])
    models = [p.get("model") for p in payloads if "model" in p]
    assert expected in models, models


@then("the stream still ends with a DONE sentinel")
def then_stream_done(ctx):
    assert ctx["events"][-1].strip() == "data: [DONE]"


@then("the scan started event is streamed first")
def then_scan_started_first(ctx):
    payload = json.loads(ctx["events"][0][len("data: ") :])
    assert payload["probability"] == 1
    assert payload["reasoning"] == "benign"


@then("the scan result event is streamed last before DONE")
def then_scan_result_last(ctx):
    assert ctx["events"][-1].strip() == "data: [DONE]"
    payload = json.loads(ctx["events"][-2][len("data: ") :])
    assert payload["probability"] == 5


@then("the inline search result is streamed in the events")
def then_inline_streamed(ctx):
    assert any("inlineSearch" in e for e in ctx["events"]), ctx["events"]


@given("a stream that fails with a MidStreamFallbackError")
def given_failing_stream(monkeypatch, ctx):
    root_cls = type("MidStreamFallbackError", (Exception,), {})
    err = RuntimeError("wrapped")
    err.__context__ = root_cls("litellm exploded")

    async def failing_gen():
        yield _content_chunk("partial")
        raise err

    ctx["failing_stream"] = failing_gen
    monkeypatch.setattr(
        open_ai_api,
        "handle_litellm_error",
        MagicMock(return_value={"content": "recovered message"}),
    )


@when("the stream pipeline is consumed")
def when_consume_pipeline(ctx, monkeypatch):
    if ctx.get("failing_stream"):
        ctx["events"] = _run_stream_pipeline(
            ctx, monkeypatch, [], stream_factory=ctx.pop("failing_stream")
        )
        return

    if ctx.get("dr_tracking") is not None:
        executor = MagicMock()
        executor.is_mcp_tool = AsyncMock(return_value=False)

        async def fake_streaming_tools(**kwargs):
            ctx["streaming_tool_calls"] = True
            return
            yield  # pragma: no cover

        monkeypatch.setattr(
            open_ai_api, "handle_streaming_tool_calls", fake_streaming_tools
        )
        ctx["pipeline_executor"] = executor

    chunks = ctx.pop("pipeline_chunks", None) or []
    if "pipeline_executor" in ctx:
        ctx["events"] = _run_stream_pipeline(
            ctx,
            monkeypatch,
            chunks,
            mcp_executor=ctx.pop("pipeline_executor"),
        )
    else:
        ctx["events"] = _run_stream_pipeline(ctx, monkeypatch, chunks)


@then("the stream contains a recovered error chunk")
def then_recovered_chunk(ctx):
    assert any("recovered message" in e for e in ctx["events"]), ctx["events"]


@given(parsers.parse('a completed tool call for "{name}"'))
def given_completed_tool_call(name, ctx):
    tool_call = ChoiceDeltaToolCall(
        index=0,
        id="call_9",
        type="function",
        function=ChoiceDeltaToolCallFunction(name=name, arguments='{"q": "x"}'),
    )
    start = ChatCompletionChunk(
        id="log-1",
        object="chat.completion.chunk",
        created=1,
        model="test-model",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(tool_calls=[tool_call]),
                finish_reason=None,
            )
        ],
    )
    finish = ChatCompletionChunk(
        id="log-1",
        object="chat.completion.chunk",
        created=1,
        model="test-model",
        choices=[Choice(index=0, delta=ChoiceDelta(), finish_reason="tool_calls")],
    )
    ctx["pipeline_chunks"] = [start, finish]


@then(parsers.parse('the streaming research handler ran for "{name}"'))
def then_streaming_research(name, ctx):
    assert ctx["streaming_tool_calls"] is True


@then("the deep research capability metric sees the capability unused")
def then_dr_metric_unused(ctx):
    recorded = ctx["dr_tracking"]
    label_calls = recorded["labels"].call_args_list
    assert any(
        call.args[:2] == ("test-model", "False") for call in label_calls
    ), label_calls


@when(
    "the stream pipeline processes an MCP tool call that streams an event and returns a result"
)
def when_mcp_event_and_result(ctx, monkeypatch):
    from aichat.protocol.open_ai_protocol import ToolMessage

    backend = MagicMock()
    backend.build_params.return_value = {}

    async def followup_gen():
        yield _content_chunk("follow-up answer")

    backend.converse = AsyncMock(return_value=followup_gen())

    executor = MagicMock()
    executor.is_mcp_tool = AsyncMock(return_value=True)

    async def tool_events(*args, **kwargs):
        yield ('data: {"brave-chat.toolStart":true}\n\n', None)
        yield (
            None,
            ToolMessage(role="tool", tool_call_id="call_1", content="tool output"),
        )

    monkeypatch.setattr(open_ai_api, "execute_tools_and_stream_events", tool_events)
    ctx["events"] = _run_stream_pipeline(
        ctx,
        monkeypatch,
        chunks=[_client_tool_call_chunk(), _finish_tool_calls_chunk()],
        backend=backend,
        mcp_executor=executor,
    )
    ctx["backend"] = backend


@then("the streamed tool event is forwarded")
def then_tool_event_forwarded(ctx):
    assert any("toolStart" in e for e in ctx["events"]), ctx["events"]


@then("the follow-up conversation is executed with the tool result")
def then_followup(ctx):
    backend = ctx["backend"]
    backend.converse.assert_awaited_once()
    llm_messages = backend.converse.await_args.args[0]
    tool_messages = [m for m in llm_messages if m.get("role") == "tool"]
    assert tool_messages, llm_messages
    assert tool_messages[0]["content"] == "tool output"


# --- deep research metric tracking fixture ---------------------------------------------


@pytest.fixture
def _dr_recorder():
    labels = MagicMock()
    metric = MagicMock(labels=labels)
    return metric, labels


@pytest.fixture
def dr_tracking(monkeypatch):
    labels = MagicMock()
    labels.return_value = MagicMock(inc=MagicMock())
    metric = MagicMock(labels=labels)
    monkeypatch.setattr(open_ai_api, "DEEP_RESEARCH_CAPABILITY_TOTAL", metric)
    tracker = {"metric": metric, "labels": labels}
    yield tracker


@given("deep research tracking is armed")
def given_dr_tracking(dr_tracking, ctx):
    ctx["dr_tracking"] = dr_tracking
