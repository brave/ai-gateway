import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
    AssistantMessage,
    ImageContentPart,
    ImageUrl,
    TextContentPart,
    ToolCall,
    ToolCallFunction,
    ToolMessage,
    UserMessage,
)
from aichat.responses import CompactionMetadata
from aichat.serve import open_ai_adapter
from aichat.serve.services.security_settings import security_settings
from aichat.serve.tool_parser import AlignmentCheckData

FEATURE = Path(__file__).parent / "features" / "openai_adapter.feature"
scenarios(str(FEATURE))


@pytest.fixture
def ctx():
    return {}


def _tool_chunk(name=None, arguments=None, finish_reason=None):
    delta_kwargs = {}
    if name is not None or arguments is not None:
        delta_kwargs["tool_calls"] = [
            ChoiceDeltaToolCall(
                index=0,
                id="call_1",
                type="function",
                function=ChoiceDeltaToolCallFunction(name=name, arguments=arguments),
            )
        ]
    return ChatCompletionChunk(
        id="log-1",
        object="chat.completion.chunk",
        created=1,
        model="test-model",
        choices=[
            Choice(
                index=0, delta=ChoiceDelta(**delta_kwargs), finish_reason=finish_reason
            )
        ],
    )


def _untrusted_history():
    return [
        UserMessage(content="hi"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=ToolCallFunction(name="web_search", arguments="{}"),
                )
            ],
        ),
        ToolMessage(role="tool", tool_call_id="call_1", content="fetched page"),
    ]


def _trusted_history():
    return [UserMessage(content="hi")]


def _mock_scanner(allowed=False, reasoning="blocked"):
    return SimpleNamespace(
        scan=AsyncMock(
            return_value=SimpleNamespace(allowed=allowed, reasoning=reasoning)
        )
    )


# --- ChunkEmitter ------------------------------------------------------------------------


@given(parsers.parse("a chunk emitter with compaction metadata {with_metadata}"))
def given_emitter(with_metadata, ctx):
    ctx["emitter"] = open_ai_adapter.ChunkEmitter(
        CompactionMetadata(
            summary="s",
            compacted_message_indices=[0],
            tokens_before=10,
            tokens_after=2,
        )
        if with_metadata == "yes"
        else None
    )
    ctx["with_metadata"] = with_metadata


@given(parsers.parse('a first chunk payload "{chunk_payload}"'))
def given_first_chunk(chunk_payload, ctx):
    ctx["chunk_payload"] = chunk_payload


@when("the chunk is emitted")
def when_emit(ctx):
    emitter = ctx["emitter"]
    payload = ctx["chunk_payload"]
    if payload.startswith("data: "):
        first = emitter.emit(payload + "\n\n")
    else:
        first = emitter.emit(payload)
    ctx["first_emitted"] = first
    ctx["second_emitted"] = emitter.emit('data: {"id":2}\n\n')


@then(
    parsers.parse("the emitted chunk carries compaction metadata {metadata_expected}")
)
def then_metadata(metadata_expected, ctx):
    expected = metadata_expected == "yes"
    if expected:
        assert "compaction" in ctx["first_emitted"]
    else:
        assert "compaction" not in ctx["first_emitted"]


@then(parsers.parse("a second emission carries no metadata {second_expected}"))
def then_second(second_expected, ctx):
    if second_expected == "yes":
        assert "compaction" not in ctx["second_emitted"]


# --- _maybe_truncate_response -------------------------------------------------------------


@given(parsers.parse("a trace response of length {length:d}"))
def given_trace_response(length, ctx):
    ctx["trace_response"] = "x" * length


@when("the response is truncated for the trace")
def when_truncate(ctx):
    ctx["truncated"] = open_ai_adapter._maybe_truncate_response(ctx["trace_response"])


@then(
    parsers.parse(
        'the truncated response starts with "..." {ellipsis} and keeps the tail {tail_kept}'
    )
)
def then_truncated(ellipsis, tail_kept, ctx):
    truncated = ctx["truncated"]
    if ellipsis == "yes":
        assert truncated.startswith("...")
        assert truncated.endswith("x" * 10)
    else:
        assert not truncated.startswith("...")
    assert truncated.endswith("x" * 10) if tail_kept == "yes" else True


# --- build_trace_from_openai_messages -------------------------------------------------------


@given(parsers.parse("an openai trace with a {description}"))
def given_trace_messages(description, ctx):
    messages = []
    assistant_response = None
    if description == "tool result":
        messages.append(ToolMessage(role="tool", tool_call_id="c1", content="out"))
    elif description == "plain user message":
        messages.append(UserMessage(content="hello there"))
    elif description == "image attachment":
        messages.append(
            UserMessage(
                content=[
                    ImageContentPart(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,AAA"),
                    )
                ]
            )
        )
    elif description == "assistant reply":
        messages.append(AssistantMessage(content="partial answer"))
    elif description == "assistant tool call":
        messages.append(
            AssistantMessage(
                content="thinking",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=ToolCallFunction(name="search", arguments='{"q":1}'),
                    )
                ],
            )
        )
    elif description == "system message":
        messages.append(SimpleNamespace(role="system", content="be nice"))
    elif description == "final assistant response":
        messages.append(UserMessage(content="hi"))
        assistant_response = "streamed tail"
    ctx["trace_messages"] = messages
    ctx["assistant_response"] = assistant_response


@when("the conversation trace is built")
def when_build_trace(ctx):
    ctx["trace"] = open_ai_adapter.build_trace_from_openai_messages(
        ctx["trace_messages"], ctx.get("assistant_response")
    )


@then(parsers.parse('the trace contains "{fragment}"'))
def then_trace_contains(fragment, ctx):
    assert fragment in ctx["trace"], ctx["trace"]


# --- extract_user_message_from_openai_messages ------------------------------------------------


@given(parsers.parse("an openai message list with user content {description}"))
def given_user_messages(description, ctx):
    description = description.replace(" ", "_")
    if description == "a_plain_string":
        ctx["user_messages"] = [UserMessage(content="hello there")]
    elif description == "text_parts":
        ctx["user_messages"] = [
            UserMessage(content=[TextContentPart(type="text", text="hello there")])
        ]
    elif description == "attachment_parts_only":
        ctx["user_messages"] = [
            UserMessage(
                content=[
                    ImageContentPart(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,AAA"),
                    )
                ]
            )
        ]
    else:  # no_user_message
        ctx["user_messages"] = [AssistantMessage(content="assistant only")]


@when("the original user message is extracted")
def when_extract_user(ctx):
    ctx["extracted_user"] = open_ai_adapter.extract_user_message_from_openai_messages(
        ctx["user_messages"]
    )


@then(parsers.parse("the extracted message is {extracted}"))
def then_extracted_user(extracted, ctx):
    expected = None if extracted == "none" else extracted
    assert ctx["extracted_user"] == expected


# --- perform_alignment_check --------------------------------------------------------------------


@given(parsers.parse("the alignment scanner is {scanner_state}"))
def given_alignment_scanner(scanner_state, monkeypatch):
    if scanner_state == "available":
        monkeypatch.setattr(open_ai_adapter, "alignment_scanner", _mock_scanner())
    else:
        monkeypatch.setattr(open_ai_adapter, "alignment_scanner", None)


@given("a conversation with untrusted tool results")
def given_untrusted_conversation(ctx):
    ctx["alignment_messages"] = _untrusted_history()


# TODO: remove test_open_ai_adapter_cov_test.py#TestAlignment test/aichat/serve/test_open_ai_adapter_cov_test.py:380
@when(
    parsers.parse(
        'an alignment check runs for tool "{tool}" with arguments "{arguments}"'
    )
)
def when_alignment_check(tool, arguments, ctx):
    ctx["alignment_result"] = asyncio.run(
        open_ai_adapter.perform_alignment_check(
            tool, arguments, ctx["alignment_messages"]
        )
    )


@then(parsers.parse("the alignment verdict allowed is {allowed}"))
def then_alignment_allowed(allowed, ctx):
    assert ctx["alignment_result"].allowed == (allowed == "True")


# --- extract_tool_arguments_from_buffered_chunks ---------------------------------------------------


@given(parsers.parse("buffered chunks with tool arguments {description}"))
def given_buffered_args(description, ctx):
    ctx["args_description"] = description
    if description == "a single chunk":
        ctx["buffered"] = [_tool_chunk(name="web_search", arguments='{"q": "x"}')]
    elif description == "split chunks":
        ctx["buffered"] = [
            _tool_chunk(name="web_search", arguments='{"q"'),
            _tool_chunk(name=None, arguments=": 1}"),
        ]
    else:
        ctx["buffered"] = []


@when("the tool arguments are extracted from the buffer")
def when_extract_args(ctx):
    ctx["buffer_args"] = open_ai_adapter.extract_tool_arguments_from_buffered_chunks(
        ctx["buffered"]
    )


@then(parsers.parse("the extracted arguments are {arguments}"))
def then_args(arguments, ctx):
    expected = "" if "nothing" in arguments else arguments.strip('"')
    assert ctx["buffer_args"] == expected


# --- process_non_streaming_alignment_check -----------------------------------------------------------


@given(parsers.parse("a non-streaming response with {description}"))
def given_ns_response(description, ctx):
    if description == "tool calls and untrusted content":
        ctx["ns_response"] = {
            "choices": [
                {
                    "message": {
                        "content": "calling tool",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "web_search", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }
        ctx["ns_messages"] = _untrusted_history()
    elif description == "no choices":
        ctx["ns_response"] = {"choices": []}
        ctx["ns_messages"] = _untrusted_history()
    elif description == "no tool calls":
        ctx["ns_response"] = {"choices": [{"message": {"content": "plain"}}]}
        ctx["ns_messages"] = _trusted_history()
    else:  # trusted content only
        ctx["ns_response"] = {
            "choices": [
                {
                    "message": {
                        "content": "calling tool",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "web_search", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }
        ctx["ns_messages"] = _trusted_history()
    ctx["ns_description"] = description


@given("the tool is in the alignment bypass list")
def given_bypass_tool(monkeypatch):
    monkeypatch.setattr(
        security_settings,
        "allowed_bypass_tools",
        {"alignment_check_bypass": ["web_search"], "tool_result_bypass": []},
    )


@when("the non-streaming alignment check runs")
def when_ns_alignment(ctx):
    ctx["ns_result"] = asyncio.run(
        open_ai_adapter.process_non_streaming_alignment_check(
            ctx["ns_response"], ctx["ns_messages"]
        )
    )


@then(parsers.parse("the response tool calls carry alignment metadata {decorated}"))
def then_ns_alignment(decorated, ctx):
    response = ctx["ns_result"]
    choices = response.get("choices", [{}])
    tool_calls = choices[0].get("message", {}).get("tool_calls") if choices else None
    if decorated == "yes":
        assert tool_calls and tool_calls[0]["alignment_check"]["allowed"] is False
    else:
        if tool_calls:
            assert "alignment_check" not in tool_calls[0]


# --- has_recent_untrusted_tool_results -----------------------------------------------------------------


@given(parsers.parse("a message history with {description}"))
def given_history(description, ctx, monkeypatch):
    kind = description.replace(" ", "_")
    ctx["pi_history"] = None
    if kind == "no_messages":
        ctx["history"] = []
    elif kind == "a_plain_user_reply_only":
        ctx["history"] = _trusted_history()
    elif kind == "bypassed_tool_results_only":
        monkeypatch.setattr(
            security_settings,
            "allowed_bypass_tools",
            {"alignment_check_bypass": [], "tool_result_bypass": ["web_search"]},
        )
        ctx["history"] = _untrusted_history()
    else:  # unbypassed_tool_results / tool_results_without_an_assistant_call
        ctx["history"] = _untrusted_history()
    if kind == "unbypassed_tool_results":
        ctx["pi_history"] = ctx["history"]
    elif kind == "no_tool_results":
        ctx["pi_history"] = _trusted_history()


@given("the tool_result_bypass list includes web_search")
def given_result_bypass(monkeypatch):
    monkeypatch.setattr(
        security_settings,
        "allowed_bypass_tools",
        {"alignment_check_bypass": [], "tool_result_bypass": ["web_search"]},
    )


# TODO: remove test_open_ai_adapter_cov_test.py#TestRecentToolResults test/aichat/serve/test_open_ai_adapter_cov_test.py:452
@when("recent tool results are checked")
def when_recent_untrusted(ctx, monkeypatch):
    if "bypassed tool results only" == ctx.get("history_desc"):
        pass
    ctx["recent_untrusted"] = open_ai_adapter.has_recent_untrusted_tool_results(
        ctx["history"]
    )


@then(parsers.parse("untrusted results are present {present}"))
def then_recent_untrusted(present, ctx):
    assert ctx["recent_untrusted"] == (present == "True")


# --- extract_latest_tool_call_results ------------------------------------------------------------------


@given(parsers.parse("tool result messages with content {description}"))
def given_tool_results(description, ctx):
    if description == "two string results":
        ctx["tool_messages"] = [
            ToolMessage(role="tool", tool_call_id="c1", content="first"),
            ToolMessage(role="tool", tool_call_id="c2", content="second"),
        ]
    elif description == "a text part result":
        ctx["tool_messages"] = [
            ToolMessage(
                role="tool",
                tool_call_id="c1",
                content=[TextContentPart(type="text", text="part text")],
            )
        ]
    else:  # no tool results
        ctx["tool_messages"] = [UserMessage(content="user only")]


# TODO: remove test_open_ai_adapter_cov_test.py#TestRecentToolResults test/aichat/serve/test_open_ai_adapter_cov_test.py:452
@when("the latest tool results are extracted")
def when_latest_results(ctx):
    ctx["latest_results"] = open_ai_adapter.extract_latest_tool_call_results(
        ctx["tool_messages"]
    )


@then(parsers.parse("the extracted content is {content}"))
def then_latest_results(content, ctx):
    expected = None if content == "none" else content.replace("-SEP-", "\n---\n")
    assert ctx["latest_results"] == expected


# --- maybe_perform_prompt_injection_scan ------------------------------------------------------------------


@given(parsers.parse("prompt injection scanning is {enabled}"))
def given_pi_enabled(enabled, monkeypatch):
    monkeypatch.setattr(
        security_settings, "prompt_injection_scanning_enabled", enabled == "True"
    )


@given(parsers.parse("the request capability is {capability}"))
def given_capability(capability, ctx):
    ctx["pi_capability"] = None if capability == "none" else capability


@when("the injection scan is launched")
def when_launch_scan(ctx):
    async def run():
        return open_ai_adapter.maybe_perform_prompt_injection_scan(
            _untrusted_history(), ctx.get("pi_capability")
        )

    ctx["scan_task"] = asyncio.run(run())


@then(parsers.parse("the scan task started {started}"))
def then_scan_started(started, ctx):
    if started == "yes":
        task = ctx["scan_task"]
        assert task is not None
        task.cancel()
    else:
        assert ctx["scan_task"] is None


# --- _perform_prompt_injection_scan -----------------------------------------------------------------------


@given(parsers.parse("the injection scanner is {scanner_state}"))
def given_injection_scanner(scanner_state, monkeypatch):
    if scanner_state == "available":
        monkeypatch.setattr(
            open_ai_adapter,
            "injection_scanner",
            SimpleNamespace(
                scan=AsyncMock(
                    return_value=SimpleNamespace(
                        probability=4, reasoning="suspicious", duration_ms=12.0
                    )
                )
            ),
        )
    else:
        monkeypatch.setattr(open_ai_adapter, "injection_scanner", None)


@when("the injection scan task executes")
def when_execute_scan(ctx):
    ctx["pi_result"] = asyncio.run(
        open_ai_adapter._perform_prompt_injection_scan(ctx["pi_history"])
    )


@then(parsers.parse("the scan probability is {probability:d}"))
def then_pi_probability(probability, ctx):
    assert ctx["pi_result"].probability == probability


# --- process_non_streaming_prompt_injection_scan --------------------------------------------------------------


@given(parsers.parse("a non-streaming response payload {payload}"))
def given_ns_payload(payload, ctx):
    if payload == "a dict":
        ctx["ns_payload"] = {"choices": [{"message": {"content": "ok"}}]}
    else:  # an object response

        class _Obj:
            def model_dump(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        ctx["ns_payload"] = _Obj()


@given(parsers.parse("an injection scan task that {task_state}"))
def given_scan_task(task_state, ctx):
    if task_state == "completes":
        ctx["ns_task"] = asyncio.run(_make_completed_task())
    elif task_state == "raises":
        ctx["ns_task"] = asyncio.run(_make_failed_task())
    else:
        ctx["ns_task"] = None


async def _make_completed_task():
    return asyncio.create_task(_resolve(SimpleNamespace(probability=3, reasoning="ok")))


async def _resolve(value):
    return value


async def _make_failed_task():
    return asyncio.create_task(_boom())


async def _boom():
    raise RuntimeError("scan failed")


@when("the non-streaming injection scan runs")
def when_ns_injection(ctx):
    ctx["ns_injected"] = asyncio.run(
        open_ai_adapter.process_non_streaming_prompt_injection_scan(
            ctx["ns_payload"], ctx["ns_task"]
        )
    )


@then(parsers.parse("the response carries prompt_injection_scan {carried}"))
def then_ns_injection(carried, ctx):
    response = ctx["ns_injected"]
    if carried == "yes":
        assert response["prompt_injection_scan"]["probability"] == 3
    else:
        assert "prompt_injection_scan" not in response


# --- handle_alignment_buffering ---------------------------------------------------------------------------------


@given("alignment checking is enabled")
def given_alignment_enabled(monkeypatch):
    monkeypatch.setattr(security_settings, "alignment_checking_enabled", True)


@given(parsers.parse('a stream chunk with tool call "{tool_name}"'))
def given_stream_chunk(tool_name, ctx):
    ctx["stream_chunk"] = _tool_chunk(name=tool_name, arguments="{}")


@given(parsers.parse("the buffer state is {state}"))
def given_buffer_state(state, ctx, monkeypatch):
    ctx["buffer_state"] = state
    ctx["buffer_manager"] = open_ai_adapter.OpenAIToolCallBufferManager()
    if state == "fresh":
        ctx["stream_chunk"] = _tool_chunk(name="web_search", arguments='{"q"')
    elif state == "finishing":
        manager = ctx["buffer_manager"]
        manager.start_buffering(_tool_chunk(name="web_search", arguments='{"q"'))
        ctx["stream_chunk"] = _tool_chunk(
            name=None, arguments=": 1}", finish_reason="tool_calls"
        )
        monkeypatch.setattr(open_ai_adapter, "perform_alignment_check", _fake_alignment)
        ctx["alignment"] = AlignmentCheckData(allowed=True, reasoning="ok")
    elif state == "flushing":
        manager = ctx["buffer_manager"]
        manager.start_buffering(_tool_chunk(name="web_search", arguments='{"q"'))
        manager.add_to_buffer(_tool_chunk(name=None, arguments=": 1}"))
        ctx["stream_chunk"] = ChatCompletionChunk(
            id="log-1",
            object="chat.completion.chunk",
            created=1,
            model="test-model",
            choices=[
                Choice(index=0, delta=ChoiceDelta(content="done"), finish_reason="stop")
            ],
        )
        monkeypatch.setattr(open_ai_adapter, "perform_alignment_check", _fake_alignment)
    else:  # passthrough: content chunk while buffering inactive
        ctx["stream_chunk"] = ChatCompletionChunk(
            id="log-1",
            object="chat.completion.chunk",
            created=1,
            model="test-model",
            choices=[
                Choice(index=0, delta=ChoiceDelta(content="hi"), finish_reason=None)
            ],
        )


async def _fake_alignment(*args, **kwargs):
    return AlignmentCheckData(allowed=True, reasoning="ok")


@when("the alignment buffering handles the chunk")
def when_handle_buffering(ctx):
    manager = ctx["buffer_manager"]
    should_continue, chunks, tool_name = asyncio.run(
        open_ai_adapter.handle_alignment_buffering(
            manager,
            ctx["stream_chunk"],
            getattr(ctx["stream_chunk"].choices[0], "finish_reason", None),
            _untrusted_history(),
            None,
            MagicMock(),
            [],
            None,
        )
    )
    ctx["buffer_continue"] = should_continue
    ctx["buffer_chunks"] = chunks
    ctx["buffer_tool_name"] = tool_name


@then(
    parsers.parse("buffering decision continue is {continue_flag} with chunks {chunks}")
)
def then_buffering(continue_flag, chunks, ctx):
    assert ctx["buffer_continue"] == (continue_flag == "True")
    assert len(ctx["buffer_chunks"]) == int(chunks)


# --- process_buffered_tool_call_chunks ---------------------------------------------------------------------------


@given(
    parsers.parse(
        'buffered tool call chunks for "{tool_name}" with alignment verdict allowed {allowed}'
    )
)
def given_buffered_tool_chunks(tool_name, allowed, monkeypatch, ctx):
    verdict = AlignmentCheckData(allowed=allowed == "True", reasoning="verdict")
    monkeypatch.setattr(
        open_ai_adapter,
        "perform_alignment_check",
        AsyncMock(return_value=verdict),
    )
    ctx["buffered_chunks"] = [
        _tool_chunk(name=tool_name, arguments='{"q": 1}'),
        _tool_chunk(name=None, arguments=": 2}", finish_reason="tool_calls"),
    ]
    ctx["tracked_tool_call"] = SimpleNamespace(
        function_name=tool_name, id="call_1", alignment_check=None
    )


@when("the buffered tool call chunks are processed")
def when_process_buffered(ctx):
    async def run():
        return [
            event
            async for event in open_ai_adapter.process_buffered_tool_call_chunks(
                {"name": "web_search", "id": "call_1"},
                ctx["buffered_chunks"],
                _untrusted_history(),
                None,
                None,
                tool_calls_in_response=[ctx["tracked_tool_call"]],
            )
        ]

    ctx["buffered_events"] = asyncio.run(run())


@then("the streamed chunk carries the alignment verdict")
def then_streamed_verdict(ctx):
    assert any("alignment_check" in e for e in ctx["buffered_events"]), ctx[
        "buffered_events"
    ]
    assert any(
        '"allowed":false' in e or '"allowed": false' in e
        for e in ctx["buffered_events"]
    )


@then("the tracked tool call carries the alignment verdict")
def then_tracked_verdict(ctx):
    assert ctx["tracked_tool_call"].alignment_check is not None
    assert ctx["tracked_tool_call"].alignment_check.reasoning == "verdict"
