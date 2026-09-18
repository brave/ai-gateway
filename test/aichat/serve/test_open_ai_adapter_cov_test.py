"""Coverage tests for aichat/serve/open_ai_adapter.py.

Authored against pre-lint-fix prod (a288d7e). All shapes probed dynamically in VM:
- openai lib chunk: ChatCompletionChunk/Choice/ChoiceDelta/ChoiceDeltaToolCall/
  ChoiceDeltaToolCallFunction from openai.types.chat(.chat_completion_chunk).
- Protocol messages: UserMessage/AssistantMessage/ToolMessage/TextContentPart.
- AlignmentCheckData(allowed, reasoning) dataclass from aichat.serve.tool_parser.
- AlignmentCheckResult(allowed, reasoning, duration_ms=0.0).
- PromptInjectionResult(probability, reasoning, duration_ms=0.0) dataclass.
- CompactionMetadata(summary, compacted_message_indices, tokens_before, tokens_after).
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from aichat.protocol.open_ai_protocol import (
    AssistantMessage,
    TextContentPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from aichat.responses import CompactionMetadata, PromptInjectionScanResult
from aichat.serve import open_ai_adapter as A


def _ptc(name="my_tool", arguments="{}", tc_id=None):
    return A.ToolCall(
        index=0, type="function", function={"name": name, "arguments": arguments}
    )


def _openai_chunk2(arguments):
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

    return ChatCompletionChunk(
        id="2",
        object="chat.completion.chunk",
        created=1,
        model="test-model",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            function=ChoiceDeltaToolCallFunction(arguments=arguments),
                        )
                    ]
                ),
                finish_reason=None,
            )
        ],
    )


from aichat.serve.tool_parser import AlignmentCheckData


def _chunk(tool_name=None, arguments="", finish_reason=None, content=None):
    tool_calls = None
    if tool_name is not None:
        tool_calls = [
            ChoiceDeltaToolCall(
                index=0,
                id="t1",
                function=ChoiceDeltaToolCallFunction(
                    name=tool_name, arguments=arguments
                ),
                type="function",
            )
        ]
    return ChatCompletionChunk(
        id="1",
        object="chat.completion.chunk",
        created=1,
        model="test-model",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
    )


def _tool_msg(content="tool result"):
    return ToolMessage(role="tool", tool_call_id="t1", content=content)


def _msgs():
    return [UserMessage(content="hello"), _tool_msg()]


# ---------------------------------------------------------------------------
# ChunkEmitter
# ---------------------------------------------------------------------------


class TestChunkEmitter:
    def test_no_compaction_info_passthrough(self):
        emitter = A.ChunkEmitter(None)
        chunk_str = 'data: {"a": 1}\n\n'
        assert emitter.emit(chunk_str) == chunk_str

    def test_injects_metadata_into_first_data_chunk(self):
        cm = CompactionMetadata.model_construct(
            summary="s", compacted_message_indices=[1, 2]
        )
        emitter = A.ChunkEmitter(cm)
        out = emitter.emit('data: {"id": "x"}\n\n')
        payload = json.loads(out[6:].strip())
        assert payload["metadata"]["compaction"]["compacted_message_indices"] == [1, 2]
        out2 = emitter.emit('data: {"id": "y"}\n\n')
        assert "metadata" not in json.loads(out2[6:].strip())

    def test_non_data_chunk_not_injected(self):
        cm = CompactionMetadata.model_construct(compacted_message_indices=[1])
        emitter = A.ChunkEmitter(cm)
        assert emitter.emit("data: [DONE]\n\n") == "data: [DONE]\n\n"

    def test_invalid_json_ignored(self):
        cm = CompactionMetadata.model_construct(compacted_message_indices=[1])
        emitter = A.ChunkEmitter(cm)
        assert emitter.emit("data: notjson\n\n") == "data: notjson\n\n"


# ---------------------------------------------------------------------------
# OpenAIToolCallBufferManager
# ---------------------------------------------------------------------------


class TestBufferManager:
    def test_should_start_while_buffering_false(self):
        mgr = A.OpenAIToolCallBufferManager()
        mgr.is_buffering = True
        assert mgr.should_start_buffering(_chunk("my_tool"), _msgs()) is False

    def test_should_start_no_untrusted_false(self, monkeypatch):
        monkeypatch.setattr(A, "does_contain_untrusted_content", lambda m: False)
        mgr = A.OpenAIToolCallBufferManager()
        assert mgr.should_start_buffering(_chunk("my_tool"), _msgs()) is False

    def test_should_start_no_tool_calls_false(self):
        mgr = A.OpenAIToolCallBufferManager()
        assert mgr.should_start_buffering(_chunk(content="hi"), _msgs()) is False

    def test_should_start_bypass_tool_false(self, monkeypatch):
        monkeypatch.setattr(
            A.security_settings,
            "allowed_bypass_tools",
            {"alignment_check_bypass": ["web_search"], "tool_result_bypass": []},
        )
        mgr = A.OpenAIToolCallBufferManager()
        assert mgr.should_start_buffering(_chunk("web_search"), _msgs()) is False

    def test_should_start_true(self, monkeypatch):
        monkeypatch.setattr(A, "does_contain_untrusted_content", lambda m: True)
        mgr = A.OpenAIToolCallBufferManager()
        assert mgr.should_start_buffering(_chunk("my_tool"), _msgs()) is True

    def test_start_extracts_tool_info_and_buffers(self):
        mgr = A.OpenAIToolCallBufferManager()
        c1 = _chunk("my_tool", arguments='{"a"')
        mgr.start_buffering(c1)
        assert mgr.is_buffering is True
        assert mgr.tool_info == {"name": "my_tool", "id": "t1"}
        assert mgr.buffer == [c1]

    def test_add_and_continue_and_get_and_reset(self):
        mgr = A.OpenAIToolCallBufferManager()
        c1 = _chunk("my_tool", arguments='{"a"')
        mgr.start_buffering(c1)
        c2 = _chunk("my_tool", arguments=": 1}")
        mgr.add_to_buffer(c2)
        assert mgr.should_continue_buffering(c2) is True
        assert mgr.should_continue_buffering(_chunk(content="x")) is False
        data = mgr.get_buffered_data()
        assert data["tool_info"]["name"] == "my_tool"
        mgr.reset()
        assert mgr.is_buffering is False
        assert mgr.buffer == []
        assert mgr.tool_info == {}

    def test_should_continue_finish_reason_true(self):
        mgr = A.OpenAIToolCallBufferManager()
        mgr.is_buffering = True
        assert (
            mgr.should_continue_buffering(_chunk("my_tool", finish_reason="tool_calls"))
            is True
        )

    def test_should_continue_plain_false(self):
        mgr = A.OpenAIToolCallBufferManager()
        mgr.is_buffering = True
        assert mgr.should_continue_buffering(_chunk(content="x")) is False


# ---------------------------------------------------------------------------
# StreamingInjectionScanEmitter
# ---------------------------------------------------------------------------


class TestStreamingInjectionScanEmitter:
    def test_scan_started_none_task(self):
        assert A.StreamingInjectionScanEmitter(None).scan_started_event() is None

    @pytest.mark.asyncio
    async def test_scan_started_with_task(self):
        fut = asyncio.get_running_loop().create_future()
        emitter = A.StreamingInjectionScanEmitter(fut)
        event = emitter.scan_started_event()
        assert event.model_dump()["type"] == "prompt_injection_scan"

    @pytest.mark.asyncio
    async def test_try_get_result_pending_none(self):
        fut = asyncio.get_running_loop().create_future()
        emitter = A.StreamingInjectionScanEmitter(fut)
        assert emitter.try_get_result() is None

    @pytest.mark.asyncio
    async def test_try_get_result_exception_none(self):
        fut = asyncio.get_running_loop().create_future()
        fut.set_exception(RuntimeError("boom"))
        assert A.StreamingInjectionScanEmitter(fut).try_get_result() is None

    @pytest.mark.asyncio
    async def test_try_get_result_done(self):
        fut: asyncio.Future = asyncio.Future()
        fut.set_result(PromptInjectionScanResult(probability=1, reasoning="r"))
        emitter = A.StreamingInjectionScanEmitter(fut)
        result = emitter.try_get_result()
        assert result is not None
        assert result.probability == 1

    @pytest.mark.asyncio
    async def test_await_result(self):
        fut: asyncio.Future = asyncio.Future()
        fut.set_result(PromptInjectionScanResult(probability=0, reasoning="ok"))
        emitter = A.StreamingInjectionScanEmitter(fut)
        result = await emitter.await_result()
        assert result.reasoning == "ok"

    @pytest.mark.asyncio
    async def test_await_result_exception_none(self):
        fut: asyncio.Future = asyncio.Future()
        fut.set_exception(RuntimeError("x"))
        emitter = A.StreamingInjectionScanEmitter(fut)
        assert await emitter.await_result() is None


# ---------------------------------------------------------------------------
# does_contain_untrusted_content / trace helpers
# ---------------------------------------------------------------------------


class TestUntrustedContent:
    def test_trusted_user_parts_false(self):
        msgs = [
            UserMessage(content=[TextContentPart(text="hi")]),
            AssistantMessage(content="resp"),
        ]
        assert A.does_contain_untrusted_content(msgs) is False

    def test_user_untrusted_part_true(self):
        msgs = [UserMessage.model_construct(role="user", content=[object()])]
        assert A.does_contain_untrusted_content(msgs) is True

    def test_non_bypass_tool_true(self):
        tc = ToolCall(
            index=0, type="function", function={"name": "my_tool", "arguments": "{}"}
        )
        msgs = [
            AssistantMessage.model_construct(
                role="assistant", content="", tool_calls=[tc]
            ),
            _tool_msg(),
        ]
        assert A.does_contain_untrusted_content(msgs) is True

    def test_all_bypass_false(self, monkeypatch):
        monkeypatch.setattr(
            A.security_settings,
            "allowed_bypass_tools",
            {"tool_result_bypass": ["my_tool"], "alignment_check_bypass": []},
        )
        tc = ToolCall(
            index=0, type="function", function={"name": "my_tool", "arguments": "{}"}
        )
        msgs = [
            AssistantMessage.model_construct(
                role="assistant", content="", tool_calls=[tc]
            ),
            _tool_msg(),
        ]
        assert A.does_contain_untrusted_content(msgs) is False

    def test_empty_false(self):
        assert A.does_contain_untrusted_content([]) is False


class TestTraceHelpers:
    def test_format_text_part(self):
        assert (
            A._format_content_part_for_trace(TextContentPart(text="hi")) == "USER: hi"
        )

    def test_truncate_long(self, monkeypatch):
        monkeypatch.setattr(A.security_settings, "alignment_truncation_char_limit", 5)
        out = A._maybe_truncate_response("abcdefgh")
        assert out.startswith("...") and out.endswith("h")

    def test_truncate_short(self):
        assert A._maybe_truncate_response("hi ") == "hi"

    def test_build_trace_tool_omitted(self):
        assert (
            A.build_trace_from_openai_messages([_tool_msg()])
            == "TOOL: tool output is omitted for security reasons"
        )

    def test_build_trace_user_and_assistant(self):
        msgs = [UserMessage(content="hello"), AssistantMessage(content="answer")]
        trace = A.build_trace_from_openai_messages(msgs)
        assert "USER: hello" in trace
        assert "ASSISTANT: " in trace

    def test_build_trace_assistant_tool_calls(self):
        tc = ToolCall(
            index=0,
            type="function",
            function={"name": "my_tool", "arguments": '{"x":1}'},
        )
        msgs = [
            AssistantMessage.model_construct(
                role="assistant", content="", tool_calls=[tc]
            )
        ]
        trace = A.build_trace_from_openai_messages(msgs)
        assert "ACTION: my_tool" in trace
        assert "ACTION INPUT:" in trace

    def test_extract_user_message_str(self):
        assert (
            A.extract_user_message_from_openai_messages([UserMessage(content="hi")])
            == "hi"
        )

    def test_extract_user_message_none(self):
        assert (
            A.extract_user_message_from_openai_messages([])
            == "User message is not found"
        )


# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------


class _FakeScanner:
    async def scan(self, user_message, trace, injection_scan_task=None):
        from aichat.services.alignment_check_service import AlignmentCheckResult as _ACR

        return _ACR(allowed=False, reasoning="blocked")


class TestAlignment:
    @pytest.mark.asyncio
    async def test_perform_alignment_check_scanner(self, monkeypatch):
        monkeypatch.setattr(A, "alignment_scanner", _FakeScanner())
        result = await A.perform_alignment_check(
            "my_tool", "{}", [UserMessage(content="u")]
        )
        assert result.allowed is False
        assert result.reasoning == "blocked"

    @pytest.mark.asyncio
    async def test_perform_alignment_check_no_scanner(self, monkeypatch):
        monkeypatch.setattr(A, "alignment_scanner", None)
        result = await A.perform_alignment_check("my_tool", "{}", [])
        assert result.allowed is True
        assert result.reasoning == "Alignment scanner not available"

    def test_extract_tool_arguments(self):
        chunks = [_chunk("my_tool", arguments='{"a"'), _openai_chunk2(": 1}")]
        assert A.extract_tool_arguments_from_buffered_chunks(chunks) == '{"a": 1}'


class TestBufferedStream:
    @pytest.mark.asyncio
    async def test_process_buffered_chunks_aligns_first_tool_chunk(self, monkeypatch):
        async def fake_align(
            tool_name, args, messages, assistant_response=None, injection_scan_task=None
        ):
            return AlignmentCheckData(allowed=False, reasoning="nope")

        monkeypatch.setattr(A, "perform_alignment_check", fake_align)
        chunks = [_chunk("my_tool", arguments="{}"), _chunk(finish_reason="tool_calls")]
        tool_calls_in_response = []
        out = [
            c
            async for c in A.process_buffered_tool_call_chunks(
                {"name": "my_tool", "id": "t1"},
                chunks,
                [UserMessage(content="u")],
                mcp_executor=MagicMock(),
                backend=MagicMock(),
                tool_calls_in_response=tool_calls_in_response,
            )
        ]
        assert out and out[0].startswith("data: ")
        payload = json.loads(out[0][6:].strip())
        tc = payload["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "my_tool"
        assert tc.get("alignment_check") == {"allowed": False, "reasoning": "nope"}
        assert len(out) == 1  # finish_reason chunk skipped (server-side execution)

    @pytest.mark.asyncio
    async def test_process_buffered_chunks_no_execution_yields_all(self, monkeypatch):
        async def fake_align(*a, **k):
            return AlignmentCheckData(allowed=True, reasoning=None)

        monkeypatch.setattr(A, "perform_alignment_check", fake_align)
        chunks = [_chunk("my_tool"), _chunk(finish_reason="tool_calls")]
        out = [
            c
            async for c in A.process_buffered_tool_call_chunks(
                {"name": "my_tool", "id": "t1"}, chunks, [UserMessage(content="u")]
            )
        ]
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Recent tool results
# ---------------------------------------------------------------------------


class TestRecentToolResults:
    def test_trailing_tool_messages_bypass(self, monkeypatch):
        monkeypatch.setattr(
            A.security_settings,
            "allowed_bypass_tools",
            {"tool_result_bypass": ["code_execution_tool"]},
        )
        tc = ToolCall(
            index=0,
            type="function",
            id="t1",
            function={"name": "code_execution_tool", "arguments": "{}"},
        )
        msgs = [
            AssistantMessage.model_construct(
                role="assistant", content="", tool_calls=[tc]
            ),
            _tool_msg("ok"),
        ]
        assert A.has_recent_untrusted_tool_results(msgs) is False

    def test_non_bypass_true(self, monkeypatch):
        monkeypatch.setattr(
            A.security_settings,
            "allowed_bypass_tools",
            {"tool_result_bypass": []},
        )
        msgs = [AssistantMessage(content="a"), _tool_msg()]
        assert A.has_recent_untrusted_tool_results(msgs) is True

    def test_no_trailing_tools_false(self):
        assert A.has_recent_untrusted_tool_results([UserMessage(content="u")]) is False

    def test_extract_latest_results_joined_reversed(self):
        msgs = [UserMessage(content="u"), _tool_msg("one"), _tool_msg("two")]
        assert A.extract_latest_tool_call_results(msgs) == "one\n---\ntwo"


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------


class TestPromptInjection:
    def test_disabled_none(self, monkeypatch):
        monkeypatch.setattr(
            A.security_settings, "prompt_injection_scanning_enabled", False
        )
        assert A.maybe_perform_prompt_injection_scan([], "content_agent") is None

    def test_capability_skip(self):
        assert A.maybe_perform_prompt_injection_scan([_tool_msg()], "chat") is None

    def test_no_recent_untrusted_none(self, monkeypatch):
        monkeypatch.setattr(A, "has_recent_untrusted_tool_results", lambda m: False)
        assert A.maybe_perform_prompt_injection_scan([], "content_agent") is None

    @pytest.mark.asyncio
    async def test_happy_creates_task(self, monkeypatch):
        async def fake_scan(messages):
            return None

        monkeypatch.setattr(A, "_perform_prompt_injection_scan", fake_scan)
        task = A.maybe_perform_prompt_injection_scan([_tool_msg()], "content_agent")
        assert task is not None
        task.cancel()

    @pytest.mark.asyncio
    async def test_scan_no_scanner(self, monkeypatch):
        monkeypatch.setattr(A, "injection_scanner", None)
        result = await A._perform_prompt_injection_scan([UserMessage(content="u")])
        assert result.probability == 1
        assert "not available" in result.reasoning

    @pytest.mark.asyncio
    async def test_scan_success(self):
        class _Sc:
            async def scan(self, content):
                return A.PromptInjectionResult(probability=5, reasoning="suspicious")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(A, "injection_scanner", _Sc())
            result = await A._perform_prompt_injection_scan([_tool_msg("payload")])
        assert result.probability == 5
        assert result.reasoning == "suspicious"

    @pytest.mark.asyncio
    async def test_non_streaming_scan(self):
        fut: asyncio.Future = asyncio.Future()
        fut.set_result(A.PromptInjectionResult(probability=2, reasoning="why"))
        response = {"choices": []}
        out = await A.process_non_streaming_prompt_injection_scan(response, fut)
        assert out["prompt_injection_scan"]["probability"] == 2

    @pytest.mark.asyncio
    async def test_non_streaming_scan_no_task(self):
        response = {"choices": []}
        out = await A.process_non_streaming_prompt_injection_scan(response, None)
        assert "prompt_injection_scan" not in out

    @pytest.mark.asyncio
    async def test_non_streaming_scan_exception(self):
        fut: asyncio.Future = asyncio.Future()
        fut.set_exception(RuntimeError("x"))
        response = {"choices": []}
        out = await A.process_non_streaming_prompt_injection_scan(response, fut)
        assert "prompt_injection_scan" not in out
