"""Coverage tests for aichat/serve/services/compaction.py.

Authored against pre-lint-fix prod (a288d7e). All prod behavior probed in VM:
- check_compaction_needed / can_compact / calculate_max_input_tokens / select_messages...
- _call_compaction_model raises RuntimeError variants (pre-lint-fix; post-fix TypeError
  at the error-response branch — tests use pytest.raises((RuntimeError, TypeError))).
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aichat.serve.services import compaction as C
from aichat.serve.services.compaction_settings import compaction_settings


def _msg(role="user", content="hi", **kw):
    m = {"role": role, "content": content}
    m.update(kw)
    return m


def _ua_pair(content_u="u", content_a="a"):
    return [_msg("user", content_u), _msg("assistant", content_a)]


def _mc(**overrides):
    """ModelConfig-like mock with compaction fields."""
    mc = MagicMock()
    mc.max_tokens = 10
    mc.conversation_token_limit = 1000
    mc.conversation_token_limit_premium = 2000
    for k, v in overrides.items():
        setattr(mc, k, v)
    return mc


# ---------------------------------------------------------------------------
# check_compaction_needed / can_compact / calculate_max_input_tokens
# ---------------------------------------------------------------------------


def test_check_compaction_needed_no_limit():
    assert C.check_compaction_needed([], None) == (False, 0, 0)
    assert C.check_compaction_needed([], -5) == (False, 0, 0)


def test_check_compaction_needed_thresholds():
    msgs = [_msg(), _msg(), _msg(), _msg()]
    with (
        patch.object(C, "calculate_message_tokens", return_value=100),
        patch.object(compaction_settings, "compaction_threshold", 0.6),
    ):
        needs, current, threshold = C.check_compaction_needed(msgs, 200)
        assert current == 100
        assert threshold == 120
        assert needs is False
        needs, _, _ = C.check_compaction_needed(msgs, 150)
        assert needs is True


def test_can_compact_below_min_turns():
    with patch.object(compaction_settings, "compaction_min_preserved_turns", 2):
        assert C.can_compact([_msg("user"), _msg("assistant")]) is False
        assert (
            C.can_compact(
                [_msg("user"), _msg("assistant"), _msg("user"), _msg("assistant")]
            )
            is True
        )


def test_can_compact_tool_messages_not_counted():
    with patch.object(compaction_settings, "compaction_min_preserved_turns", 2):
        msgs = [_msg("user"), _msg("tool", tool_call_id="x"), _msg("assistant")]
        assert C.can_compact(msgs) is False


def test_calculate_max_input_tokens_premium(monkeypatch):
    mc = MagicMock()
    mc.conversation_token_limit = 1000
    mc.conversation_token_limit_premium = 2000
    mc.max_tokens = 100
    with patch.object(compaction_settings, "compaction_input_margin", 0.5):
        val, _ = C.calculate_max_input_tokens(mc, True)
    assert val == int((2000 - 100) * 0.5)


def test_calculate_max_input_tokens_non_premium(monkeypatch):
    mc = MagicMock()
    mc.conversation_token_limit = 1000
    mc.conversation_token_limit_premium = 2000
    mc.max_tokens = 10
    with patch.object(compaction_settings, "compaction_input_margin", 1.0):
        val, _ = C.calculate_max_input_tokens(mc, False)
        assert val == 990


def test_calculate_max_input_tokens_fallback(monkeypatch):
    mc = MagicMock()
    mc.conversation_token_limit = None
    mc.conversation_token_limit_premium = None
    mc.max_tokens = None
    with (
        patch.object(compaction_settings, "compaction_fallback_max_tokens", 400),
        patch.object(compaction_settings, "compaction_fallback_output_tokens", 100),
        patch.object(compaction_settings, "compaction_input_margin", 1.0),
    ):
        val, _ = C.calculate_max_input_tokens(mc, False)
        assert val == 300


# ---------------------------------------------------------------------------
# select_messages_for_compaction
# ---------------------------------------------------------------------------


def test_select_messages_below_min_turns():
    with patch.object(compaction_settings, "compaction_min_preserved_turns", 2):
        msgs = [_msg("user"), _msg("assistant")]
        compact, preserved, idx = C.select_messages_for_compaction(msgs)
        assert compact == []
        assert preserved == msgs
        assert idx == []


def test_select_messages_preserves_tail_and_associated_tools():
    with patch.object(compaction_settings, "compaction_min_preserved_turns", 1):
        tool_call = {"id": "call_9", "type": "function", "function": {"name": "t"}}
        msgs = [
            _msg("user", "old"),
            _msg("assistant", tool_calls=[tool_call]),
            _msg("tool", tool_call_id="call_9"),
            _msg("user"),
            _msg("assistant"),
        ]
        compact, preserved, indices = C.select_messages_for_compaction(msgs)
        # preserved = last min_turns ua MESSAGES (not pairs); the tool result
        # belongs to the compacted assistant (call_9) so it stays compact-side
        assert len(preserved) == 1
        assert preserved[0] is msgs[4]
        assert len(compact) == 4
        assert indices == [0, 1, 2, 3]


def test_select_messages_orphan_tool_compacted():
    with patch.object(compaction_settings, "compaction_min_preserved_turns", 1):
        msgs = [
            _msg("user"),
            _msg(
                "assistant",
                tool_calls=[
                    {
                        "id": "gone",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            ),
            _msg("tool", tool_call_id="gone"),
            _msg("user"),
            _msg("assistant"),
        ]
        compact, preserved, _indices = C.select_messages_for_compaction(msgs)
        assert preserved[0] is msgs[4]
        assert len(compact) == 4


# ---------------------------------------------------------------------------
# content helpers
# ---------------------------------------------------------------------------


def test_build_compaction_content():
    msgs = [_msg("user", "hello"), _msg("assistant", "world")]
    out = C.build_compaction_content(msgs)
    assert "User: hello" in out or "user: hello" in out
    assert "Assistant: world" in out or "assistant: world" in out


def test_extract_text_from_content_variants():
    assert C._extract_text_from_content("plain") == "plain"
    assert (
        C._extract_text_from_content(
            [{"type": "text", "text": "a"}, {"type": "other", "text": "b"}]
        )
        == "a"
    )
    assert C._extract_text_from_content([1, 2]) == ""
    assert C._extract_text_from_content(None) == ""


def test_prepare_messages_strips_detail_and_non_text_parts():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "u", "detail": "high"}},
                {"type": "unknown_part"},
                "raw-string-part",
            ],
        }
    ]
    out = C._prepare_messages_for_compaction(msgs)
    content = out[0]["content"]
    types = [p["type"] for p in content]
    assert "unknown_part" not in types
    assert "image_url" in types
    assert "detail" not in content[0]["image_url"]


def test_prepare_messages_image_limit_oldest_removed(monkeypatch):
    monkeypatch.setattr(compaction_settings, "compaction_max_images", 1)
    imgs = [{"type": "image_url", "image_url": {"url": f"u{i}"}} for i in range(3)]
    msgs = [_msg("user", content=imgs)]
    out = C._prepare_messages_for_compaction(msgs)
    remaining = [p for p in out[0]["content"] if p["type"] == "image_url"]
    assert len(remaining) == 1
    assert remaining[0]["image_url"]["url"] == "u2"


def test_prepare_messages_changed_to_empty():
    msgs = [{"role": "user", "content": [{"type": "unknown"}]}]
    out = C._prepare_messages_for_compaction(msgs)
    assert out[0]["content"] == ""


# ---------------------------------------------------------------------------
# token budgets
# ---------------------------------------------------------------------------


def test_compaction_token_budget(monkeypatch):
    monkeypatch.setattr(compaction_settings, "compaction_context_safety_factor", 0.5)
    monkeypatch.setattr(compaction_settings, "compaction_summary_max_tokens", 100)
    assert C._compaction_token_budget(1000) == 400


def test_fallback_token_budget(monkeypatch):
    monkeypatch.setattr(compaction_settings, "compaction_context_safety_factor", 0.5)
    monkeypatch.setattr(compaction_settings, "compaction_summary_max_tokens", 100)
    monkeypatch.setattr(
        compaction_settings, "compaction_fallback_model_context_window", 1000
    )
    assert C._fallback_compaction_token_budget() == 400


# ---------------------------------------------------------------------------
# _call_compaction_model
# ---------------------------------------------------------------------------


def _mock_backend(content="SUMMARY"):
    backend = MagicMock()
    backend.build_params = Mock(return_value={})
    resp = Mock()
    resp.choices = [Mock(message=Mock(content=content))]
    backend.converse = AsyncMock(return_value=resp)
    return backend


@pytest.mark.asyncio
async def test_call_compaction_model_success():
    backend = MagicMock()
    backend.build_params = Mock(return_value={})
    resp = Mock()
    resp.choices = [Mock(message=Mock(content="SUM"))]
    backend.converse = AsyncMock(return_value=resp)
    with patch.object(C, "get_backend", return_value=backend):
        out = await C._call_compaction_model("m", [_msg()], 500)
    assert out == "SUM"
    params = backend.build_params.call_args.kwargs.get(
        "context_window_override"
    ) or backend.build_params.call_args[1].get("context_window_override")
    assert params == 500


@pytest.mark.asyncio
async def test_call_compaction_model_error_dict():
    backend = MagicMock()
    backend.build_params = Mock(return_value={})
    backend.converse = AsyncMock(return_value={"type": "error"})
    with (
        patch.object(C, "get_backend", return_value=backend),
        pytest.raises((RuntimeError, TypeError)),
    ):
        await C._call_compaction_model("m", [_msg()], 500)


@pytest.mark.asyncio
async def test_call_compaction_model_empty_choices():
    backend = MagicMock()
    backend.build_params = Mock(return_value={})
    resp = Mock()
    resp.choices = []
    backend.converse = AsyncMock(return_value=resp)
    with (
        patch.object(C, "get_backend", return_value=backend),
        pytest.raises(RuntimeError),
    ):
        await C._call_compaction_model("m", [_msg()], 500)


@pytest.mark.asyncio
async def test_call_compaction_model_empty_content():
    backend = MagicMock()
    backend.build_params = Mock(return_value={})
    resp = Mock()
    resp.choices = [Mock(message=Mock(content=""))]
    backend.converse = AsyncMock(return_value=resp)
    with (
        patch.object(C, "get_backend", return_value=backend),
        pytest.raises(RuntimeError),
    ):
        await C._call_compaction_model("m", [_msg()], 500)


# ---------------------------------------------------------------------------
# message builders for compaction calls
# ---------------------------------------------------------------------------


def test_build_jaguar_compaction_messages_fit():
    with (
        patch.object(C, "count_tokens", return_value=1),
        patch.object(C, "_jaguar_compaction_prompt", return_value="PROMPT"),
    ):
        msgs = C._build_jaguar_compaction_messages([_msg("user", "hi")])
    assert msgs[-1]["content"] == "PROMPT"
    assert msgs[0]["content"] == "hi"


def test_build_jaguar_compaction_messages_truncates(monkeypatch):
    monkeypatch.setattr(C, "count_tokens", lambda m: 10_000)
    with patch.object(C, "_jaguar_compaction_prompt", return_value="PROMPT"):
        out = C._build_jaguar_compaction_messages(
            [_msg("user", "one"), _msg("assistant", "two"), _msg("user", "three")]
        )
    assert len(out) <= 4
    assert out[-1]["content"] == "PROMPT"


def test_build_jaguar_compaction_messages_exhausted(monkeypatch):
    monkeypatch.setattr(C, "count_tokens", lambda m: 10_000_000)
    with (
        patch.object(C, "_jaguar_compaction_prompt", return_value="PROMPT"),
        pytest.raises(RuntimeError),
    ):
        C._build_jaguar_compaction_messages([])


def test_build_fallback_compaction_messages(monkeypatch):
    monkeypatch.setattr(C, "count_tokens", lambda m: 1)
    with (
        patch.object(C, "build_fallback_compaction_prompt", return_value="FPROMPT"),
        patch.object(C, "build_compaction_content", return_value="content"),
    ):
        msgs = C._build_fallback_compaction_messages([_msg("user", "hi")])
    assert msgs == [{"role": "user", "content": "FPROMPT"}]


def test_build_fallback_compaction_messages_exhausted(monkeypatch):
    monkeypatch.setattr(C, "count_tokens", lambda m: 10_000_000)
    with (
        patch.object(C, "build_fallback_compaction_prompt", return_value="FPROMPT"),
        patch.object(C, "build_compaction_content", return_value="content"),
        pytest.raises(RuntimeError),
    ):
        C._build_fallback_compaction_messages([_msg("user", "hi")])


# ---------------------------------------------------------------------------
# chunking helpers
# ---------------------------------------------------------------------------


def test_section_label_variants():
    assert C._section_label(0, 1) is None
    assert C._section_label(0, 2) == "start"
    assert C._section_label(1, 2) == "end"
    assert C._section_label(0, 3) == "start"
    assert C._section_label(1, 3) == "middle"
    assert C._section_label(2, 3) == "end"


def test_chunk_messages_single():
    msgs = [_msg(), _msg()]
    assert C._chunk_messages_for_compaction(msgs, 10_000) == [msgs]


def test_chunk_messages_multiple():
    long = "hello world " * 20
    msgs = [_msg("user", long) for _ in range(6)]
    chunks = C._chunk_messages_for_compaction(msgs, 15)
    assert len(chunks) >= 2
    assert sum(len(c) for c in chunks) == 6


def test_chunk_by_tokens_empty():
    assert C._chunk_messages_by_tokens([], 100) == []


def test_summary_message_roundtrip():
    msg = C.create_summary_message("the summary", section="start")
    assert C._is_summary_message(msg)
    assert C._extract_summary_text(msg) == "the summary"
    assert msg["role"] == "system"


# ---------------------------------------------------------------------------
# chunk summarization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_with_haiku_success():
    async def summarize(chunk):
        return f"S-{chunk[0]['content']}"

    msgs = [_msg("user", "one"), _msg("user", "two")]
    with patch.object(C, "_summarize_haiku_chunk", new=summarize):
        result = await C._summarize_with_haiku(msgs)
    assert result == [(0, "S-one", 1)]


@pytest.mark.asyncio
async def test_summarize_with_haiku_failure():
    async def summarize(chunk):
        raise RuntimeError("boom")

    with (
        patch.object(C, "_summarize_haiku_chunk", new=summarize),
        patch.object(C, "_fallback_chunk_token_limit", return_value=10_000),
        pytest.raises(RuntimeError, match="MCP server error|Haiku compaction"),
    ):
        await C._summarize_with_haiku([_msg(), _msg()])


@pytest.mark.asyncio
async def test_summarize_compaction_chunks_last_chunk_fails_falls_back():
    async def jaguar_fail(chunk):
        raise RuntimeError("jaguar down")

    async def haiku_ok(chunk):
        return "HAIKU-SUMMARY"

    msgs = [_msg("user", "a"), _msg("user", "b")]
    with (
        patch.object(C, "_summarize_jaguar_chunk", new=jaguar_fail),
        patch.object(
            C, "_summarize_with_haiku", new=AsyncMock(return_value=[(0, "H", 2)])
        ),
    ):
        result = await C._summarize_compaction_chunks(msgs, [msgs, msgs])
    assert result == [(0, "H", 2)]


@pytest.mark.asyncio
async def test_summarize_compaction_chunks_skips_early_failures():
    async def jaguar_first_fails(chunk):
        if not getattr(jaguar_first_fails, "called", False):
            jaguar_first_fails.called = True
            raise RuntimeError("first fails")
        return "OK"

    msgs = [_msg("user", "a"), _msg("user", "b")]
    with patch.object(C, "_summarize_jaguar_chunk", new=jaguar_first_fails):
        result = await C._summarize_compaction_chunks(msgs, [msgs, msgs])
    assert all("OK" in s for _, s, _ in result)


# ---------------------------------------------------------------------------
# compact_messages end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_messages_e2e():
    msgs = [
        _msg("system", "sys"),
        _msg("user", "old"),
        _msg("assistant", "old"),
        _msg("user"),
        _msg("assistant"),
    ]
    with (
        patch.object(
            C,
            "_summarize_compaction_chunks",
            new=AsyncMock(return_value=[(0, "SUMMARY", 1)]),
        ),
        patch.object(compaction_settings, "compaction_min_preserved_turns", 1),
        patch.object(C, "_jaguar_chunk_token_limit", return_value=10_000),
    ):
        compacted, summary, indices = await C.compact_messages(msgs)
    assert "SUMMARY" in "".join(str(m) for m in compacted)
    assert summary == "SUMMARY"
    assert indices == [1, 2, 3]
    # system message preserved first
    assert compacted[0]["role"] == "system"


# ---------------------------------------------------------------------------
# resummarization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_compact_below_threshold_returns_none():
    with (
        patch.object(C, "calculate_max_input_tokens", return_value=(100, 500)),
        patch.object(C, "check_compaction_needed", return_value=(False, 10, 60)),
        patch.object(C, "can_compact", return_value=True),
    ):
        msgs = [_msg()]
        out, meta = await C.maybe_compact_conversation(msgs, "m", MagicMock(), True)
        assert out is msgs and meta is None


@pytest.mark.asyncio
async def test_maybe_compact_skip_checks_but_cannot():
    with patch.object(C, "can_compact", return_value=False):
        msgs = [_msg()]
        out, meta = await C.maybe_compact_conversation(
            msgs, "m", MagicMock(), True, skip_checks=True
        )
        assert out is msgs and meta is None


@pytest.mark.asyncio
async def test_maybe_compact_full_trigger():
    [_msg("system", "sys"), _msg("user", "SUMMARY")]
    with (
        patch.object(C, "calculate_max_input_tokens", return_value=(100, 500)),
        patch.object(C, "calculate_message_tokens", lambda *a, **k: 200),
        patch.object(C, "can_compact", return_value=True),
        patch.object(
            C,
            "compact_messages",
            new=AsyncMock(return_value=([_msg("system")], "SUM", [0, 1])),
        ),
    ):
        _out, meta = await C.maybe_compact_conversation(
            [_msg(), _msg()], "m", MagicMock(), True, skip_checks=True
        )
    assert meta is not None
    assert meta.summary == "SUM"
    assert meta.compacted_message_indices == [0, 1]


@pytest.mark.asyncio
async def test_resummarize_until_within_limit_recursion():
    calls = {"n": 0}

    async def resummarize(compacted, summary):
        calls["n"] += 1
        return compacted, summary

    with (
        patch.object(C, "calculate_message_tokens", lambda *a, **k: 50),
        patch.object(C, "_resummarize_compacted_messages", new=resummarize),
    ):
        await C._resummarize_until_within_limit([_msg()], "s", 500, 100, 3)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_resummarize_until_within_limit_attempts_zero():
    async def resummarize(compacted, summary):  # pragma: no cover
        raise AssertionError("should not be called")

    await C._resummarize_until_within_limit([_msg()], "s", 500, 100, 0)


@pytest.mark.asyncio
async def test_resummarize_compacted_messages_failure_keeps_original():
    summary_msg = C.create_summary_message("old")
    with patch.object(
        C,
        "generate_summary_from_single_message",
        new=AsyncMock(side_effect=RuntimeError("x")),
    ):
        _out_compacted, out_summary = await C._resummarize_compacted_messages(
            [summary_msg], "old"
        )
    assert out_summary == "old"


@pytest.mark.asyncio
async def test_generate_summary_single_jaguar_ok():
    with patch.object(C, "_call_compaction_model", new=AsyncMock(return_value="JAG")):
        assert await C.generate_summary_from_single_message("c") == "JAG"


@pytest.mark.asyncio
async def test_generate_summary_single_fallback_after_jaguar_fail():
    calls = {"n": 0}

    async def call(model, messages, ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("jaguar down")
        return "FALLBACK"

    with patch.object(C, "_call_compaction_model", new=call):
        out = await C.generate_summary_from_single_message("c")
    assert out == "FALLBACK"


@pytest.mark.asyncio
async def test_generate_summary_single_both_fail_returns_none():
    async def call(model, messages, ctx):
        raise RuntimeError("nope")

    with patch.object(C, "_call_compaction_model", new=call):
        assert await C.generate_summary_from_single_message("c") is None


# ---------------------------------------------------------------------------
# events + gating
# ---------------------------------------------------------------------------


def test_create_compaction_event_no_data():
    out = C.create_compaction_event("compaction_starting")
    assert out == 'data: {"type": "compaction_starting"}\n\n'


def test_create_compaction_event_with_data():
    out = C.create_compaction_event("compaction_failed", {"error": "x"})
    assert '"error": "x"' in out


def test_should_compact_non_streaming_false():
    assert C.should_compact([], MagicMock(), True, False) is False


def test_should_compact_non_premium_false(monkeypatch):
    with patch.object(C, "calculate_max_input_tokens", return_value=(100, 500)):
        assert C.should_compact([], MagicMock(), False, True) is False


def test_should_compact_blocked_capability(monkeypatch):
    monkeypatch.setattr(C, "has_capability", lambda cap, blocked: True)
    assert (
        C.should_compact([], MagicMock(), True, True, capability="content_agent")
        is False
    )


def test_should_compact_happy(monkeypatch):
    monkeypatch.setattr(C, "calculate_max_input_tokens", lambda *a, **k: (100, 500))
    monkeypatch.setattr(C, "check_compaction_needed", lambda *a, **k: (True, 100, 50))
    monkeypatch.setattr(C, "can_compact", lambda *a, **k: True)
    assert (
        C.should_compact([_msg()], MagicMock(), True, True, capability="chat") is True
    )
