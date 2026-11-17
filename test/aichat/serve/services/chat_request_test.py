from unittest.mock import MagicMock, patch

import pytest

from aichat.serve.services.chat_request import (
    build_backend_params,
    parse_chat_request,
)

EPHEMERAL = {"type": "ephemeral"}


def _body(messages, tools=None, **extra):
    body = {"model": "test-model", "messages": messages, **extra}
    if tools is not None:
        body["tools"] = tools
    return body


def _tool(name, **extra):
    return {"type": "function", "function": {"name": name}, **extra}


def _backend(uses_caching=True):
    backend = MagicMock()
    backend.config.upstream_model = "model"
    backend.uses_prompt_caching.return_value = uses_caching
    backend.build_params.return_value = {}
    return backend


def _build(body, messages):
    backend = _backend()
    with patch(
        "aichat.serve.services.chat_request.apply_claude_upstream_sampling_params"
    ):
        build_backend_params(backend, parse_chat_request(body), messages)
    return backend.build_params.call_args.kwargs


class TestParseCacheMarks:
    def test_finds_message_part_and_tool_marks(self):
        request = parse_chat_request(
            _body(
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "s", "cache_control": EPHEMERAL}
                        ],
                    },
                    {"role": "user", "content": "a"},
                    {"role": "user", "content": "b", "cache_control": EPHEMERAL},
                ],
                tools=[
                    _tool("a"),
                    _tool("b", cache_control=EPHEMERAL),
                    {
                        "type": "function",
                        "function": {"name": "c", "cache_control": EPHEMERAL},
                    },
                ],
            )
        )
        assert request.cached_message_indices == {0, 2}
        assert request.cached_tool_indices == {1, 2}
        assert request.has_cache_marks

    @pytest.mark.parametrize(
        "mark", [None, {"type": "persistent"}, "ephemeral", {"ttl": "1h"}]
    )
    def test_ignores_invalid_marks(self, mark):
        request = parse_chat_request(
            _body([{"role": "user", "content": "a", "cache_control": mark}])
        )
        assert not request.has_cache_marks

    def test_no_marks(self):
        request = parse_chat_request(_body([{"role": "user", "content": "a"}]))
        assert not request.has_cache_marks


class TestBuildBackendParamsCacheMarks:
    def test_no_marks_leaves_server_checkpoints(self):
        messages = [{"role": "user", "content": "a"}]
        kwargs = _build(_body([{"role": "user", "content": "a"}]), messages)
        assert "client_cache_marks" not in kwargs
        assert messages == [{"role": "user", "content": "a"}]

    def test_marks_last_part_and_strips_ttl(self):
        raw = [
            {"role": "system", "content": "s"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a", "cache_control": EPHEMERAL},
                    {"type": "text", "text": "b"},
                ],
            },
            {
                "role": "user",
                "content": "c",
                "cache_control": {**EPHEMERAL, "ttl": "1h"},
            },
        ]
        messages = [
            {"role": "system", "content": "s"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            },
            {"role": "user", "content": "c"},
        ]
        kwargs = _build(_body(raw), messages)
        assert kwargs["client_cache_marks"] is True
        assert messages[0] == {"role": "system", "content": "s"}
        assert messages[1]["content"] == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b", "cache_control": EPHEMERAL},
        ]
        assert messages[2]["content"] == [
            {"type": "text", "text": "c", "cache_control": EPHEMERAL}
        ]

    def test_marks_tools(self):
        body = _body(
            [{"role": "user", "content": "a"}],
            tools=[_tool("a"), _tool("b", cache_control={**EPHEMERAL, "ttl": "1h"})],
        )
        kwargs = _build(body, [{"role": "user", "content": "a"}])
        assert kwargs["client_cache_marks"] is True
        assert "cache_control" not in kwargs["tools"][0]
        assert kwargs["tools"][1]["cache_control"] == EPHEMERAL

    def test_caps_marks_keeping_tools_and_latest_messages(self):
        raw = [
            {"role": "user", "content": str(i), "cache_control": EPHEMERAL}
            for i in range(5)
        ]
        messages = [{"role": "user", "content": str(i)} for i in range(5)]
        body = _body(raw, tools=[_tool("a", cache_control=EPHEMERAL)])
        kwargs = _build(body, messages)
        assert kwargs["tools"][0]["cache_control"] == EPHEMERAL
        marked = [i for i, m in enumerate(messages) if isinstance(m["content"], list)]
        assert marked == [2, 3, 4]

    def test_skips_message_marks_when_message_count_changed(self):
        raw = [
            {"role": "user", "content": "a", "cache_control": EPHEMERAL},
            {"role": "user", "content": "b"},
        ]
        messages = [{"role": "user", "content": "a"}]
        kwargs = _build(_body(raw), messages)
        assert kwargs["client_cache_marks"] is False
        assert messages == [{"role": "user", "content": "a"}]

    def test_drops_marks_when_caching_unavailable(self):
        raw = [{"role": "user", "content": "a", "cache_control": EPHEMERAL}]
        messages = [{"role": "user", "content": "a"}]
        backend = _backend(uses_caching=False)
        body = _body(raw, tools=[_tool("a", cache_control=EPHEMERAL)])
        with patch(
            "aichat.serve.services.chat_request.apply_claude_upstream_sampling_params"
        ):
            build_backend_params(backend, parse_chat_request(body), messages)
        kwargs = backend.build_params.call_args.kwargs
        assert "client_cache_marks" not in kwargs
        assert "cache_control" not in kwargs["tools"][0]
        assert messages == [{"role": "user", "content": "a"}]
