import json

import pytest
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

from aichat.protocol.open_ai_protocol import ToolMessage
from aichat.serve.services import search_stream
from aichat.serve.services.chat_request import parse_chat_request
from aichat.serve.services.search_stream import (
    _hide_search_tool_calls,
    _Round,
    relax_followup_tool_choice,
    stream_with_search,
)

SEARCH_TOOL = {"type": "function", "function": {"name": "brave_web_search"}}
CLIENT_TOOL = {"type": "function", "function": {"name": "get_weather"}}
SEARCH_NAMES = {"brave_web_search"}


def _body(**extra):
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "brave_search_tools": True,
        **extra,
    }


def _tool_call(index, name, arguments, call_id=None):
    return ChatCompletionDeltaToolCall(
        id=call_id or f"call_{index}",
        index=index,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _chunk(tool_calls=None, content=None, finish_reason=None, usage=None):
    chunk = ModelResponseStream(
        id="r",
        choices=[
            StreamingChoices(
                delta=Delta(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
    )
    if usage is not None:
        chunk.usage = usage
    return chunk


def _search_chunk(arguments='{"query": "q"}'):
    return _chunk(
        tool_calls=[_tool_call(0, "brave_web_search", arguments)],
        finish_reason="tool_calls",
    )


class _Stream:
    def __init__(self, *chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


def _patch_search_tools(monkeypatch, outputs=()):
    async def execute(tool_calls, *_args):
        for output in outputs:
            yield None, output
        for tool_call in tool_calls:
            yield (
                None,
                ToolMessage(role="tool", tool_call_id=tool_call.id, content="r"),
            )

    monkeypatch.setattr(search_stream, "execute_tools_and_stream_events", execute)


async def _collect(response, next_round):
    return [
        event
        async for event in stream_with_search(
            response,
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            inline_search=False,
            search_tools=[SEARCH_TOOL],
            mcp_executor=None,
            next_round=next_round,
        )
    ]


def _payloads(events):
    return [
        json.loads(event.removeprefix("data: "))
        for event in events
        if event != "data: [DONE]\n\n"
    ]


def test_search_rejects_multiple_choices():
    with pytest.raises(ValueError, match="require n to be 1"):
        parse_chat_request(_body(n=2))


def test_search_allows_a_single_choice():
    assert parse_chat_request(_body(n=1)).search_tools is True


@pytest.mark.parametrize(
    ("tool_choice", "tools", "expected"),
    [
        (
            {"type": "function", "function": {"name": "brave_web_search"}},
            [CLIENT_TOOL, SEARCH_TOOL],
            "auto",
        ),
        ("required", [SEARCH_TOOL], "auto"),
        ("required", [CLIENT_TOOL, SEARCH_TOOL], "required"),
        (
            {"type": "function", "function": {"name": "get_weather"}},
            [CLIENT_TOOL, SEARCH_TOOL],
            {"type": "function", "function": {"name": "get_weather"}},
        ),
        ("auto", [SEARCH_TOOL], "auto"),
    ],
)
def test_followups_relax_a_forced_search_tool_choice(tool_choice, tools, expected):
    params = {"tool_choice": tool_choice}
    relax_followup_tool_choice(params, tools, [SEARCH_TOOL])
    assert params["tool_choice"] == expected


def test_followups_leave_params_without_tool_choice_untouched():
    params = {}
    relax_followup_tool_choice(params, [SEARCH_TOOL], [SEARCH_TOOL])
    assert params == {}


def test_visible_tool_calls_are_reindexed_after_hiding_search():
    chunk = _chunk(
        tool_calls=[
            _tool_call(0, "brave_web_search", "{}"),
            _tool_call(1, "get_weather", "{}"),
        ]
    )

    assert _hide_search_tool_calls(chunk, _Round(), SEARCH_NAMES)
    tool_calls = chunk.choices[0].delta.tool_calls
    assert [(tc.index, tc.function.name) for tc in tool_calls] == [(0, "get_weather")]


def test_hidden_search_chunk_with_usage_is_kept():
    state = _Round()
    chunk = _search_chunk()
    chunk.usage = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    state.accumulator.process_chunk(chunk.choices[0].delta.tool_calls)

    assert _hide_search_tool_calls(chunk, state, SEARCH_NAMES)
    assert chunk.choices == []
    assert chunk.usage.total_tokens == 3


@pytest.mark.asyncio
async def test_round_limit_ends_with_a_length_finish_reason(monkeypatch):
    _patch_search_tools(monkeypatch)
    rounds = 0

    async def next_round(_messages):
        nonlocal rounds
        rounds += 1
        return _Stream(_search_chunk())

    events = await _collect(_Stream(_search_chunk()), next_round)

    assert rounds == search_stream._MAX_TOOL_ROUNDS
    assert events[-1] == "data: [DONE]\n\n"
    assert _payloads(events)[-1]["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_malformed_search_arguments_end_with_an_error(monkeypatch):
    _patch_search_tools(monkeypatch)

    async def next_round(_messages):
        raise AssertionError("no follow-up expected")

    events = await _collect(_Stream(_search_chunk('{"query":')), next_round)

    assert events[-1] == "data: [DONE]\n\n"
    assert _payloads(events)[-1]["error"]["type"] == "invalid_tool_call"


@pytest.mark.asyncio
async def test_followup_error_ends_with_an_error(monkeypatch):
    _patch_search_tools(monkeypatch)

    async def next_round(_messages):
        return {"type": "error", "content": "Rate limited.", "code": 42900}

    events = await _collect(_Stream(_search_chunk()), next_round)

    assert events[-1] == "data: [DONE]\n\n"
    assert _payloads(events)[-1]["error"] == {
        "message": "Rate limited.",
        "type": "internal_error",
        "code": "internal_error",
    }


@pytest.mark.asyncio
async def test_followup_exception_ends_with_an_error(monkeypatch):
    _patch_search_tools(monkeypatch)

    async def next_round(_messages):
        raise RuntimeError("backend down")

    events = await _collect(_Stream(_search_chunk()), next_round)

    assert events[-1] == "data: [DONE]\n\n"
    assert _payloads(events)[-1]["error"] == {
        "message": "Search stream failed.",
        "type": "internal_error",
        "code": "internal_error",
    }


@pytest.mark.asyncio
async def test_search_output_is_sent_outside_tool_call_deltas(monkeypatch):
    output_content = [{"type": "brave-chat.webSources", "sources": []}]
    _patch_search_tools(
        monkeypatch,
        outputs=[
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "call_0",
                                    "index": 0,
                                    "output_content": output_content,
                                }
                            ]
                        },
                    }
                ]
            }
        ],
    )

    async def next_round(_messages):
        return _Stream(_chunk(content="answer", finish_reason="stop"))

    payloads = _payloads(await _collect(_Stream(_search_chunk()), next_round))

    assert {
        "tool_call_id": "call_0",
        "tool_name": "brave_web_search",
        "output_content": output_content,
        "object": "brave-chat.toolOutput",
    } in payloads
    assert not any(
        choice["delta"].get("tool_calls")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
