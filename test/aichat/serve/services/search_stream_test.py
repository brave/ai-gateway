import json

import pytest
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)

from aichat.serve.services import search_stream
from aichat.serve.services.chat_request import parse_chat_request
from aichat.serve.services.search_stream import (
    relax_followup_tool_choice,
    stream_with_search,
)

SEARCH_TOOL = {"type": "function", "function": {"name": "brave_web_search"}}
CLIENT_TOOL = {"type": "function", "function": {"name": "get_weather"}}


def _body(**extra):
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "brave_search_tools": True,
        **extra,
    }


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


class _SearchCallStream:
    def __init__(self):
        self._chunks = iter(
            [
                ModelResponseStream(
                    id="r",
                    choices=[
                        StreamingChoices(
                            delta=Delta(
                                tool_calls=[
                                    ChatCompletionDeltaToolCall(
                                        id="call_1",
                                        index=0,
                                        type="function",
                                        function=Function(
                                            name="brave_web_search",
                                            arguments='{"query": "q"}',
                                        ),
                                    )
                                ]
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
            ]
        )

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


@pytest.mark.asyncio
async def test_round_limit_ends_with_a_length_finish_reason(monkeypatch):
    async def run_mcp_tool_round(*args):
        args[-1].append({"role": "tool", "tool_call_id": "call_1", "content": "r"})
        if False:
            yield ""

    monkeypatch.setattr(search_stream, "run_mcp_tool_round", run_mcp_tool_round)
    rounds = 0

    async def next_round(_messages):
        nonlocal rounds
        rounds += 1
        return _SearchCallStream()

    events = [
        event
        async for event in stream_with_search(
            _SearchCallStream(),
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            inline_search=False,
            search_tools=[SEARCH_TOOL],
            mcp_executor=None,
            next_round=next_round,
        )
    ]

    assert rounds == search_stream._MAX_TOOL_ROUNDS
    assert events[-1] == "data: [DONE]\n\n"
    final = json.loads(events[-2].removeprefix("data: "))
    assert final["choices"][0]["finish_reason"] == "length"
