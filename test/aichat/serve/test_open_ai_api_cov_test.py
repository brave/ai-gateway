from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from aichat.protocol.open_ai_protocol import Request as OpenAIRequest
from aichat.serve import open_ai_api


def _request(model="test-model", tools=None):
    return OpenAIRequest(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        tools=tools or [],
    )


def _content_chunk(content="hi", finish_reason=None):
    return ChatCompletionChunk(
        id="chatcmpl-1",
        object="chat.completion.chunk",
        created=1700000000,
        model="test-model",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=content, role="assistant"),
                finish_reason=finish_reason,
                logprobs=None,
            )
        ],
    )


def _tool_call_chunk(name="mcp_tool", arguments='{"q":"x"}', tool_id="call_1", index=0):
    return ChatCompletionChunk(
        id="chatcmpl-1",
        object="chat.completion.chunk",
        created=1700000000,
        model="test-model",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(
                    role="assistant",
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=index,
                            id=tool_id,
                            function=ChoiceDeltaToolCallFunction(
                                arguments=name and "{}" or "{}", name=name
                            ),
                            type="function",
                        )
                    ],
                ),
                finish_reason=None,
                logprobs=None,
            )
        ],
    )


def _finish_chunk(reason="tool_calls"):
    return ChatCompletionChunk(
        id="chatcmpl-1",
        object="chat.completion.chunk",
        created=1700000000,
        model="test-model",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(role="assistant"),
                finish_reason=reason,
                logprobs=None,
            )
        ],
    )


def _response(chunks):
    async def gen():
        for c in chunks:
            yield c

    return gen()


def _backend(content="follow-up-ok"):
    backend = MagicMock()
    backend.build_params = Mock(return_value={})
    converse_gen = _response([_content_chunk(content, finish_reason="stop")])

    async def _converse(*args, **kwargs):
        async for c in converse_gen:
            yield c

    backend.converse = AsyncMock(side_effect=_converse)
    return backend


@pytest.mark.asyncio
async def test_process_streaming_response_mcp_server_side_followup():
    """Tool call classified as MCP tool -> execute -> follow-up converse -> DONE (904-1057)."""

    request = _request()
    chunks = [_tool_call_chunk(), _finish_chunk("tool_calls")]
    mcp_executor = MagicMock()
    mcp_executor.is_mcp_tool = AsyncMock(return_value=True)
    backend = _backend()
    prompts_obj = MagicMock()
    prompts_obj.augment = Mock(return_value=[])

    from aichat.protocol.open_ai_protocol import ToolMessage

    async def fake_execute(tool_calls, *args, **kwargs):
        yield None, ToolMessage(role="tool", tool_call_id="call_1", content="ok")

    with patch.object(
        open_ai_api, "execute_tools_and_stream_events", side_effect=fake_execute
    ):
        out = []
        async for item in open_ai_api.process_streaming_response(
            _response(chunks),
            request,
            [],
            mcp_executor=mcp_executor,
            backend=backend,
            prompts=prompts_obj,
        ):
            out.append(item)

    joined = "".join(out)
    assert "follow-up-ok" in joined
    assert joined.rstrip("\n").endswith("data: [DONE]")
    assert backend.converse.await_count == 1


@pytest.mark.asyncio
async def test_process_streaming_response_near_model_passes_api_key():
    """near- model -> params['api_key'] from external_service_settings (near- branch)."""

    request = _request(model="near-foo")
    chunks = [_tool_call_chunk(), _finish_chunk("tool_calls")]
    mcp_executor = MagicMock()
    mcp_executor.is_mcp_tool = AsyncMock(return_value=True)
    backend = _backend()

    from aichat.protocol.open_ai_protocol import ToolMessage

    async def fake_execute(*args, **kwargs):
        yield None, ToolMessage(role="tool", tool_call_id="call_1", content="ok")

    ext = MagicMock()
    ext.near_api_key = "NEARKEY"
    with (
        patch.object(
            open_ai_api, "execute_tools_and_stream_events", side_effect=fake_execute
        ),
        patch.object(open_ai_api, "external_service_settings", ext, create=True),
    ):
        out = []
        async for item in open_ai_api.process_streaming_response(
            _response(chunks),
            request,
            [],
            mcp_executor=mcp_executor,
            backend=backend,
            prompts=MagicMock(augment=Mock(return_value=[])),
        ):
            out.append(item)

    # converse received params carrying near api key
    assert backend.converse.await_count == 1
    kwargs = backend.converse.await_args
    params = kwargs.args[2] if len(kwargs.args) > 2 else kwargs.kwargs.get("params")
    assert params.get("api_key") == "NEARKEY"


@pytest.mark.asyncio
async def test_process_streaming_response_client_tool_calls_stops():
    """Tool call not MCP -> client tool call path: DONE immediately (1064-1111)."""

    request = _request()
    chunks = [_tool_call_chunk(name="client_tool"), _finish_chunk("tool_calls")]
    mcp_executor = MagicMock()
    mcp_executor.is_mcp_tool = AsyncMock(return_value=False)
    backend = _backend()

    out = []
    async for item in open_ai_api.process_streaming_response(
        _response(chunks),
        request,
        [],
        mcp_executor=mcp_executor,
        backend=backend,
        prompts=MagicMock(augment=Mock(return_value=[])),
    ):
        out.append(item)

    joined = "".join(out)
    assert "data: [DONE]" in joined
    assert backend.converse.await_count == 0


@pytest.mark.asyncio
async def test_stream_with_compaction_happy_path_near_api_key():
    """_stream_with_compaction 457-523: events + near- api_key branch."""

    request = _request(model="near-foo")
    model_config = MagicMock()
    backend = _backend(content="compacted-follow-up")
    ext = MagicMock()
    ext.near_api_key = "NEARKEY"

    async def fake_execute(*args, **kwargs):
        return [], [], MagicMock()

    with (
        patch.object(
            open_ai_api,
            "maybe_compact_conversation",
            new=AsyncMock(return_value=([], None)),
        ) as mc,
        patch.object(
            open_ai_api,
            "prepare_tools_for_request",
            new=AsyncMock(return_value=([], [], MagicMock())),
        ),
        patch.object(open_ai_api, "augment_messages", new=AsyncMock(return_value=[])),
        patch.object(
            open_ai_api, "simplify_messages_for_llm", new=Mock(side_effect=lambda m: m)
        ),
        patch.object(open_ai_api, "external_service_settings", ext, create=True),
    ):
        out = []
        async for item in open_ai_api._stream_with_compaction(
            request,
            [{"role": "user", "content": "Hello"}],
            model_config,
            True,
            backend,
            request_start_time=1.0,
        ):
            out.append(item)

    joined = "".join(out)
    assert "compaction_starting" in joined
    assert "follow-up-ok" in joined or "compacted-follow-up" in joined
    assert "data: [DONE]" in joined
    mc.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_with_compaction_failure_emits_failed_event():
    """_stream_with_compaction exception path -> compaction_failed + DONE."""

    request = _request()
    model_config = MagicMock()
    backend = MagicMock()
    backend.build_params = Mock(side_effect=RuntimeError("boom"))

    with patch.object(
        open_ai_api,
        "maybe_compact_conversation",
        new=AsyncMock(return_value=([], None)),
    ):
        out = []
        async for item in open_ai_api._stream_with_compaction(
            request, [{"role": "user", "content": "Hello"}], model_config, True, backend
        ):
            out.append(item)

    joined = "".join(out)
    assert "compaction_failed" in joined
    assert "data: [DONE]" in joined
