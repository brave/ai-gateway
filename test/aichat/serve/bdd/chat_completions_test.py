import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import BackgroundTasks, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.protocol.open_ai_protocol import Request as OpenAIRequest
from aichat.protocol.open_ai_protocol import Tool, ToolFunction, UserMessage
from aichat.responses import CompactionMetadata
from aichat.serve import open_ai_api
from aichat.serve.constants import CONTENT_FILTER_MESSAGE
from aichat.serve.conversation_settings import conversation_settings
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.server_settings import server_settings
from aichat.serve.services.compaction_settings import compaction_settings
from aichat.serve.services.security_settings import security_settings
from test.aichat.serve.bdd.helpers import (
    chunk,
    client_tool_call_chunk,
    collect_streaming_response,
    common_params,
    content_chunk,
    finish_tool_calls_chunk,
    make_backend,
    mock_request,
    run_stream_pipeline,
    sse_data_lines,
    stub_pipeline,
    usage_chunk,
)
from test.aichat.serve.bdd.helpers import (
    mock_model_config as _model_config,
)

FEATURE = Path(__file__).parent / "features" / "chat_completions.feature"
scenarios(str(FEATURE))


@pytest.fixture
def ctx():
    return {}


@given(parsers.parse('a fake backend for model "{model}" replying "{reply}"'))
def given_backend_reply(model, reply, ctx):
    ctx["model"] = model
    ctx["backend"] = make_backend(model, reply=reply)


@given(
    parsers.parse(
        'a fake backend for model "{model}" replying with model "{reply_model}"'
    )
)
def given_backend_reply_model(model, reply_model, ctx):
    ctx["model"] = model
    import litellm

    response = litellm.ModelResponse(
        id="resp-1",
        object="chat.completion",
        created=1,
        model=reply_model,
        choices=[
            litellm.Choices(
                index=0,
                message=litellm.Message(role="assistant", content="ok"),
                finish_reason="stop",
            )
        ],
    )
    backend = MagicMock()
    backend.build_params.return_value = {"temperature": 0.7}
    backend.converse = AsyncMock(return_value=response)
    ctx["backend"] = backend
    ctx["object_response"] = True


@given(
    parsers.parse(
        'a fake backend for model "{model}" streaming a chunk with content "{content}"'
    )
)
def given_backend_stream(model, content, ctx):
    ctx["model"] = model
    ctx["backend"] = make_backend(
        model, stream_chunks=[content_chunk(content, model=model)]
    )


@given(parsers.parse('a fake backend for model "{model}" failing with code {code:d}'))
def given_backend_failure(model, code, ctx):
    ctx["model"] = model
    backend = MagicMock()
    backend.build_params.return_value = {}
    backend.converse = AsyncMock(
        return_value={"type": "error", "code": code, "content": "backend down"}
    )
    ctx["backend"] = backend


@given("the model config does not support tools")
def given_no_tool_support(ctx):
    ctx["no_tool_support"] = True


@given("NEAR ohttp verification returns a config")
def given_near_verified(ctx, monkeypatch):
    monkeypatch.setattr(
        open_ai_api,
        "verify_and_get_ohttp_config",
        AsyncMock(return_value={"keys": ["k"]}),
    )
    ctx["near_verified"] = True


@given('the NEAR API key is "near-secret"')
def given_near_key_sentinel(ctx, monkeypatch):
    # Sentinel so the api_key assertion cannot vacuously pass on None==None.
    monkeypatch.setattr(external_service_settings, "near_api_key", "near-secret")


@given("a non-streaming chat request carrying a brave-conversation-title part")
def given_title_request(ctx, monkeypatch):
    title_response = JSONResponse({"ok": True})
    complete_title = AsyncMock(return_value=title_response)
    monkeypatch.setattr(open_ai_api, "complete_conversation_title_chat", complete_title)
    ctx["complete_title"] = complete_title
    ctx["title_response"] = title_response
    ctx["request"] = SimpleNamespace(
        model="test-model",
        messages=[
            SimpleNamespace(
                role="user",
                content=[{"type": "brave-conversation-title", "text": "Title me"}],
            )
        ],
        stream=False,
        tools=None,
        brave_capability=None,
        brave_mcp_tools_exclude=None,
        brave_mcp_tools_include=None,
    )


@given(parsers.parse("the conversation rounds maximum is {maximum:d}"))
def given_rounds_maximum(maximum, monkeypatch):
    monkeypatch.setattr(conversation_settings, "max_conversation_rounds", maximum)


@given(parsers.parse("the absolute token ceiling is {ceiling:d}"))
def given_absolute_ceiling(ceiling, monkeypatch):
    monkeypatch.setattr(compaction_settings, "absolute_max_tokens", ceiling)


@given(parsers.parse("token trimming reports {tokens:d} tokens"))
def given_trimmed_tokens(tokens, monkeypatch):
    monkeypatch.setattr(
        open_ai_api,
        "maybe_trim_messages",
        lambda messages, **kwargs: (messages, tokens, 0),
    )


@given("compaction will be triggered")
def given_compaction_triggered(monkeypatch):
    monkeypatch.setattr(open_ai_api, "should_compact", MagicMock(return_value=True))


@given(
    parsers.parse(
        'compaction produces metadata for messages "{compacted_messages_json}"'
    )
)
def given_compaction_metadata(compacted_messages_json, monkeypatch):
    compacted = json.loads(compacted_messages_json)
    metadata = CompactionMetadata(
        summary="summarized history",
        compacted_message_indices=[0],
        tokens_before=100,
        tokens_after=10,
    )
    monkeypatch.setattr(
        open_ai_api,
        "maybe_compact_conversation",
        AsyncMock(return_value=(compacted, metadata)),
    )


@given("compaction blows up")
def given_compaction_failure(monkeypatch):
    monkeypatch.setattr(
        open_ai_api,
        "maybe_compact_conversation",
        AsyncMock(side_effect=RuntimeError("compaction exploded")),
    )


@given("an active model override without premium fallback")
def given_model_override(ctx):
    ctx["model_override"] = True


@given(
    parsers.parse(
        'a non-streaming chat request for model "{model}" with {count:d} user messages'
    )
)
def given_request_many_messages(model, count, ctx):
    ctx["model"] = model
    ctx["request"] = OpenAIRequest(
        model=model,
        messages=[UserMessage(content=f"message {i}") for i in range(count)],
        stream=False,
        tools=None,
    )


@given(parsers.parse('a non-streaming chat request for model "{model}"'))
def given_non_streaming_request(model, ctx):
    ctx["model"] = model
    ctx["request"] = OpenAIRequest(
        model=model,
        messages=[UserMessage(content="Hello, how are you?")],
        stream=False,
        tools=None,
    )


@given(
    parsers.parse(
        'a non-streaming chat request for model "{model}" with a get_weather tool'
    )
)
def given_request_with_tool(model, ctx):
    ctx["model"] = model
    ctx["request"] = OpenAIRequest(
        model=model,
        messages=[UserMessage(content="What's the weather like?")],
        stream=False,
        tools=[
            Tool(
                type="function",
                function=ToolFunction(
                    name="get_weather",
                    description="Get current weather",
                    parameters={"type": "object", "properties": {}},
                ),
            )
        ],
    )


@given(parsers.parse('a streaming chat request for model "{model}"'))
def given_streaming_request(model, ctx):
    ctx["model"] = model
    ctx["request"] = OpenAIRequest(
        model=model,
        messages=[UserMessage(content="Hello, how are you?")],
        stream=True,
        tools=None,
    )
    if "backend" not in ctx:
        # Streaming requests must not silently fall back to a dict backend;
        # give them a real streaming converse unless a Given already stubbed one.
        ctx["backend"] = make_backend(model, stream_chunks=[content_chunk("hi", model)])


@when("the chat completion is processed")
def when_processed(ctx, monkeypatch):
    model = ctx["model"]
    model_config = ctx.get("model_config") or _model_config(
        model_id=model, tool_support=not ctx.get("no_tool_support", False)
    )
    get_backend = stub_pipeline(monkeypatch, ctx, model_config)
    get_backend.return_value = ctx["backend"]

    select_model = AsyncMock(return_value=model)
    monkeypatch.setattr(open_ai_api, "select_model_for_request", select_model)
    ctx["select_model"] = select_model

    monkeypatch.setattr(security_settings, "alignment_checking_enabled", False)

    raw_request = mock_request(model_override=ctx.get("model_override", False))
    background_tasks = MagicMock(spec=BackgroundTasks)
    fastapi_response = MagicMock(spec=Response)
    fastapi_response.headers = {}
    common = common_params(model)

    try:
        ctx["result"] = asyncio.run(
            open_ai_api.v1_chat_completions(
                raw_request=raw_request,
                request=ctx["request"],
                background_tasks=background_tasks,
                fastapi_response=fastapi_response,
                common=common,
            )
        )
        ctx["error"] = None
    except Exception as exc:
        ctx["error"] = exc
    ctx["raw_request"] = raw_request
    ctx["fastapi_response"] = fastapi_response


@when(
    parsers.parse(
        "the stream pipeline processes a usage chunk for {total_tokens:d} total tokens"
    )
)
def when_usage_chunk(total_tokens, ctx, monkeypatch):
    ctx["events"] = run_stream_pipeline(
        ctx, monkeypatch, chunks=[usage_chunk(total_tokens)]
    )


@when("the stream pipeline processes a chunk stopped by content_filter")
def when_content_filter(ctx, monkeypatch):
    ctx["events"] = run_stream_pipeline(
        ctx,
        monkeypatch,
        chunks=[chunk(content=None, finish_reason="content_filter")],
    )


@when("the stream pipeline processes a completed client tool call")
def when_client_tool_call(ctx, monkeypatch):
    executor = MagicMock()
    executor.is_mcp_tool = AsyncMock(return_value=False)
    ctx["events"] = run_stream_pipeline(
        ctx,
        monkeypatch,
        chunks=[client_tool_call_chunk(), finish_tool_calls_chunk()],
        backend=AsyncMock(),
        mcp_executor=executor,
    )


@when(
    parsers.parse(
        'the stream pipeline processes a completed MCP tool call with result "{result}"'
    )
)
def when_mcp_tool_call(result, ctx, monkeypatch):
    ctx["tool_result"] = result
    backend = MagicMock()
    backend.build_params.return_value = {}

    async def followup_gen():
        yield content_chunk("follow-up answer")

    backend.converse = AsyncMock(return_value=followup_gen())

    executor = MagicMock()
    executor.is_mcp_tool = AsyncMock(return_value=True)

    from aichat.protocol.open_ai_protocol import ToolMessage

    async def tool_events(*args, **kwargs):
        yield (None, ToolMessage(role="tool", tool_call_id="call_1", content=result))

    monkeypatch.setattr(open_ai_api, "execute_tools_and_stream_events", tool_events)
    ctx["events"] = run_stream_pipeline(
        ctx,
        monkeypatch,
        chunks=[client_tool_call_chunk(), finish_tool_calls_chunk()],
        backend=backend,
        mcp_executor=executor,
    )
    ctx["backend"] = backend


@then(parsers.parse('the JSON completion contains "{content}"'))
def then_json_contains(content, ctx):
    result = ctx["result"]
    assert result["choices"][0]["message"]["content"] == content


@then("the result is a streaming SSE response")
def then_streaming_sse(ctx):
    assert isinstance(ctx["result"], StreamingResponse)
    assert ctx["result"].media_type == "text/event-stream"


@then(parsers.parse("an HTTP error with status {status:d} is raised"))
def then_http_error(status, ctx):
    assert ctx["error"] is not None
    assert ctx["error"].status_code == status


@then("the backend received no tools")
def then_no_tools(ctx):
    backend = ctx["backend"]
    args, _ = backend.build_params.call_args
    tools_arg = args[1]
    assert tools_arg == []


@then("the backend params include the NEAR API key")
def then_near_api_key(ctx):
    backend = ctx["backend"]
    args, _ = backend.converse.await_args
    params = args[2]
    assert params["api_key"] == external_service_settings.near_api_key


@then(parsers.parse('the stream headers contain "{header}" set to "{value}"'))
def then_stream_header(header, value, ctx):
    assert ctx["result"].headers[header] == value


@then("the title service answered and model selection never ran")
def then_title_bypassed(ctx):
    ctx["complete_title"].assert_awaited_once()
    ctx["select_model"].assert_not_awaited()


@then(parsers.parse("a JSON error response with status {status:d} is returned"))
def then_json_error(status, ctx):
    assert ctx["result"].status_code == status


@then("the first stream event mentions compaction_starting")
def then_first_event_compaction(ctx):
    events = collect_streaming_response(ctx, ctx["result"])
    assert "compaction_starting" in events[0]


@then("the stream mentions compaction_failed")
def then_compaction_failed(ctx):
    events = collect_streaming_response(ctx, ctx["result"])
    assert any("compaction_failed" in e for e in events)


@then("the stream ends with a DONE sentinel")
def then_done_sentinel(ctx):
    events = collect_streaming_response(ctx, ctx["result"])
    assert events[-1].strip() == "data: [DONE]"


@then("the JSON completion model equals the placeholder model")
def then_placeholder_model(ctx):
    result = ctx["result"]
    model = result.model if hasattr(result, "model") else result["model"]
    assert model == server_settings.placeholder_model


@then(
    parsers.parse(
        "a content receipt event for {total_tokens:d} total tokens is emitted"
    )
)
def then_content_receipt(total_tokens, ctx):
    receipts = [
        line for line in sse_data_lines(ctx["events"]) if "contentReceipt" in line
    ]
    assert receipts, ctx["events"]
    payload = json.loads(receipts[-1])
    assert payload["total_tokens"] == total_tokens


@then(parsers.parse("a usage chunk for {total_tokens:d} total tokens is emitted"))
def then_usage_chunk(total_tokens, ctx):
    usage = [line for line in sse_data_lines(ctx["events"]) if '"usage"' in line]
    assert usage, ctx["events"]
    payload = json.loads(usage[-1])
    assert payload["usage"]["total_tokens"] == total_tokens


@then("the accumulated content is the content filter message")
def then_content_filter_message(ctx):
    assert any(CONTENT_FILTER_MESSAGE in e for e in ctx["events"])


@then("the finish reason is preserved as content_filter")
def then_finish_reason_preserved(ctx):
    chunks = [
        line for line in sse_data_lines(ctx["events"]) if '"content_filter"' in line
    ]
    assert chunks, ctx["events"]


@then("the tool call is forwarded to the client")
def then_tool_call_forwarded(ctx):
    forwarded = [
        line for line in sse_data_lines(ctx["events"]) if '"tool_calls"' in line
    ]
    assert forwarded, ctx["events"]


@then("no follow-up conversation is executed")
def then_no_followup(ctx):
    ctx["pipeline_backend"].converse.assert_not_awaited()
