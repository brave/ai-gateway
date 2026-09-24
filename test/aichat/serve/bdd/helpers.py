"""Shared BDD helpers.

Non-test module: bindings import builders from here instead of from each
other, keeping pytest-bdd step definitions and cross-test imports out of
individual test files.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from aichat.serve import open_ai_api
from aichat.serve.services.models import ModelConfig
from aichat.serve.services.security_settings import security_settings


def mock_model_config(**overrides):
    """MagicMock-style model config for pipeline-level scenarios."""
    cfg = MagicMock()
    cfg.model_id = overrides.get("model_id", "test-model")
    cfg.backend = "litellm"
    cfg.tool_support = overrides.get("tool_support", True)
    cfg.file_support = True
    cfg.image_support = True
    cfg.conversation_token_limit = None
    cfg.conversation_token_limit_premium = None
    cfg.prompt_caching_support = True
    cfg.content_agent_support = False
    cfg.truncation_continuation = False
    return cfg


def dataclass_model_config(**overrides):
    """Real ModelConfig dataclass for code paths that read typed fields."""
    return ModelConfig(
        model_id=overrides.get("model_id", "test-model"),
        upstream_model=overrides.get("upstream_model", "up/test-model"),
        backend=overrides.get("backend", "litellm"),
        api_base=None,
        api_key=None,
        inference_profile=None,
        system_prompt_support=True,
        prompt_caching_support=False,
        prompt_caching_enabled=False,
        tool_support=overrides.get("tool_support", True),
        image_support=overrides.get("image_support", True),
        audio_support=True,
        video_support=True,
        file_support=overrides.get("file_support", True),
        friendly_name="Test Model",
        maker="Test",
        max_tokens=1000,
        max_tokens_premium=None,
        conversation_token_limit=None,
        conversation_token_limit_premium=None,
        free=True,
        key=None,
        rate_limit=None,
        rate_limit_interval_seconds=None,
        max_pages=None,
        max_pages_premium=None,
        extra_body=None,
        truncation_continuation=overrides.get("truncation_continuation", False),
    )


def chunk(
    content=None,
    finish_reason=None,
    tool_calls=None,
    usage=None,
    model="test-model",
):
    delta_kwargs = {}
    if content is not None:
        delta_kwargs["content"] = content
    if tool_calls:
        delta_kwargs["tool_calls"] = tool_calls
    return ChatCompletionChunk(
        id="log-1",
        object="chat.completion.chunk",
        created=1,
        model=model,
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(**delta_kwargs),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def content_chunk(content, model="test-model"):
    return chunk(content=content, finish_reason=None, model=model)


def client_tool_call_chunk():
    tool_call = ChoiceDeltaToolCall(
        index=0,
        id="call_1",
        type="function",
        function=ChoiceDeltaToolCallFunction(
            name="get_weather", arguments='{"city": "SF"}'
        ),
    )
    return chunk(tool_calls=[tool_call], finish_reason=None)


def finish_tool_calls_chunk():
    return chunk(finish_reason="tool_calls")


def usage_chunk(total_tokens):
    return ChatCompletionChunk(
        id="log-1",
        object="chat.completion.chunk",
        created=1,
        # Upstream chunks may carry a different model string ("test model")
        # than the requested one ("test-model"); the pipeline must not rewrite
        # usage-only chunks, so the space is deliberate.
        model="test model",
        choices=[],
        usage=CompletionUsage(
            prompt_tokens=10, completion_tokens=32, total_tokens=total_tokens
        ),
    )


def mock_request(model_override=False):
    request = MagicMock(spec=Request)
    # url must expose .path (and stay falsy-safe) like a real starlette URL.
    request.url = SimpleNamespace(path="/v1/chat/completions", host="test.com")
    request.headers = {"authorization": "Bearer test-token"}
    request.state.model_override = model_override
    request.state.model_override_premium_fallback = None
    request.state.httpx_client = None
    return request


def common_params(model):
    return {
        "model": model,
        "is_premium_host": False,
        "has_valid_premium_credential": False,
        "x_forwarded_for": None,
        "api_version": 2,
    }


def make_backend(model, reply=None, stream_chunks=None):
    backend = MagicMock()
    backend.build_params.return_value = {"temperature": 0.7}
    if stream_chunks is not None:

        async def stream_gen():
            for c in stream_chunks:
                yield c

        # side_effect (not return_value): each converse call must receive a
        # FRESH generator — retries/follow-ups would otherwise consume an
        # already-exhausted stream.
        backend.converse = AsyncMock(side_effect=lambda *a, **k: stream_gen())
    else:
        backend.converse = AsyncMock(
            return_value={
                "id": "resp-1",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
    return backend


def stub_pipeline(monkeypatch, ctx, model_config):
    monkeypatch.setattr(
        open_ai_api, "check_requests_common", AsyncMock(return_value=None)
    )

    prompts_stub = SimpleNamespace(prompts=[])
    monkeypatch.setattr(open_ai_api, "prompts", prompts_stub)

    monkeypatch.setattr(open_ai_api.mcp_settings, "mcp_enabled", False)

    get_backend = MagicMock()
    get_model_config = MagicMock(return_value=model_config)
    monkeypatch.setattr(open_ai_api, "get_backend", get_backend)
    monkeypatch.setattr(open_ai_api, "get_model_config", get_model_config)
    ctx["get_backend"] = get_backend
    ctx["model_config"] = model_config
    if "backend" not in ctx:
        ctx["backend"] = make_backend(ctx["model"], reply="ok")
    return get_backend


def run_stream_pipeline(
    ctx,
    monkeypatch,
    chunks,
    backend=None,
    mcp_executor=None,
    requested_model=None,
    stream_factory=None,
):
    monkeypatch.setattr(security_settings, "alignment_checking_enabled", False)
    backend = backend or MagicMock()

    if stream_factory is not None:

        async def response_gen():
            async for c in stream_factory():
                yield c

    else:

        async def response_gen():
            for c in chunks:
                yield c

    request = ctx["request"]
    model_config = mock_model_config(model_id=request.model)
    prompts_stub = SimpleNamespace(prompts=[])

    async def collect():
        events = []
        async for event in open_ai_api.process_streaming_response(
            response_gen(),
            request,
            [],
            mcp_executor,
            backend,
            model_config,
            prompts_stub,
            None,
            requested_model=requested_model,
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())
    ctx["pipeline_backend"] = backend
    return events


def collect_streaming_response(ctx, response):
    cached = ctx.get("stream_events")
    if cached is not None:
        return cached

    async def collect():
        return [event async for event in response.body_iterator]

    events = asyncio.run(collect())
    ctx["stream_events"] = events
    return events


def sse_data_lines(events):
    lines = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: "):
                lines.append(line[len("data: ") :])
    return lines


def new_patch(ctx):
    """Manual MonkeyPatch registered on ctx["undo"] (LIFO-undone in teardown)."""
    mp = pytest.MonkeyPatch()
    ctx["undo"].append(mp)
    return mp


def completion_response(content):
    """Minimal non-streaming completion: choices[0].message.content."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
