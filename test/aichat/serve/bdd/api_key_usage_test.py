import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage
from prometheus_client import REGISTRY
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import api_key_chat_api
from aichat.serve.services import api_key_usage

FEATURE = "features/api_key_usage.feature"

scenarios(FEATURE)

API_KEY = "sk_test_k1_s1_abcdefgh"
MODEL = "test-model"
_OUTCOMES = ("settled", "missing_usage", "unavailable")


def _settle_count(outcome: str) -> float:
    return (
        REGISTRY.get_sample_value("api_key_usage_settle_total", {"outcome": outcome})
        or 0.0
    )


@pytest.fixture
def ctx(monkeypatch):
    state = {
        "settles": [],
        "settle_result": {},
        "body_extra": {},
        "backend": "bedrock",
        "api_base": None,
        "usage": None,
        "metrics_before": {outcome: _settle_count(outcome) for outcome in _OUTCOMES},
    }

    async def usage_settle(_client, **kwargs):
        state["settles"].append(kwargs)
        return state["settle_result"]

    monkeypatch.setattr(api_key_usage.internal_client, "usage_settle", usage_settle)
    return state


class _UpstreamStream:
    def __init__(self, usage, gate: asyncio.Event | None = None):
        self._chunks = iter(_upstream_chunks(usage))
        self._gate = gate
        self._sent = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        if self._gate is not None and self._sent == 1:
            await self._gate.wait()
        self._sent += 1
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


def _upstream_chunks(usage) -> list[ModelResponseStream]:
    chunks = [
        ModelResponseStream(
            id="c", choices=[StreamingChoices(delta=Delta(content=f"t{i}"))]
        )
        for i in range(3)
    ]
    if usage is not None:
        final = ModelResponseStream(id="c", choices=[])
        final.usage = usage
        chunks.append(final)
    return chunks


def _request(body: dict) -> MagicMock:
    request = MagicMock(spec=Request)
    request.json = AsyncMock(return_value=body)
    request.headers = {}
    request.state = MagicMock()
    return request


async def _wait_for_background_tasks() -> None:
    while api_key_usage._background_tasks:
        await asyncio.gather(*list(api_key_usage._background_tasks))
        await asyncio.sleep(0)


async def _handle(ctx, *, stream: bool, disconnect: str | None = None) -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        **ctx["body_extra"],
    }
    if stream:
        body["stream"] = True

    backend = MagicMock()
    backend.config = SimpleNamespace(
        backend=ctx["backend"], api_base=ctx["api_base"], upstream_model="upstream"
    )
    backend.build_params.return_value = (
        {"stream_options": {"include_usage": True}} if stream else {}
    )
    gate = asyncio.Event() if disconnect == "during_read" else None
    if stream:
        backend.converse = AsyncMock(return_value=_UpstreamStream(ctx["usage"], gate))
    else:
        response = MagicMock()
        response.usage = ctx["usage"]
        response.model_dump.return_value = {"id": "x"}

        async def converse(*_args, **_kwargs):
            if gate is not None:
                await gate.wait()
            return response

        backend.converse = AsyncMock(side_effect=converse)

    with (
        patch("aichat.serve.api_key_chat_api.run_dynamic_leo", new_callable=AsyncMock),
        patch(
            "aichat.serve.api_key_chat_api.select_model_for_request",
            new_callable=AsyncMock,
            side_effect=lambda model, **_: model,
        ),
        patch("aichat.serve.api_key_chat_api.get_backend", return_value=backend),
        patch("aichat.serve.api_key_chat_api.apply_claude_upstream_sampling_params"),
    ):
        handler = asyncio.create_task(
            api_key_chat_api.handle_chat_completions(_request(body), API_KEY)
        )
        if disconnect == "during_read" and not stream:
            await asyncio.sleep(0.01)
            handler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handler
            gate.set()
        else:
            resp = await handler

    events: list[str] = []
    if stream:
        iterator = resp.body_iterator
        if disconnect == "between_chunks":
            events.append(await iterator.__anext__())
            await iterator.aclose()
        elif disconnect == "during_read":
            consumer = asyncio.create_task(iterator.__anext__())
            events.append(await consumer)
            consumer = asyncio.create_task(iterator.__anext__())
            await asyncio.sleep(0.01)
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer
            gate.set()
        else:
            events = [event async for event in iterator]
    await _wait_for_background_tasks()

    ctx["events"] = events
    ctx["params"] = backend.converse.await_args.kwargs["params"]


@given("the internal API is enabled")
def _(monkeypatch):
    monkeypatch.setattr(
        api_key_chat_api.internal_settings, "internal_api_enabled", True
    )


@given("the internal API is disabled")
def _(monkeypatch):
    monkeypatch.setattr(
        api_key_chat_api.internal_settings, "internal_api_enabled", False
    )


@given(parsers.parse('the upstream model is served by the "{backend}" backend'))
def _(ctx, backend):
    ctx["backend"] = backend
    ctx["api_base"] = None


@given(
    parsers.parse(
        'the upstream model is served by the "{backend}" backend at "{api_base}"'
    )
)
def _(ctx, backend, api_base):
    ctx["backend"] = backend
    ctx["api_base"] = None if api_base == "none" else api_base


@given(
    parsers.parse(
        "the upstream reports usage of {prompt:d} prompt, {completion:d} completion, "
        "{cache_write:d} cache write, {cache_read:d} cache read and "
        "{reasoning:d} reasoning tokens"
    )
)
def _(ctx, prompt, completion, cache_write, cache_read, reasoning):
    ctx["usage"] = Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cache_creation_input_tokens=cache_write,
        cache_read_input_tokens=cache_read,
        reasoning_tokens=reasoning,
    )


@given("the upstream reports no usage")
def _(ctx):
    ctx["usage"] = None


@given("the client requests stream usage")
def _(ctx):
    ctx["body_extra"]["stream_options"] = {"include_usage": True}


@given("the client disables stream usage")
def _(ctx):
    ctx["body_extra"]["stream_options"] = {"include_usage": False}


@given("aichat-internal cannot settle usage")
def _(ctx):
    ctx["settle_result"] = None


@when("a non-streaming api-key chat completion is handled")
def _(ctx):
    asyncio.run(_handle(ctx, stream=False))


@when(
    "a non-streaming api-key chat completion is disconnected while waiting for the upstream"
)
def _(ctx):
    asyncio.run(_handle(ctx, stream=False, disconnect="during_read"))


@when(parsers.parse("{count:d} non-streaming api-key chat completions are handled"))
def _(ctx, count):
    for _ in range(count):
        asyncio.run(_handle(ctx, stream=False))


@when("a streaming api-key chat completion is handled")
def _(ctx):
    asyncio.run(_handle(ctx, stream=True))


@when("a streaming api-key chat completion is disconnected after the first chunk")
def _(ctx):
    asyncio.run(_handle(ctx, stream=True, disconnect="between_chunks"))


@when(
    "a streaming api-key chat completion is disconnected while waiting for the upstream"
)
def _(ctx):
    asyncio.run(_handle(ctx, stream=True, disconnect="during_read"))


@then("usage was settled once")
def _(ctx):
    assert len(ctx["settles"]) == 1


@then("usage was not settled")
def _(ctx):
    assert ctx["settles"] == []


@then(parsers.parse("usage was settled {count:d} times with distinct transaction ids"))
def _(ctx, count):
    assert len(ctx["settles"]) == count
    assert len({s["transaction_id"] for s in ctx["settles"]}) == count


@then(
    parsers.parse(
        'the settled usage carried the api key, model and provider "{provider}"'
    )
)
def _(ctx, provider):
    settled = ctx["settles"][-1]
    assert settled["api_key"] == API_KEY
    assert settled["model"] == MODEL
    assert settled["provider"] == provider


@then(
    parsers.parse(
        "the settled usage carried {prompt:d} prompt, {completion:d} completion, "
        "{cache_write:d} cache write, {cache_read:d} cache read and "
        "{reasoning:d} reasoning tokens"
    )
)
def _(ctx, prompt, completion, cache_write, cache_read, reasoning):
    settled = ctx["settles"][-1]
    assert settled["prompt_tokens"] == prompt
    assert settled["completion_tokens"] == completion
    assert settled["cache_write_tokens"] == cache_write
    assert settled["cache_read_tokens"] == cache_read
    assert settled["reasoning_tokens"] == reasoning


@then(parsers.parse('the settled transaction id starts with "{prefix}"'))
def _(ctx, prefix):
    assert ctx["settles"][-1]["transaction_id"].startswith(prefix)


@then(parsers.parse('the settle outcome "{outcome}" was recorded'))
def _(ctx, outcome):
    assert _settle_count(outcome) == ctx["metrics_before"][outcome] + 1


@then("the stream ends with a DONE sentinel")
def _(ctx):
    assert ctx["events"][-1] == "data: [DONE]\n\n"


@then("the client stream contains no usage")
def _(ctx):
    assert not any('"prompt_tokens"' in event for event in ctx["events"])


@then("the client stream contains the usage")
def _(ctx):
    assert any('"prompt_tokens":13' in event for event in ctx["events"])


@then("the upstream request asked for stream usage")
def _(ctx):
    assert ctx["params"]["stream_options"]["include_usage"] is True
