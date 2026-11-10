import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import api_key_chat_api

scenarios("features/api_key_search_capability.feature")

API_KEY = "test_key_" + "a" * 10
SEARCH_TOOL = {"type": "function", "function": {"name": "brave_web_search"}}


class _Stream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


@pytest.fixture
def ctx(monkeypatch):
    state = {"verify_calls": 0, "verdict": {"allowed": True}}

    async def api_key_verify(_client, api_key, model=None):
        state["verify_calls"] += 1
        return state["verdict"]

    monkeypatch.setattr(
        api_key_chat_api.internal_client, "api_key_verify", api_key_verify
    )
    return state


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


@given(parsers.parse('the default key capabilities are "{capabilities}"'))
def _(monkeypatch, capabilities):
    monkeypatch.setattr(
        api_key_chat_api.api_key_chat_settings,
        "default_key_capabilities",
        capabilities.split(","),
    )


@given(parsers.parse('verify allows the key with capabilities "{capabilities}"'))
def _(ctx, capabilities):
    ctx["verdict"] = {"allowed": True, "key_capabilities": capabilities.split(",")}


@given("verify allows the key without capabilities")
def _(ctx):
    ctx["verdict"] = {"allowed": True}


@given("verify allows the key with empty capabilities")
def _(ctx):
    ctx["verdict"] = {"allowed": True, "key_capabilities": []}


def _request_completion(ctx, flags):
    request = MagicMock(spec=Request)
    request.json = AsyncMock(
        return_value={
            "model": "requested-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            **flags,
        }
    )
    request.headers = {"authorization": f"Bearer {API_KEY}"}
    request.state = MagicMock()

    backend = MagicMock()
    backend.config = SimpleNamespace(
        backend="bedrock", api_base=None, upstream_model="upstream"
    )
    backend.build_params.side_effect = lambda **kwargs: {"tools": kwargs["tools"]}
    backend.converse = AsyncMock(return_value=_Stream())
    load_search_tools = AsyncMock(return_value=([SEARCH_TOOL], None))

    async def run():
        with (
            patch(
                "aichat.serve.services.api_key_chat.run_dynamic_leo",
                new_callable=AsyncMock,
            ),
            patch(
                "aichat.serve.services.api_key_chat.select_model_for_request",
                new_callable=AsyncMock,
                side_effect=lambda model, **_: model,
            ),
            patch(
                "aichat.serve.services.api_key_chat.get_backend",
                MagicMock(return_value=backend),
            ),
            patch(
                "aichat.serve.services.api_key_chat.load_search_tools",
                load_search_tools,
            ),
            patch(
                "aichat.serve.services.chat_request.apply_claude_upstream_sampling_params"
            ),
        ):
            return await api_key_chat_api.chat_completions(request)

    ctx["response"] = asyncio.run(run())
    body = getattr(ctx["response"], "body", None)
    ctx["body"] = json.loads(body) if body else None
    ctx["converse"] = backend.converse
    ctx["load_search_tools"] = load_search_tools


@when(parsers.parse('a streaming api-key chat completion is requested with "{flag}"'))
def _(ctx, flag):
    _request_completion(ctx, {flag: True})


@when("a streaming api-key chat completion is requested without search")
def _(ctx):
    _request_completion(ctx, {})


@then(parsers.parse("the response status is {status:d}"))
def _(ctx, status):
    assert ctx["response"].status_code == status


@then(parsers.parse('the error code is "{code}"'))
def _(ctx, code):
    assert ctx["body"]["error"]["code"] == code


@then("verify was not called")
def _(ctx):
    assert ctx["verify_calls"] == 0


@then("search tools were loaded")
def _(ctx):
    ctx["load_search_tools"].assert_awaited_once()


@then("search tools were not loaded")
def _(ctx):
    ctx["load_search_tools"].assert_not_awaited()


@then("a completion was served")
def _(ctx):
    ctx["converse"].assert_awaited_once()


@then("the completion was served with the search tool")
def _(ctx):
    ctx["converse"].assert_awaited_once()
    assert ctx["converse"].call_args.kwargs["params"]["tools"] == [SEARCH_TOOL]


@then("no completion was served")
def _(ctx):
    ctx["converse"].assert_not_awaited()
