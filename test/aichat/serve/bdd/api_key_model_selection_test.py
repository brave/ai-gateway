import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from litellm.types.utils import Usage
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import api_key_chat_api
from aichat.serve.services import api_key_usage

scenarios("features/api_key_model_selection.feature")

API_KEY = "test_key_" + "a" * 10


class _Completion:
    def __init__(self):
        self.model = "upstream"
        self.usage = Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5)

    def model_dump(self):
        return {"id": "x", "model": self.model}


@pytest.fixture
def ctx(monkeypatch):
    state = {"verify_calls": [], "settles": [], "verdict": {"allowed": True}}

    async def api_key_verify(_client, api_key, model=None):
        state["verify_calls"].append(model)
        return state["verdict"]

    async def usage_settle(_client, **kwargs):
        state["settles"].append(kwargs)
        return {}

    async def api_key_preflight(_client, **kwargs):
        return {"allowed": True}

    monkeypatch.setattr(
        api_key_chat_api.internal_client, "api_key_verify", api_key_verify
    )
    monkeypatch.setattr(api_key_usage.internal_client, "usage_settle", usage_settle)
    monkeypatch.setattr(
        api_key_usage.internal_client, "api_key_preflight", api_key_preflight
    )
    return state


@given("the internal API is enabled")
def _(monkeypatch):
    monkeypatch.setattr(
        api_key_chat_api.internal_settings, "internal_api_enabled", True
    )


@given(parsers.parse('verify allows the key and selects model "{model}"'))
def _(ctx, model):
    ctx["verdict"] = {"allowed": True, "model": model}


@given("verify allows the key without selecting a model")
def _(ctx):
    ctx["verdict"] = {"allowed": True}


@given(parsers.parse('verify denies the key with reason "{reason}"'))
def _(ctx, reason):
    ctx["verdict"] = {"allowed": False, "reason": reason}


@when(parsers.parse('an api-key chat completion is requested for model "{model}"'))
def _(ctx, model):
    request = MagicMock(spec=Request)
    request.json = AsyncMock(
        return_value={"model": model, "messages": [{"role": "user", "content": "hi"}]}
    )
    request.headers = {"authorization": f"Bearer {API_KEY}"}
    request.state = MagicMock()

    backend = MagicMock()
    backend.config = SimpleNamespace(
        backend="bedrock", api_base=None, upstream_model="upstream"
    )
    backend.build_params.return_value = {}
    backend.converse = AsyncMock(return_value=_Completion())
    get_backend = MagicMock(return_value=backend)

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
            patch("aichat.serve.services.api_key_chat.get_backend", get_backend),
            patch(
                "aichat.serve.services.chat_request.apply_claude_upstream_sampling_params"
            ),
        ):
            response = await api_key_chat_api.chat_completions(request)
        while api_key_usage._background_tasks:
            await asyncio.gather(*list(api_key_usage._background_tasks))
            await asyncio.sleep(0)
        return response

    ctx["response"] = asyncio.run(run())
    ctx["body"] = json.loads(ctx["response"].body)
    ctx["served"] = [c.args[0] for c in get_backend.call_args_list]


@then(parsers.parse('verify was asked about model "{model}"'))
def _(ctx, model):
    assert ctx["verify_calls"] == [model]


@then(parsers.parse('the completion was served by model "{model}"'))
def _(ctx, model):
    assert ctx["served"] == [model]


@then("no completion was served")
def _(ctx):
    assert ctx["served"] == []


@then(parsers.parse("the response status is {status:d}"))
def _(ctx, status):
    assert ctx["response"].status_code == status


@then(parsers.parse('the response reports model "{model}"'))
def _(ctx, model):
    assert ctx["body"]["model"] == model


@then(parsers.parse('the error code is "{code}"'))
def _(ctx, code):
    assert ctx["body"]["error"]["code"] == code


@then(parsers.parse('usage was settled for model "{model}"'))
def _(ctx, model):
    assert [s["model"] for s in ctx["settles"]] == [model]


@then("no usage was settled")
def _(ctx):
    assert ctx["settles"] == []
