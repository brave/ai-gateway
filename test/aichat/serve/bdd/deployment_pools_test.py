import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import REGISTRY
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve.backend import litellm as litellm_backend
from aichat.serve.backend.litellm import LitellmBackend
from aichat.serve.services import deployment_pool_settings as dps_module
from test.aichat.serve.bdd.helpers import dataclass_model_config

FEATURE = Path(__file__).parent / "features" / "deployment_pools.feature"
scenarios(str(FEATURE))

_MODEL_ID = "qwen-14b-instruct"
_POOLS = {
    "short_text": {
        "address": "http://vllm-short:8000/v1",
        "max_model_tokens": 20480,
    },
    "long_text": {
        "address": "http://vllm-long:8000/v1",
        "max_model_tokens": 81920,
    },
    "image": {"address": None, "enabled": False},
}


@pytest.fixture
def ctx():
    return {}


def _router_completion():
    return {
        "id": "resp-1",
        "object": "chat.completion",
        "created": 1,
        "model": _MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }


def _install_model_catalog(monkeypatch):
    models = {
        _MODEL_ID: {
            "backend": "litellm",
            "type": "llm",
            "upstream_model": "Qwen/Qwen3-14B",
            "address": "http://vllm-long:8000/v1",
            "deployment_pools": _POOLS,
        }
    }
    monkeypatch.setattr(litellm_backend.model_settings, "models", models)


@given(parsers.parse('model "{model_id}" has deployment pools configured'))
def given_pooled_model(model_id, monkeypatch, ctx):
    assert model_id == _MODEL_ID
    _install_model_catalog(monkeypatch)
    router = MagicMock()
    router.acompletion = AsyncMock(return_value=_router_completion())
    monkeypatch.setattr(litellm_backend, "get_global_router", lambda: router)
    ctx["router"] = router
    ctx["model_config"] = dataclass_model_config(
        model_id=model_id,
        upstream_model="Qwen/Qwen3-14B",
    )


@given("deployment pool routing is disabled")
def given_routing_off(monkeypatch):
    monkeypatch.setattr(
        dps_module.deployment_pool_settings, "deployment_pools_enabled", False
    )


@given("deployment pool routing is enabled")
def given_routing_on(monkeypatch):
    monkeypatch.setattr(
        dps_module.deployment_pool_settings, "deployment_pools_enabled", True
    )


@given("a short user message for pool routing")
def given_short_message(ctx):
    ctx["messages"] = [{"role": "user", "content": "hello"}]
    ctx["max_tokens"] = 1024


@given("a long user message for pool routing")
def given_long_message(ctx, monkeypatch):
    ctx["messages"] = [{"role": "user", "content": "large"}]
    ctx["max_tokens"] = 8192
    monkeypatch.setattr(
        "aichat.serve.services.deployment_pools.calculate_message_tokens",
        lambda _messages: 25000,
    )


@given("deployment pool shadow metrics baseline is captured")
def given_metric_baseline(ctx):
    ctx["metric_before"] = (
        REGISTRY.get_sample_value(
            "deployment_pool_route_total",
            {
                "model": _MODEL_ID,
                "pool": "short_text",
                "route_reason": "short",
                "mode": "shadow",
            },
        )
        or 0.0
    )


@when("the litellm backend runs chat completion")
def when_backend_converses(ctx):
    backend = LitellmBackend(ctx["model_config"])

    async def _run():
        return await backend.converse(
            ctx["messages"],
            stream=False,
            params={"max_tokens": ctx["max_tokens"]},
        )

    ctx["response"] = asyncio.run(_run())


def _completion_kwargs(ctx):
    return ctx["router"].acompletion.await_args.kwargs


@then("the router completion has no pool tags")
def then_no_tags(ctx):
    metadata = _completion_kwargs(ctx).get("metadata") or {}
    assert not metadata.get("tags")


@then(parsers.parse('the router completion includes pool tag "{tag}"'))
def then_pool_tag(ctx, tag):
    metadata = _completion_kwargs(ctx).get("metadata") or {}
    assert metadata.get("tags") == [tag]


@then(
    parsers.parse(
        'deployment_pool_route_total increases for pool "{pool}" mode "{mode}"'
    )
)
def then_shadow_metric(ctx, pool, mode):
    value = REGISTRY.get_sample_value(
        "deployment_pool_route_total",
        {"model": _MODEL_ID, "pool": pool, "route_reason": "short", "mode": mode},
    )
    assert value == ctx["metric_before"] + 1
