"""Coverage tests for aichat/serve/backend/litellm.py (litellm)."""

import asyncio
from unittest.mock import MagicMock, patch

import litellm as litellm_lib
import pytest

from aichat.serve.backend import litellm as M

# --- apply_claude_upstream_sampling_params ---------------------------------


def test_apply_claude_opus_pops_all_sampling():
    params = {"temperature": 1, "top_p": 0.9, "top_k": 40}
    M.apply_claude_upstream_sampling_params("claude-opus-4", params)
    assert params == {}


def test_apply_claude_sonnet_pops_all_sampling():
    params = {"temperature": 1, "top_p": 0.9, "top_k": 40}
    M.apply_claude_upstream_sampling_params("claude-sonnet-4", params)
    assert params == {}


def test_apply_claude_other_pops_top_p_only():
    params = {"temperature": 1, "top_p": 0.9, "top_k": 40}
    M.apply_claude_upstream_sampling_params("claude-3-haiku", params)
    assert params == {"temperature": 1, "top_k": 40}


def test_apply_claude_none_upstream_returns_early():
    params = {"temperature": 1}
    M.apply_claude_upstream_sampling_params(None, params)
    assert params == {"temperature": 1}


# --- bedrock_mantle_openai_responses_routing --------------------------------


@pytest.mark.asyncio
async def test_routing_non_bedrock_yields_without_flag():
    before = getattr(litellm_lib, "route_all_chat_openai_to_responses", None)
    async with M.bedrock_mantle_openai_responses_routing("gpt-4", "vllm"):
        assert (
            getattr(litellm_lib, "route_all_chat_openai_to_responses", None) == before
        )
    assert getattr(litellm_lib, "route_all_chat_openai_to_responses", None) == before


@pytest.mark.asyncio
async def test_routing_bedrock_sets_and_restores_flag(monkeypatch):
    monkeypatch.setattr(M, "is_bedrock_mantle_openai_responses_model", lambda *_: True)
    monkeypatch.setattr(litellm_lib, "route_all_chat_openai_to_responses", False)
    async with M.bedrock_mantle_openai_responses_routing(
        "openai.gpt-4", "bedrock_mantle"
    ):
        assert litellm_lib.route_all_chat_openai_to_responses is True
    assert litellm_lib.route_all_chat_openai_to_responses is False


# --- retry callback + extras -------------------------------------------------


def test_retry_callback_singleton():
    with patch.dict(M._retry_callbacks, {}, clear=True):
        first = M.get_retry_callback("m1")
        again = M.get_retry_callback("m1")
        other = M.get_retry_callback("m2")
        assert first is again
        assert first is not other
        assert first.model_id == "m1"


def test_bedrock_mantle_completion_extras(monkeypatch):
    monkeypatch.setattr(
        M, "get_bedrock_mantle_bearer_token", lambda: "tok", raising=False
    )
    assert M._bedrock_mantle_completion_extras("bedrock_mantle") == {"api_key": "tok"}
    assert M._bedrock_mantle_completion_extras("vllm") == {}


# --- _get_litellm_model_string ------------------------------------------------


def test_model_string_bedrock_mantle_openai_bridge():
    assert (
        M._get_litellm_model_string("openai.gpt-4", "http://x", None, "bedrock_mantle")
        == "openai/openai.gpt-4"
    )


def test_model_string_bedrock_mantle_plain():
    assert (
        M._get_litellm_model_string(
            "anthropic.claude", "http://x", None, "bedrock_mantle"
        )
        == "bedrock_mantle/anthropic.claude"
    )


def test_model_string_inference_profile_env_set(monkeypatch):
    monkeypatch.setenv("MY_PROFILE", "arn:aws:profile-abc")
    assert (
        M._get_litellm_model_string("m", "http://x", "MY_PROFILE")
        == "bedrock/converse/arn:aws:profile-abc"
    )


def test_model_string_inference_profile_env_unset(monkeypatch):
    monkeypatch.delenv("MY_PROFILE", raising=False)
    assert M._get_litellm_model_string("m", "http://x", "MY_PROFILE") == "bedrock/m"


def test_model_string_near_api_base():
    assert M._get_litellm_model_string("m", "https://api.near.ai/x", None) == "openai/m"


def test_model_string_default_hosted_vllm():
    assert M._get_litellm_model_string("m", "http://x", None) == "hosted_vllm/m"


# --- _build_router_entry -------------------------------------------------------


def test_build_router_entry_missing_model_returns_none():
    with patch.object(M.model_settings, "models", {}, create=True):
        assert M._build_router_entry("nope") is None


def _model_config(**overrides):
    cfg = {
        "upstream_model": "up",
        "address": "http://x",
        "backend": "vllm",
    }
    cfg.update(overrides)
    return cfg


def test_build_router_entry_vllm_zero_costs(monkeypatch):
    monkeypatch.setattr(
        M.model_settings,
        "models",
        {"m1": {"upstream_model": "up", "address": "http://x", "backend": "vllm"}},
        raising=False,
    )
    entry = M._build_router_entry("m1")
    assert entry["model_name"] == "m1"
    assert entry["litellm_params"]["model"].endswith("up")
    assert entry["model_info"]["input_cost_per_token"] == 0


def test_build_router_entry_near_uses_api_key(monkeypatch):
    near_cfg = {
        "upstream_model": "up",
        "address": "http://near",
        "backend": "vllm",
    }
    monkeypatch.setattr(M.model_settings, "models", {"near-m": near_cfg}, raising=False)
    monkeypatch.setattr(
        M.external_service_settings, "near_api_key", "nk", raising=False
    )
    entry = M._build_router_entry("near-m")
    assert entry["litellm_params"]["api_key"] == "nk"


def test_build_router_entry_passthrough_fields(monkeypatch):
    monkeypatch.setattr(
        M.model_settings,
        "models",
        {
            "p1": {
                "upstream_model": "up",
                "address": "http://x",
                "backend": "vllm",
                "endpoint": "/ep",
                "method": "POST",
                "api_key": "pk",
            }
        },
        raising=False,
    )
    entry = M._build_router_entry("p1")
    assert entry["litellm_params"]["endpoint"] == "/ep"
    assert entry["litellm_params"]["method"] == "POST"
    assert entry["litellm_params"]["api_key"] == "pk"


def test_build_router_entry_bedrock_mantle_drops_output_config(monkeypatch):
    monkeypatch.setattr(
        M.model_settings,
        "models",
        {
            "b1": {
                "upstream_model": "openai.gpt-4",
                "address": "http://x",
                "backend": "bedrock_mantle",
            }
        },
        raising=False,
    )
    entry = M._build_router_entry("b1")
    assert entry["litellm_params"]["additional_drop_params"] == ["output_config"]


# --- get_global_router ----------------------------------------------------------


def test_get_global_router_builds_entries(monkeypatch):
    monkeypatch.setattr(M, "_global_router", None, raising=False)
    models = {
        "a": {"backend": "vllm", "upstream_model": "upa", "address": "http://x"},
        "b": {"backend": "vllm", "upstream_model": "upb", "address": "http://x"},
        "ens": {
            "type": "ensemble",
            "backend": "litellm",
            "models": [{"model": "a", "weight": 1}, {"model": "b", "weight": 3}],
        },
        "m1": {
            "type": "llm",
            "backend": "litellm",
            "fallback_models": ["f1"],
            "upstream_model": "up-m1",
            "address": "http://x",
        },
        "f1": {"backend": "vllm", "upstream_model": "up-f1", "address": "http://x"},
        "img": {
            "type": "image_generation",
            "backend": "litellm",
            "upstream_model": "up-img",
            "address": "http://x",
        },
        "tts": {
            "type": "tts",
            "backend": "litellm",
            "upstream_model": "up-tts",
            "address": "http://x",
        },
        "emb": {
            "type": "embedding",
            "backend": "litellm",
            "upstream_model": "up-emb",
            "address": "http://x",
        },
        "cls": {
            "type": "classifier",
            "backend": "litellm",
            "upstream_model": "up-cls",
            "address": "http://x",
        },
        "stt": {
            "type": "speech_to_text",
            "backend": "litellm",
            "upstream_model": "up-stt",
            "address": "http://x",
        },
        "bad": {"type": "martian", "backend": "litellm"},
    }
    router_mock = MagicMock()
    router_factory = MagicMock(return_value=router_mock)
    monkeypatch.setattr(M, "Router", router_factory, raising=True)
    monkeypatch.setattr(M.model_settings, "models", models, raising=False)
    router = M.get_global_router()
    assert router is router_mock
    kwargs = router_factory.call_args.kwargs
    names = [e["model_name"] for e in kwargs["model_list"]]
    assert names == [
        "a",
        "b",
        "ens",
        "ens",
        "m1",
        "f1",
        "img",
        "tts",
        "emb",
        "cls",
        "stt",
    ]
    assert kwargs["fallbacks"] == [{"m1": ["f1"]}]
    assert litellm_lib.in_memory_llm_clients_cache.max_size_in_memory == 100
    assert M.get_global_router() is router


# --- LitellmBackend init ---------------------------------------------------------


def test_litellm_backend_init_sets_profile_and_template(monkeypatch):
    monkeypatch.setattr(M, "get_global_router", lambda: MagicMock(), raising=True)
    monkeypatch.setattr(M, "get_retry_callback", lambda mid: MagicMock(), raising=True)
    monkeypatch.setenv("MY_PROFILE", "arn:aws:bedrock:us-west-2:x")
    cfg = MagicMock()
    cfg.model_id = "m1"
    cfg.inference_profile = "MY_PROFILE"
    cfg.extra_body = {"chat_template_kwargs": {"k": "v"}}
    backend = M.LitellmBackend(cfg)
    assert backend.inference_profile == "arn:aws:bedrock:us-west-2:x"
    assert backend.chat_template_kwargs == {"k": "v"}


def test_litellm_backend_no_inference_profile(monkeypatch):
    monkeypatch.setattr(M, "get_global_router", lambda: MagicMock(), raising=True)
    monkeypatch.setattr(M, "get_retry_callback", lambda mid: MagicMock(), raising=True)
    monkeypatch.delenv("MY_PROFILE", raising=False)
    cfg = MagicMock()
    cfg.model_id = "m1"
    cfg.inference_profile = "MY_PROFILE"
    cfg.extra_body = None
    backend = M.LitellmBackend(cfg)
    assert backend.inference_profile is None
    assert backend.chat_template_kwargs == {}


# --- build_params fallback chain --------------------------------------------------


def test_build_params_fallback_chain(monkeypatch):
    monkeypatch.setattr(
        M.model_settings,
        "models",
        {
            "f1": {"upstream_model": "claude-3-haiku", "reasoning_effort": "x"},
            "f2": {"upstream_model": "claude-opus-4"},
        },
        raising=False,
    )
    monkeypatch.setattr(
        M.model_settings,
        "models",
        M.model_settings.__dict__.get("models", {}),
        raising=False,
    )
    monkeypatch.setattr(M, "count_tokens", lambda *a, **k: 1000, raising=True)
    cfg = MagicMock()
    cfg.model_id = "m1"
    cfg.upstream_model = "m1"
    cfg.fallback_models = ["f1", "f2", "f1"]
    cfg.max_tokens = 10
    cfg.reasoning_effort = "low"
    cfg.prompt_caching_support = False
    cfg.extra_body = {}
    backend = M.LitellmBackend.__new__(M.LitellmBackend)
    backend.config = cfg
    params = backend.build_params(
        stream=False, tools=None, messages=[], context_window_override=100
    )
    assert params["max_tokens"] == 512
    # f2 lacks reasoning_effort support -> chain drops it
    assert "reasoning_effort" not in params


def test_build_params_drops_reasoning_effort_without_fallback_support(monkeypatch):
    monkeypatch.setattr(
        M.model_settings,
        "models",
        {"f1": {"upstream_model": "up-f1"}},
        raising=False,
    )
    monkeypatch.setattr(M, "count_tokens", lambda *a, **k: 1000, raising=True)
    cfg = MagicMock()
    cfg.model_id = "m1"
    cfg.upstream_model = "m1"
    cfg.fallback_models = ["f1"]
    cfg.max_tokens = 10
    cfg.reasoning_effort = "low"
    cfg.prompt_caching_support = False
    cfg.extra_body = {}
    backend = M.LitellmBackend.__new__(M.LitellmBackend)
    backend.config = cfg
    params = backend.build_params(
        stream=False, tools=None, messages=[], context_window_override=100
    )
    assert "reasoning_effort" not in params


# --- handle_litellm_error ---------------------------------------------------------


class _FakeErr(Exception):
    pass


def test_handle_error_context_window():
    resp = M.handle_litellm_error(
        litellm_lib.ContextWindowExceededError("too long", model="m", llm_provider="p")
    )
    assert resp["code"] == M.ErrorCode.EXCEEDED_CONTEXT_LENGTH.value
    assert "Conversation length exceeded" in resp["content"]


def test_handle_error_connection():
    resp = M.handle_litellm_error(
        litellm_lib.APIConnectionError("refused", model="m", llm_provider="p")
    )
    assert "issue connecting" in resp["content"]


def test_handle_error_bad_request_json_message():
    resp = M.handle_litellm_error(
        litellm_lib.BadRequestError(
            '{"error":{"message":"bad input"}}', model="m", llm_provider="p"
        )
    )
    assert resp["content"] == "bad input"


def test_handle_error_bedrock_length():
    resp = M.handle_litellm_error(
        litellm_lib.BadRequestError(
            "BedrockException length limit exceeded", model="m", llm_provider="p"
        )
    )
    assert resp["content"] == "Your request is too large."


def test_handle_error_uncaught():
    resp = M.handle_litellm_error(ValueError("weird"))
    assert resp["code"] == M.ErrorCode.INTERNAL_ERROR.value


def test_extract_message_already_borrowed():
    msg = M.extract_message_from_bad_request_error(
        litellm_lib.BadRequestError(
            '{"error":{"message":"Already borrowed x"}}', model="m", llm_provider="p"
        )
    )
    assert "temporarily busy" in msg


def test_yield_error_response():

    async def _consume():
        chunks = []
        async for chunk in M.yield_error_response({"content": "boom"}, model="m1"):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_consume())
    assert chunks[0].choices[0].delta.content == "boom"
    assert chunks[0].model == "m1"
