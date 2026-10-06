"""Tests for MCP model tier headers."""

from unittest.mock import patch

from aichat.serve.services.mcp.context import (
    DEFAULT_TIER,
    DEFAULT_TOOL_ORCHESTRATION,
    HEADER_TIER,
    HEADER_TOOL_ORCHESTRATION,
    mcp_context_headers_for_model,
    tier_http_headers_for_model,
)
from aichat.serve.services.models import ModelConfig


def test_tier_http_headers_for_model_defaults_to_free():
    assert tier_http_headers_for_model("missing-model") == {
        HEADER_TIER: DEFAULT_TIER,
        HEADER_TOOL_ORCHESTRATION: DEFAULT_TOOL_ORCHESTRATION,
    }


def test_mcp_context_headers_for_model_native_orchestration():
    model_config = ModelConfig(
        model_id="frontier-model",
        upstream_model="sonnet-upstream",
        backend="bedrock",
        api_base=None,
        api_key=None,
        inference_profile=None,
        system_prompt_support=True,
        prompt_caching_support=None,
        prompt_caching_enabled=None,
        tool_support=True,
        image_support=False,
        audio_support=False,
        video_support=False,
        file_support=True,
        friendly_name="Frontier",
        maker="AWS",
        max_tokens=8192,
        max_tokens_premium=8192,
        conversation_token_limit=128000,
        conversation_token_limit_premium=128000,
        free=False,
        key="frontier-key",
        rate_limit=None,
        rate_limit_interval_seconds=None,
        max_pages=3,
        max_pages_premium=3,
        extra_body=None,
        token_amount_tier="premium",
        tool_orchestration="native",
    )
    with patch("aichat.serve.services.mcp.context.model_settings") as mock_settings:
        mock_settings.models = {"frontier-model": {}}
        with patch(
            "aichat.serve.services.mcp.context.get_model_config",
            return_value=model_config,
        ):
            assert mcp_context_headers_for_model("frontier-model") == {
                HEADER_TIER: "premium",
                HEADER_TOOL_ORCHESTRATION: "native",
            }


def test_tier_http_headers_for_model_uses_model_config_tier():
    model_config = ModelConfig(
        model_id="premium-model",
        upstream_model="premium-upstream",
        backend="bedrock",
        api_base=None,
        api_key=None,
        inference_profile=None,
        system_prompt_support=True,
        prompt_caching_support=None,
        prompt_caching_enabled=None,
        tool_support=True,
        image_support=False,
        audio_support=False,
        video_support=False,
        file_support=True,
        friendly_name="Premium Model",
        maker="AWS",
        max_tokens=8192,
        max_tokens_premium=8192,
        conversation_token_limit=128000,
        conversation_token_limit_premium=128000,
        free=False,
        key="premium-key",
        rate_limit=None,
        rate_limit_interval_seconds=None,
        max_pages=3,
        max_pages_premium=3,
        extra_body=None,
        token_amount_tier="premium",
    )
    with patch("aichat.serve.services.mcp.context.model_settings") as mock_settings:
        mock_settings.models = {"premium-model": {}}
        with patch(
            "aichat.serve.services.mcp.context.get_model_config",
            return_value=model_config,
        ):
            assert tier_http_headers_for_model("premium-model") == {
                HEADER_TIER: "premium",
                HEADER_TOOL_ORCHESTRATION: DEFAULT_TOOL_ORCHESTRATION,
            }
