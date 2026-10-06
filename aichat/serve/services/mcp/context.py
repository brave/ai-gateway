from __future__ import annotations

from typing import Literal

from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.models import get_model_config

ModelTier = Literal["premium", "freemium", "free"]
ToolOrchestrationMode = Literal["assisted", "native"]

HEADER_TIER = "X-Brave-Tier"
HEADER_TOOL_ORCHESTRATION = "X-Brave-Tool-Orchestration"
DEFAULT_TIER: ModelTier = "free"
DEFAULT_TOOL_ORCHESTRATION: ToolOrchestrationMode = "assisted"


def _normalize_tool_orchestration(raw: str | None) -> ToolOrchestrationMode:
    if (raw or "").strip().lower() == "native":
        return "native"
    return DEFAULT_TOOL_ORCHESTRATION


def mcp_context_headers_for_model(model: str) -> dict[str, str]:
    tier: ModelTier = DEFAULT_TIER
    orchestration: ToolOrchestrationMode = DEFAULT_TOOL_ORCHESTRATION
    if model in model_settings.models:
        model_config = get_model_config(model)
        if model_config is not None:
            tier = model_config.token_amount_tier
            orchestration = _normalize_tool_orchestration(
                model_config.tool_orchestration
            )
    return {
        HEADER_TIER: tier,
        HEADER_TOOL_ORCHESTRATION: orchestration,
    }


def tier_http_headers_for_model(model: str) -> dict[str, str]:
    """Backward-compatible alias; returns tier and tool orchestration headers."""
    return mcp_context_headers_for_model(model)
