"""Classify vLLM deployment pools for models with ``deployment_pools`` config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aichat.llm.metrics import DEPLOYMENT_POOL_ROUTE_TOTAL
from aichat.serve.services.deployment_pool_settings import deployment_pool_settings
from aichat.serve.services.model_settings import model_settings
from aichat.serve.utils import calculate_message_tokens

POOL_SHORT = "short_text"
POOL_LONG = "long_text"
POOL_IMAGE = "image"

_TAG_BY_POOL = {
    POOL_SHORT: "pool:short",
    POOL_LONG: "pool:long",
    POOL_IMAGE: "pool:image",
}

_VISION_PART_TYPES = frozenset({"image_url", "image"})


@dataclass(frozen=True)
class DeploymentPoolRoute:
    pool: str
    tag: str
    reason: str


def get_deployment_pools_config(model_id: str) -> dict[str, Any] | None:
    pools = model_settings.models.get(model_id, {}).get("deployment_pools")
    if isinstance(pools, dict) and pools:
        return pools
    return None


def short_admission_token_budget(pools: dict[str, Any]) -> int:
    short_cfg = pools.get(POOL_SHORT) or {}
    if short_cfg.get("admission_max_tokens") is not None:
        return max(0, int(short_cfg["admission_max_tokens"]))
    settings = deployment_pool_settings
    max_model = int(
        short_cfg.get(
            "max_model_tokens", settings.deployment_pool_short_max_model_tokens
        )
    )
    margin = int(
        short_cfg.get(
            "admission_margin_tokens",
            settings.deployment_pool_short_admission_margin_tokens,
        )
    )
    return max(0, max_model - margin)


def long_pool_max_input_tokens(pools: dict[str, Any]) -> int:
    long_cfg = pools.get(POOL_LONG) or {}
    settings = deployment_pool_settings
    if long_cfg.get("max_input_tokens") is not None:
        return int(long_cfg["max_input_tokens"])
    if long_cfg.get("max_model_tokens") is not None:
        return int(long_cfg["max_model_tokens"])
    return settings.deployment_pool_long_max_input_tokens


def classify_deployment_pool(
    messages: list[dict],
    max_tokens: int,
    pools: dict[str, Any],
) -> DeploymentPoolRoute:
    reserved = max(0, int(max_tokens or 0))
    input_tokens = calculate_message_tokens(messages)

    if _messages_contain_vision(messages):
        image_cfg = pools.get(POOL_IMAGE) or {}
        if image_cfg.get("enabled") and image_cfg.get("address"):
            pool, reason = POOL_IMAGE, "image"
        else:
            pool, reason = POOL_LONG, "interim_vision"
    elif input_tokens + reserved > short_admission_token_budget(pools):
        pool, reason = POOL_LONG, "long"
    else:
        pool, reason = POOL_SHORT, "short"

    return DeploymentPoolRoute(pool=pool, tag=_TAG_BY_POOL[pool], reason=reason)


def pool_router_tag(pool_name: str) -> str:
    return _TAG_BY_POOL[pool_name]


def record_deployment_pool_route(
    model_id: str, route: DeploymentPoolRoute, *, routing_active: bool
) -> None:
    mode = "active" if routing_active else "shadow"
    DEPLOYMENT_POOL_ROUTE_TOTAL.labels(model_id, route.pool, route.reason, mode).inc()


def record_short_pool_failover(model_id: str) -> None:
    DEPLOYMENT_POOL_ROUTE_TOTAL.labels(model_id, POOL_LONG, "fallback", "active").inc()


def _messages_contain_vision(messages: list[dict]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in _VISION_PART_TYPES:
                return True
    return False
