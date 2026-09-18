from __future__ import annotations

from typing import Literal

from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.models import get_model_config

ModelTier = Literal["premium", "freemium", "free"]

HEADER_TIER = "X-Brave-Tier"
DEFAULT_TIER: ModelTier = "free"


def tier_http_headers_for_model(model: str) -> dict[str, str]:
    tier: ModelTier = DEFAULT_TIER
    if model in model_settings.models:
        model_config = get_model_config(model)
        if model_config is not None:
            tier = model_config.token_amount_tier
    return {HEADER_TIER: tier}
