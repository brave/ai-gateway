from pydantic_settings import BaseSettings


class DeploymentPoolSettings(BaseSettings):
    # Apply LiteLLM tag routing for models with deployment_pools in MODELS.
    # Shadow classification metrics run whenever deployment_pools is configured,
    # even when this is false.
    deployment_pools_enabled: bool = False

    # Defaults for vLLM deployment pool admission (short_text / long_text lanes).
    # Per-model deployment_pools.short_text.* in MODELS / models.json overrides these.
    # Effective short budget is admission_max_tokens, or max_model_tokens - margin (~12k default).
    deployment_pool_short_max_model_tokens: int = 12512
    deployment_pool_short_admission_margin_tokens: int = 512
    deployment_pool_long_max_input_tokens: int = 81920


deployment_pool_settings = DeploymentPoolSettings()
