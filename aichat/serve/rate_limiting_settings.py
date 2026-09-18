from pydantic_settings import BaseSettings


class RateLimitingSettings(BaseSettings):
    rate_limiting_enabled: bool = False
    salt_cache_ttl_seconds: int = 24 * 60 * 60
    automatic_premium_model_daily_response_limit: int = 3


rate_limiting_settings = RateLimitingSettings()
