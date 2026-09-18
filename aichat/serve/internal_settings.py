from pydantic_settings import BaseSettings


class InternalSettings(BaseSettings):
    # False (default/self-host): auth/SKU/rate-limit checks that would
    # consult aichat-internal are bypassed.
    internal_api_enabled: bool = False
    internal_base_url: str = "http://aichat-internal:8000"
    internal_request_timeout_seconds: float = 3.0


internal_settings = InternalSettings()
