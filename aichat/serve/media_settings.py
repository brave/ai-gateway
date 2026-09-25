from pydantic_settings import BaseSettings


class MediaSettings(BaseSettings):
    media_base_url: str = "http://ai-gateway-media-processor:8000"
    media_request_timeout_seconds: float = 15.0


media_settings = MediaSettings()
