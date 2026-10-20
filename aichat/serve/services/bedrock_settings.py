from typing import Literal

from pydantic_settings import BaseSettings


class BedrockSettings(BaseSettings):
    bedrock_cache_min_tokens: int = 4096
    bedrock_guardrail_id: str | None = None
    bedrock_guardrail_version: str | None = None
    bedrock_guardrail_stream_mode: Literal["sync", "async"] = "sync"


bedrock_settings = BedrockSettings()
