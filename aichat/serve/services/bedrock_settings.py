from pydantic_settings import BaseSettings


class BedrockSettings(BaseSettings):
    bedrock_cache_min_tokens: int = 4096


bedrock_settings = BedrockSettings()
