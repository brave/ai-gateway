from pydantic_settings import BaseSettings


class ApiKeyChatSettings(BaseSettings):
    api_key_skv2_verification_enabled: bool = True


api_key_chat_settings = ApiKeyChatSettings()
