from pydantic_settings import BaseSettings


class ApiKeyChatSettings(BaseSettings):
    api_key_prefixes: list[str] = ["test_key_"]


api_key_chat_settings = ApiKeyChatSettings()
