from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    vllm_api_key: str = "empty"
    use_vllm_chat_api: bool = False
    default_tokenizer: str = "o200k_base"


llm_settings = LLMSettings()
