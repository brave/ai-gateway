from pydantic_settings import BaseSettings


class ConversationSettings(BaseSettings):
    conversation_api: bool = False
    conversation_title_enabled: bool = True
    title_model: str = ""
    max_conversation_rounds: int = 250
    temperature: float = 0.5
    max_tokens: int = 1600
    max_tokens_premium: int = 2400
    max_context_chars_for_title_generation: int = 1200
    conversation_token_limit_for_long_context: int = 6400
    image_token_estimate: int = 0
    video_token_estimate: int = 0
    audio_token_estimate: int = 0
    file_token_estimate: int = 0


conversation_settings = ConversationSettings()
