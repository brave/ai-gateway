import logging

from pydantic_settings import BaseSettings

from aichat.serve.constants import PRODUCTION
from aichat.serve.server_settings import server_settings

logger = logging.getLogger(__name__)


class SecuritySettings(BaseSettings):
    internal_models_api_key: str = ""
    alignment_checking_enabled: bool = True
    alignment_checking_model: str = "llama-4-maverick"
    alignment_truncation_char_limit: int = 1000
    prompt_injection_scanning_enabled: bool = True
    prompt_injection_scanning_model: str = "llama-4-maverick"
    allowed_bypass_tools: dict[str, list[str]] = {
        "alignment_check_bypass": [
            "user_choice_tool",
            "scroll_element",
            "wait",
            "assistant_detail_storage",
            "move_mouse",
            "brave_web_search",
            "brave_news_search",
            "brave_faqs_search",
            "answer_brave_related_questions",
            "code_execution_tool",
            "semantic_history_search",
        ],
        "tool_result_bypass": [
            "user_choice_tool",
            "wait",
            "assistant_detail_storage",
            "code_execution_tool",
        ],
    }


security_settings = SecuritySettings()


def check_internal_models_api_key() -> None:
    if security_settings.internal_models_api_key:
        return
    if server_settings.env == PRODUCTION:
        raise RuntimeError("INTERNAL_MODELS_API_KEY must be set when ENV=production")
    logger.warning(
        "INTERNAL_MODELS_API_KEY is unset; internal model routes are unauthenticated"
    )
