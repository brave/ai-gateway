from pydantic_settings import BaseSettings


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
