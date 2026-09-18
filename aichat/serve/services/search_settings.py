from pydantic_settings import BaseSettings

RICH_SEARCH_VERTICALS = [
    "calculator",
    "unitconversion",
    "weather",
    "unixtimestamp",
    "currency",
    "stocks",
    "definitions",
    "sports",
    "cryptocurrency",
    "packagetracker",
    "formula1",
]


class SearchSettings(BaseSettings):
    brave_search_api_url: str = "https://api.search.brave.com"
    search_enabled: bool = False
    brave_search_api_key: str = ""
    brave_search_rh_api_key: str = ""
    function_calling_enabled: bool = False
    use_content_search: bool = False
    max_inline_searches: int = 10
    max_function_calls: int = 1
    max_queries_per_round: int = 2
    conversation_limit_for_search_augmentation: int = 4
    search_api_allowed_cors_origins: str = ""


search_settings = SearchSettings()
