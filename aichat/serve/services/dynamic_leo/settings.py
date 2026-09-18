import json
from typing import Any

from pydantic_settings import BaseSettings

DYNAMIC_LEO_CONFIG_DEFAULT = "{}"


class DynamicLeoSettings(BaseSettings):
    enable_dynamic_leo: bool = True
    dynamic_leo_config: dict[str, Any] = json.loads(DYNAMIC_LEO_CONFIG_DEFAULT)
    dynamic_leo_last_n_user_turns: int = 5
    dynamic_leo_embedding_similarity_threshold: float = 0.72
    dynamic_leo_embedding_model: str = ""
    dynamic_leo_embedding_timeout_seconds: float = 1.0
    dynamic_leo_androcles_timeout_seconds: float = 1.0


dynamic_leo_settings = DynamicLeoSettings()
