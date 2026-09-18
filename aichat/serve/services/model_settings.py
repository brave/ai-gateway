import json
import logging
from typing import Any

from pydantic_settings import BaseSettings

MODELS_DEFAULT = "{}"
MODEL_TRIAGING_DEFAULT = "{}"

logger = logging.getLogger(__name__)


class ModelSettings(BaseSettings):
    models: dict[str, Any] = json.loads(MODELS_DEFAULT)
    model_triaging: dict[str, Any] = json.loads(MODEL_TRIAGING_DEFAULT)
    content_agent_default_model: str = "claude-3-sonnet"

    def model_post_init(self, __context: Any, /) -> None:
        if self.model_triaging and "default" not in self.model_triaging:
            logger.warning(
                'MODEL_TRIAGING is configured but missing a "default" entry; '
                "automatic mode will fall back to an arbitrary configured model "
                "for uncovered categories."
            )


model_settings = ModelSettings()
