import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, RootModel

logger = logging.getLogger(__name__)


class DynamicLeoCategoryConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    keywords: list[str] = []
    similar: list[str] = []
    labels: dict[str, float] = {}


class DynamicLeoConfigFile(RootModel[dict[str, DynamicLeoCategoryConfig]]):
    pass


def parse_dynamic_leo_categories(
    raw: dict[str, Any],
) -> dict[str, DynamicLeoCategoryConfig] | None:
    if not raw:
        return None
    try:
        parsed = DynamicLeoConfigFile.model_validate(raw)
        logger.debug("Parsed Dynamic Leo config (%d categories)", len(parsed.root))
        return parsed.root
    except Exception as exc:
        logger.warning("Could not parse Dynamic Leo config: %s", exc)
        return None
