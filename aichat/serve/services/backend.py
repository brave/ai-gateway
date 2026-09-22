import functools
import logging

from aichat.serve.backend.base import Backend
from aichat.serve.backend.litellm import LitellmBackend, get_global_router
from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.models import ModelConfig, get_model_config

logger = logging.getLogger(__name__)


def initialize_backends() -> None:
    """
    Initialize the global Router and pre-warm backend cache for all
    configured models. This ensures any configuration errors are caught
    early and first requests are fast.
    """
    get_global_router()

    logger.info("Initializing backends for all configured models...")
    successful_count = 0
    for model_id in model_settings.models:
        model_config = model_settings.models[model_id]
        model_type = model_config.get("type", "llm")
        if model_type in [
            "image_generation",
            "tts",
            "embedding",
            "classifier",
            "speech_to_text",
            "system_one",
        ]:
            continue
        try:
            get_backend(model_id)
            successful_count += 1
        except Exception as e:
            logger.error(f"Failed to initialize backend for model {model_id}: {e}")

    logger.info(
        f"Successfully initialized "
        f"{successful_count}/{len(model_settings.models)} backends"
    )


@functools.cache
def get_backend(model_id: str) -> Backend:
    model_config: ModelConfig = get_model_config(model_id)

    if model_config.backend in ["litellm", "bedrock", "bedrock_mantle", "vllm"]:
        return LitellmBackend(model_config)
    else:
        raise ValueError(
            f"Unsupported backend '{model_config.backend}' " f"for model {model_id}"
        )


def clear_backend_cache() -> None:
    get_backend.cache_clear()
