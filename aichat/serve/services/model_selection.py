import logging
import os
import random

import httpx

from aichat.prompts.filter_tabs import TYPE as FILTER_TABS_TYPE
from aichat.prompts.reduce_focus_topics import TYPE as REDUCE_FOCUS_TOPICS_TYPE
from aichat.prompts.request_summary import TYPE as REQUEST_SUMMARY_TYPE
from aichat.prompts.suggest_focus_topics import TYPE as SUGGEST_FOCUS_TOPICS_TYPE
from aichat.prompts.suggest_focus_topics_with_emoji import (
    TYPE as SUGGEST_FOCUS_TOPICS_EMOJI_TYPE,
)
from aichat.protocol.open_ai_protocol import (
    Capability,
    CapabilityOptions,
    Message,
    has_capability,
)
from aichat.serve.androcles import (
    androcles_inference,
    task_type_from_androcles_probabilities,
)
from aichat.serve.conversation_settings import conversation_settings
from aichat.serve.rate_limiting import check_and_increment_automatic_mode_daily_count
from aichat.serve.services.androcles_prefetch import AndroclesPrefetch
from aichat.serve.services.dynamic_leo.settings import dynamic_leo_settings
from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.models import get_model_config

logger = logging.getLogger(__name__)


async def select_model_for_request(
    model: str,
    messages: list[Message],
    brave_capability: CapabilityOptions = None,
    is_premium: bool = False,
    last_user_message_content: str | None = None,
    media_type: str | None = None,
    rate_key: str | None = None,
    httpx_client: httpx.AsyncClient | None = None,
    androcles_prefetch: AndroclesPrefetch | None = None,
) -> str:
    if has_capability(brave_capability, Capability.content_agent):
        model_cfg = get_model_config(model) if model in model_settings.models else None
        return (
            model
            if model_cfg and model_cfg.content_agent_support
            else model_settings.content_agent_default_model
        )

    if model != "automatic":
        if model not in model_settings.models:
            logger.error(f"Model {model} is not supported defaulting to automatic")
            model = "automatic"
        else:
            if "models" in model_settings.models[model]:
                return _resolve_weighted_model(model)
            return model

    model_mapping = await triage_request(
        messages,
        last_user_message_content,
        media_type,
        androcles_prefetch=androcles_prefetch,
    )

    treat_as_premium = is_premium
    if not is_premium and rate_key:
        treat_as_premium = await check_and_increment_automatic_mode_daily_count(
            rate_key, httpx_client=httpx_client
        )

    selected = (
        model_mapping["premium"] if treat_as_premium else model_mapping["non-premium"]
    )
    if (
        selected in model_settings.models
        and "models" in model_settings.models[selected]
    ):
        return _resolve_weighted_model(selected)
    return selected


def _resolve_weighted_model(model: str) -> str:
    """Resolve a weighted model ensemble config to a concrete model via random selection.

    Callers must verify that `model_settings.models[model]["models"]` exists
    before calling this function.
    """
    models = model_settings.models[model]["models"]
    model_ids = [m["model"] for m in models]
    model_weights = [m["weight"] for m in models]
    resolved = random.choices(model_ids, model_weights, k=1)[0]
    return resolved


TAB_FOCUS_TYPES = {
    FILTER_TABS_TYPE,
    REDUCE_FOCUS_TOPICS_TYPE,
    SUGGEST_FOCUS_TOPICS_TYPE,
    SUGGEST_FOCUS_TOPICS_EMOJI_TYPE,
}


def _last_user_message_has_content_type(
    messages: list[Message], types: set[str] | str
) -> bool:
    """True if the last user message contains a content part matching the given type(s)."""
    if isinstance(types, str):
        types = {types}
    for msg in reversed(messages):
        if getattr(msg, "role", None) != "user":
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            return False
        return any(getattr(part, "type", None) in types for part in content)
    return False


def _get_triage_models(category: str) -> dict[str, str]:
    """Look up a triage mapping for `category`, falling back to "default",
    then to the first configured model if MODEL_TRIAGING has no usable
    entries at all."""
    mapping = model_settings.model_triaging.get(
        category
    ) or model_settings.model_triaging.get("default")
    if mapping:
        return mapping
    if model_settings.models:
        first_model = next(iter(model_settings.models))
        logger.error(
            'MODEL_TRIAGING has no entry for %r or "default"; falling back to '
            "arbitrary model %r. This is a misconfiguration.",
            category,
            first_model,
        )
        return {"premium": first_model, "non-premium": first_model}
    raise RuntimeError(
        "No MODEL_TRIAGING or MODELS configured; cannot select a model for "
        "automatic mode"
    )


async def triage_request(
    messages: list[Message],
    last_user_message_content: str | None = None,
    media_type: str | None = None,
    androcles_prefetch: AndroclesPrefetch | None = None,
) -> dict[str, str]:
    # We check for summary requests first as if there is an image this will be
    # overridden by the media type check
    brave_summary_enabled = os.environ.get("ENABLE_BRAVE_SUMMARY", "").lower() in (
        "true",
        "1",
        "yes",
    )

    if brave_summary_enabled and _last_user_message_has_content_type(
        messages, REQUEST_SUMMARY_TYPE
    ):
        return _get_triage_models("summary")

    if media_type:
        return _get_triage_models(media_type)

    if _last_user_message_has_content_type(messages, TAB_FOCUS_TYPES):
        return _get_triage_models("tab_focus")

    if await is_long_context(messages):
        return _get_triage_models("long_context")

    if androcles_prefetch is not None:
        if androcles_prefetch.task_type:
            return _get_triage_models(androcles_prefetch.task_type)
        return _get_triage_models("default")

    if last_user_message_content:
        task_type = await classify_task_with_androcles(last_user_message_content)
        if task_type:
            return _get_triage_models(task_type)

    return _get_triage_models("default")


async def is_long_context(messages: list[Message]) -> bool:
    total_chars = 0

    # Loop through messages once and accumulate text length
    for msg in messages:
        text_content = ""

        if isinstance(msg.content, str):
            text_content = msg.content
        elif isinstance(msg.content, list):
            text_parts = []
            for part in msg.content:
                if hasattr(part, "text") and part.text is not None:
                    text_parts.append(part.text)
            text_content = " ".join(text_parts)
        else:
            text_content = str(msg.content)

        if text_content.strip():
            total_chars += len(text_content)

    estimated_tokens = total_chars // 3

    logger.debug(
        f"Long context check: {estimated_tokens} estimated tokens "
        f"(threshold: {conversation_settings.conversation_token_limit_for_long_context})"
    )
    return (
        estimated_tokens
        > conversation_settings.conversation_token_limit_for_long_context
    )


async def classify_task_with_androcles(content: str) -> str | None:
    probabilities = await androcles_inference(
        content,
        timeout_seconds=dynamic_leo_settings.dynamic_leo_androcles_timeout_seconds,
    )
    return task_type_from_androcles_probabilities(probabilities)
