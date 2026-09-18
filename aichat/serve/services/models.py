import functools
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from aichat.serve.services.model_settings import model_settings


@dataclass
class ModelConfig:
    """Configuration for a model from model_settings."""

    model_id: str
    upstream_model: str
    backend: str
    api_base: str | None
    api_key: str | None

    # Bedrock-specific properties (only for bedrock backend)
    inference_profile: str | None
    system_prompt_support: bool | None
    prompt_caching_support: bool | None
    prompt_caching_enabled: bool | None

    # Model capabilities
    tool_support: bool
    image_support: bool
    audio_support: bool
    video_support: bool
    file_support: bool

    # Model metadata
    friendly_name: str | None
    maker: str | None
    max_tokens: int | None
    max_tokens_premium: int | None
    conversation_token_limit: int | None
    conversation_token_limit_premium: int | None
    free: bool
    key: str | None
    rate_limit: int | None
    rate_limit_interval_seconds: int | None

    # Page limits
    max_pages: int | None
    max_pages_premium: int | None

    # Additional configuration
    extra_body: dict | None
    training_cutoff: str | None = None  # YYYY-MM-DD, for Leo system prompt
    reasoning_effort: str | None = None
    tool_role_as_assistant: bool = (
        False  # If True, map message role "tool" -> "assistant"
    )
    fallback_models: list[str] = field(default_factory=list)
    deep_research_support: bool = False
    content_agent_support: bool = False
    e2ee_support: bool = False
    truncation_continuation: bool = False
    # Token amount tier sent to MCP servers: premium, freemium, or free.
    token_amount_tier: str = "free"

    def to_dict(self) -> dict:
        """Convert ModelConfig to dictionary for backward compatibility."""

        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def has_bedrock_fallback(self) -> bool:
        """
        True when this model is itself a Bedrock backend or any of its
        configured fallback models is a Bedrock backend. Used to decide
        whether to sanitize conversation history before sending it to
        litellm (which otherwise crashes the Bedrock converter on malformed
        tool calls during fallback).
        """
        if self.backend in ("bedrock", "bedrock_mantle"):
            return True
        for fallback_id in self.fallback_models:
            fallback_settings = model_settings.models.get(fallback_id)
            if fallback_settings and fallback_settings.get("backend") in (
                "bedrock",
                "bedrock_mantle",
            ):
                return True
        return False


@functools.cache
def get_model_config(model_id: str) -> ModelConfig | None:
    """Get ModelConfig for a model, with caching for performance."""
    model_cfg = model_settings.models[model_id]
    model_type = model_cfg.get("type", "llm")

    if model_type not in ["llm", "ensemble"]:
        return None

    return ModelConfig(
        model_id=model_id,
        upstream_model=model_cfg.get("upstream_model"),
        backend=model_cfg.get("backend"),
        api_base=model_cfg.get("address"),
        api_key=model_cfg.get("api_key"),
        # Bedrock-specific (only present for bedrock models)
        inference_profile=model_cfg.get("inference_profile"),
        system_prompt_support=model_cfg.get("system_prompt_support"),
        prompt_caching_support=model_cfg.get("prompt_caching_support"),
        prompt_caching_enabled=model_cfg.get("prompt_caching_enabled"),
        tool_role_as_assistant=model_cfg.get("tool_role_as_assistant", False),
        tool_support="tools" in model_cfg.get("capabilities", []),
        image_support="vision" in model_cfg.get("capabilities", []),
        audio_support="audio" in model_cfg.get("capabilities", []),
        video_support="video" in model_cfg.get("capabilities", []),
        file_support="files" in model_cfg.get("capabilities", []),
        deep_research_support="deep_research" in model_cfg.get("capabilities", []),
        content_agent_support="content_agent" in model_cfg.get("capabilities", []),
        # Model metadata
        friendly_name=model_cfg.get("friendly_name"),
        maker=model_cfg.get("maker"),
        max_tokens=model_cfg.get("max_tokens"),
        max_tokens_premium=model_cfg.get("max_tokens_premium"),
        conversation_token_limit=model_cfg.get("conversation_token_limit"),
        conversation_token_limit_premium=model_cfg.get(
            "conversation_token_limit_premium"
        ),
        free=model_cfg.get("free", True),
        key=model_cfg.get("key"),
        # Rate limiting
        rate_limit=model_cfg.get("rate_limit"),
        rate_limit_interval_seconds=model_cfg.get("rate_limit_interval_seconds"),
        # Page limits
        max_pages=model_cfg.get("max_pages"),
        max_pages_premium=model_cfg.get("max_pages_premium"),
        # Additional configuration
        extra_body=model_cfg.get("extra_body"),
        training_cutoff=model_cfg.get("training_cutoff"),
        reasoning_effort=model_cfg.get("reasoning_effort"),
        fallback_models=list(model_cfg.get("fallback_models", []) or []),
        e2ee_support=model_cfg.get("e2ee_support", False),
        truncation_continuation=bool(model_cfg.get("truncation_continuation")),
        token_amount_tier=model_cfg.get("token_amount_tier", "free"),
    )


@functools.cache
def _get_upstream_to_model_id_mapping() -> dict[str, str]:
    """
    Create a mapping from upstream model names to model IDs.
    This allows us to map LiteLLM's returned model names back to our friendly model IDs.
    """
    mapping = {}
    for model_id, config in model_settings.models.items():
        if config.get("type", "llm") != "llm":
            continue

        if config.get("backend") == "bedrock":
            inference_profile = config.get("inference_profile")
            profile_value = os.getenv(inference_profile) if inference_profile else None
            if profile_value:
                mapping[profile_value] = model_id
        elif config.get("backend") == "bedrock_mantle":
            upstream_model = config.get("upstream_model")
            if upstream_model:
                mapping[upstream_model] = model_id
        else:
            upstream_model = config.get("upstream_model")
            if upstream_model:
                mapping[upstream_model] = model_id
    return mapping


@functools.cache
def _get_model_id_to_bedrock_profile() -> dict[str, str]:
    """Model ID -> Bedrock profile value. Used when multiple model_ids share the same profile."""
    out = {}
    for model_id, config in model_settings.models.items():
        if config.get("type", "llm") != "llm" or config.get("backend") != "bedrock":
            continue
        inference_profile = config.get("inference_profile")
        profile_value = os.getenv(inference_profile) if inference_profile else None
        if profile_value:
            out[model_id] = profile_value
    return out


def normalize_model_name(
    litellm_model: str, requested_model_id: str | None = None
) -> str | None:
    """
    Convert LiteLLM's model string to our friendly model ID.
    When requested_model_id is set and the mapping returns a different model_id that
    shares the same Bedrock profile (e.g. claude-haiku vs claude-3-haiku), returns
    requested_model_id so the stream matches what the user asked for.
    """
    if not litellm_model:
        return None

    mapping = _get_upstream_to_model_id_mapping()

    resolved = None
    if litellm_model in mapping:
        resolved = mapping[litellm_model]
    elif "/" in litellm_model:
        parts = litellm_model.split("/", 1)
        if len(parts) > 1 and parts[1] in mapping:
            resolved = mapping[parts[1]]
    if resolved is None:
        model_name = litellm_model.split("/")[-1]
        resolved = mapping.get(model_name, model_name)

    if requested_model_id and resolved != requested_model_id:
        profiles = _get_model_id_to_bedrock_profile()
        req_profile = profiles.get(requested_model_id)
        if req_profile is not None and req_profile == profiles.get(resolved):
            return requested_model_id
    return resolved
