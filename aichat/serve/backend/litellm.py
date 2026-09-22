import asyncio
import logging
import os
import re
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

import litellm
from litellm import Router, token_counter
from litellm.router import RetryPolicy
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)

from aichat.llm.base import OpenAIChatParams
from aichat.protocol.open_ai_protocol import ErrorCode, Tool
from aichat.serve.backend.base import Backend
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.server_settings import server_settings
from aichat.serve.services.bedrock import (
    add_cache_control_to_tools,
    filter_tool_call_result_pairing,
    format_tools_for_bedrock,
    get_cache_control_injection_points,
    map_tool_role_to_assistant,
    split_assistant_content_with_tool_calls,
)
from aichat.serve.services.bedrock_mantle_auth import get_bedrock_mantle_bearer_token
from aichat.serve.services.compaction_settings import compaction_settings
from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.models import ModelConfig
from aichat.serve.utils import count_tokens

# Retry only transient errors. Deterministic failures (bad request, auth, content
# policy, context window exceeded) are never retried — the same request will fail
# identically on every attempt. Setting num_retries=0 on the router means any error
# type not explicitly listed here also fails fast.
_ROUTER_RETRY_POLICY = RetryPolicy(
    BadRequestErrorRetries=0,
    AuthenticationErrorRetries=0,
    ContentPolicyViolationErrorRetries=0,
    ContextWindowExceededErrorRetries=0,
    RateLimitErrorRetries=3,
    ServiceUnavailableErrorRetries=2,
    APIConnectionErrorRetries=2,
    TimeoutErrorRetries=1,
    InternalServerErrorRetries=1,
)

litellm.json_logs = False
litellm.drop_params = True
litellm.modify_params = True
litellm.turn_off_message_logging = True
litellm.disable_aiohttp_transport = True

logger = logging.getLogger(__name__)


def _extend_litellm_httpx_client_ttl() -> None:
    """Extend litellm's httpx-client cache TTL past a pod's lifetime.

    litellm caches one httpx client per provider (TTL 3600s). On eviction the handler is
    GC'd and its __del__ closes the connection pool, cutting in-flight streaming responses
    mid-stream (BerriAI/litellm#24929). Extending the TTL beyond the pod lifetime avoids
    that. The name is bound via `from ... import` in each consuming module, so patch every
    module that reads it, not just litellm.constants.
    """
    ttl = server_settings.litellm_httpx_client_ttl_seconds

    import litellm.constants
    import litellm.llms.custom_httpx.http_handler

    # Modules that read _DEFAULT_TTL_FOR_HTTPX_CLIENTS at client-construction time.
    modules = [litellm.constants, litellm.llms.custom_httpx.http_handler]
    try:
        import litellm.llms.openai.common_utils

        modules.append(litellm.llms.openai.common_utils)
    except ImportError:
        pass

    patched = []
    for module in modules:
        if hasattr(module, "_DEFAULT_TTL_FOR_HTTPX_CLIENTS"):
            module._DEFAULT_TTL_FOR_HTTPX_CLIENTS = ttl
            patched.append(module.__name__)

    logger.info(
        "Extended litellm httpx client cache TTL to %ss in: %s",
        ttl,
        ", ".join(patched),
    )


_extend_litellm_httpx_client_ttl()

_global_router = None
_bedrock_mantle_openai_responses_lock = asyncio.Lock()


def is_bedrock_mantle_openai_responses_model(
    upstream_model: str | None, backend: str | None
) -> bool:
    """OpenAI frontier models on bedrock-mantle (openai.gpt-5.x) are Responses-API-only."""
    return (
        backend == "bedrock_mantle"
        and bool(upstream_model)
        and upstream_model.startswith("openai.")
    )


@asynccontextmanager
async def bedrock_mantle_openai_responses_routing(
    upstream_model: str | None, backend: str | None
):
    """Route chat completions through LiteLLM's OpenAI Responses bridge for mantle GPT models."""
    if not is_bedrock_mantle_openai_responses_model(upstream_model, backend):
        yield
        return
    async with _bedrock_mantle_openai_responses_lock:
        prev = litellm.route_all_chat_openai_to_responses
        litellm.route_all_chat_openai_to_responses = True
        try:
            yield
        finally:
            litellm.route_all_chat_openai_to_responses = prev


# Keys in models.json ``extra_body`` that are router-only, not per-completion.
_EXTRA_BODY_ROUTER_ONLY_KEYS = frozenset({"weight"})


def merge_model_extra_body(params: dict[str, Any], extra_body: dict[str, Any]) -> None:
    """Apply models.json ``extra_body`` fields to litellm completion params."""
    for key, value in extra_body.items():
        if key in _EXTRA_BODY_ROUTER_ONLY_KEYS:
            continue
        if key == "chat_template_kwargs" and isinstance(value, dict):
            merged = dict(params.get("chat_template_kwargs") or {})
            merged.update(value)
            params["chat_template_kwargs"] = merged
        else:
            params[key] = value


def apply_claude_upstream_sampling_params(
    upstream_model: str | None, params: dict[str, Any]
) -> None:
    """Match Claude on Bedrock: Opus/Sonnet reject sampling params entirely; other Claude models keep temperature but reject top_p."""
    if not upstream_model:
        return
    u = upstream_model.lower()
    if "opus" in u or "sonnet" in u:
        for k in ("temperature", "top_p", "top_k"):
            params.pop(k, None)
    elif "claude" in u:
        params.pop("top_p", None)


def get_global_router() -> Router:
    """
    Get or create the global Router instance.
    Called once at startup to initialize the router with all models.

    The router manages all model endpoints and handles retries automatically.
    Each model from model_settings.models becomes an entry in the router.
    """
    global _global_router

    if _global_router is None:
        logger.info("Initializing global LiteLLM Router...")

        model_list = []
        fallbacks_dict = {}

        for model_id, model_config in model_settings.models.items():
            backend = model_config.get("backend")

            if backend not in [
                "litellm",
                "vllm",
                "bedrock",
                "bedrock_mantle",
                "triton",
            ]:
                continue

            model_type = model_config.get("type", "llm")

            if model_type == "ensemble":
                for model_entry in model_config.get("models", []):
                    extra_body = {}
                    if model_entry.get("weight"):
                        extra_body["weight"] = int(model_entry.get("weight"))

                    router_entry = _build_router_entry(
                        model_entry.get("model"),
                        extra_body=extra_body if extra_body else None,
                    )
                    if router_entry:
                        router_entry["model_name"] = model_id
                        model_list.append(router_entry)
            elif model_type == "llm":
                router_entry = _build_router_entry(model_id)
                if router_entry:
                    model_list.append(router_entry)

                fallback_models = model_config.get("fallback_models", [])
                if fallback_models:
                    fallbacks_dict[model_id] = fallback_models
                    logger.info(
                        f"Model {model_id} configured with fallbacks: {fallback_models}"
                    )
            elif model_type in [
                "image_generation",
                "tts",
                "embedding",
                "classifier",
                "speech_to_text",
                "system_one",
            ]:
                router_entry = _build_router_entry(model_id)
                if router_entry:
                    model_list.append(router_entry)
            else:
                logger.error(f"Model {model_id} type {model_type} not supported")

        router_kwargs = {
            "model_list": model_list,
            "num_retries": 0,
            "retry_after": 5,
            "retry_policy": _ROUTER_RETRY_POLICY,
        }

        if fallbacks_dict:
            # litellm expects a list where each entry is a single-key dict
            # mapping one primary model to its fallback list.
            router_kwargs["fallbacks"] = [
                {model_id: fallbacks} for model_id, fallbacks in fallbacks_dict.items()
            ]
            logger.info(f"Router configured with fallbacks: {fallbacks_dict}")

        _global_router = Router(**router_kwargs)

        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.max_size_in_memory = 100
            logger.info("Capped litellm in_memory_llm_clients_cache to 100 entries")

        logger.info(f"Global Router initialized with {len(model_list)} model endpoints")

    return _global_router


def _build_router_entry(
    model_id: str, extra_body: dict | None = None
) -> dict | list[dict] | None:
    """
    Build a single router entry from model config.

    Converts our model configuration format to the Router's expected format:
    {
        "model_name": "our-model-id",
        "litellm_params": {
            "model": "hosted_vllm/upstream-model",
            "api_base": "https://...",
            "api_key": "..."
        }
    }
    """
    try:
        model_config = model_settings.models[model_id]

        upstream_model = model_config.get("upstream_model")
        api_base = model_config.get("address")
        inference_profile = model_config.get("inference_profile")

        backend = model_config.get("backend")
        litellm_model = _get_litellm_model_string(
            upstream_model, api_base, inference_profile, backend
        )

        litellm_params = {
            "model": litellm_model,
            "api_base": api_base,
        }

        if is_bedrock_mantle_openai_responses_model(upstream_model, backend):
            litellm_params["additional_drop_params"] = ["output_config"]

        # Near AI models need explicit API key in router config
        if model_id.startswith("near-"):
            litellm_params["api_key"] = external_service_settings.near_api_key

        if extra_body:
            litellm_params.update(extra_body)

        passthrough_fields = ["endpoint", "method", "api_key"]
        for field in passthrough_fields:
            if field in model_config:
                litellm_params[field] = model_config[field]

        router_entry = {"model_name": model_id, "litellm_params": litellm_params}

        # Suppress "not in built-in cost map" warnings for self-hosted vllm models
        # by declaring explicit (zero) costs so litellm skips the lookup.
        if backend in ("vllm", "triton") or (
            api_base and "hosted_vllm" in litellm_model
        ):
            router_entry["model_info"] = {
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            }

        return router_entry
    except Exception as e:
        logger.error(f"Failed to build router entry for model {model_id}: {e}")
        return None


def _get_litellm_model_string(
    upstream_model: str,
    api_base: str | None,
    inference_profile: str | None,
    backend: str | None = None,
) -> str:
    """
    Convert our model configuration to the litellm model string format.
    """
    if is_bedrock_mantle_openai_responses_model(upstream_model, backend):
        return f"openai/{upstream_model}"
    if backend == "bedrock_mantle":
        return f"bedrock_mantle/{upstream_model}"
    if inference_profile:
        profile_value = os.getenv(inference_profile) if inference_profile else None
        if profile_value:
            return f"bedrock/converse/{profile_value}"
        else:
            return f"bedrock/{upstream_model}"
    elif api_base and "near.ai" in api_base:
        return f"openai/{upstream_model}"
    else:
        return f"hosted_vllm/{upstream_model}"


class RetryCallback:
    """Minimal callback to track retries."""

    def __init__(self, model_id: str):
        self.model_id = model_id


_retry_callbacks: dict[str, RetryCallback] = {}


def _bedrock_mantle_completion_extras(backend: str | None) -> dict[str, Any]:
    if backend == "bedrock_mantle":
        return {"api_key": get_bedrock_mantle_bearer_token()}
    return {}


def get_retry_callback(model_id: str) -> RetryCallback:
    """Return a singleton RetryCallback per model to avoid accumulating
    callback references inside litellm's fire-and-forget tasks."""
    if model_id not in _retry_callbacks:
        _retry_callbacks[model_id] = RetryCallback(model_id)
    return _retry_callbacks[model_id]


class LitellmBackend(Backend):
    """
    Backend using the global litellm Router. All models share a single
    Router instance that handles retries and routing automatically.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self.config = model_config
        self.tokenizer = token_counter

        self.router = get_global_router()
        self.retry_callback = get_retry_callback(model_config.model_id)

        self._set_inference_profile()
        self._set_chat_template_kwargs()

    # Supports both streaming and non-streaming responses
    async def converse(
        self, messages: list[dict], stream: bool, params: dict | None
    ) -> Iterator[ChatCompletionChunk] | ChatCompletion:
        # Validate and clean message history whenever the call could end up
        # at Bedrock — either as the primary backend or via litellm fallback.
        # Skipping this lets malformed tool calls in the conversation history
        # crash Bedrock's tool-call converter during fallback.
        if (
            self.config.has_bedrock_fallback()
            or self.config.backend == "bedrock_mantle"
        ):
            messages = filter_tool_call_result_pairing(
                messages, model_id=self.config.model_id
            )
            if getattr(self.config, "tool_role_as_assistant", False):
                messages = map_tool_role_to_assistant(messages)
            else:
                messages = split_assistant_content_with_tool_calls(messages)
        completion_params = {
            "model": self.config.model_id,
            "messages": messages,
            "stream": stream,
            **params,
        }
        completion_params.update(_bedrock_mantle_completion_extras(self.config.backend))

        try:
            async with bedrock_mantle_openai_responses_routing(
                self.config.upstream_model, self.config.backend
            ):
                response = await self.router.acompletion(
                    **completion_params, callbacks=[self.retry_callback]
                )
            return response
        except Exception as e:
            return handle_litellm_error(
                e, is_streaming=stream, model=self.config.model_id
            )

    def build_params(
        self,
        stream: bool,
        tools: list[Tool],
        messages: list[dict],
        context_window_override: int | None = None,
        enable_prompt_caching: bool | None = None,
    ) -> dict:
        """Build the parameters for the router completion call.

        ``context_window_override`` lets callers (e.g. compaction) use the
        model's real hardware context window instead of the business-level
        ``conversation_token_limit*`` cap when computing remaining headroom
        for ``max_tokens``.
        """
        # Start with base parameters
        params = OpenAIChatParams().model_dump()

        if self.config.extra_body:
            merge_model_extra_body(params, self.config.extra_body)

        # Callers may override the model's default caching behavior in either
        # direction, but only for models the config marks as cache-capable.
        should_use_prompt_caching = bool(self.config.prompt_caching_support) and (
            bool(self.config.prompt_caching_enabled)
            if enable_prompt_caching is None
            else enable_prompt_caching
        )

        # Add tools and tool_choice if available
        tools_cached = False
        if self.config.tool_support and tools and isinstance(tools, list):
            params["tool_choice"] = "auto"
            if self.config.backend == "bedrock":
                bedrock_tools = format_tools_for_bedrock(tools)
                # Cache the (static) tools prefix when the model supports it.
                if should_use_prompt_caching:
                    bedrock_tools = add_cache_control_to_tools(bedrock_tools)
                    tools_cached = True
                params["tools"] = bedrock_tools
            else:
                params["tools"] = [
                    t.model_dump() if hasattr(t, "model_dump") else t for t in tools
                ]

        if stream:
            params["stream_options"] = {"include_usage": True}

        if self.config.reasoning_effort:
            params["reasoning_effort"] = self.config.reasoning_effort

        apply_claude_upstream_sampling_params(self.config.upstream_model, params)
        # litellm's Router runs fallbacks (transitively) with these same params,
        # so they must also satisfy any Claude-on-Bedrock model reachable via the
        # fallback chain (Haiku rejects top_p; Opus/Sonnet reject all sampling
        # params).
        seen: set[str] = set()
        queue = list(self.config.fallback_models)
        while queue:
            fallback_id = queue.pop(0)
            if fallback_id in seen:
                continue
            seen.add(fallback_id)
            fallback_cfg = model_settings.models.get(fallback_id)
            if not fallback_cfg:
                continue
            apply_claude_upstream_sampling_params(
                fallback_cfg.get("upstream_model"), params
            )
            # Drop reasoning_effort if any fallback in the chain doesn't support it.
            if params.get("reasoning_effort") and not fallback_cfg.get(
                "reasoning_effort"
            ):
                params.pop("reasoning_effort")
            queue.extend(fallback_cfg.get("fallback_models", []) or [])

        if should_use_prompt_caching:
            # Reserve a checkpoint for the tools cachePoint (set on the tools
            # list above) so the total never exceeds MAX_CACHE_POINTS.
            cache_injection_points = get_cache_control_injection_points(
                messages, reserved_points=1 if tools_cached else 0
            )
            if cache_injection_points:
                params["cache_control_injection_points"] = cache_injection_points

        # Match calculate_max_input_tokens in compaction: reserve output budget explicitly
        # so vLLM does not apply a large default (e.g. 8192) that exceeds remaining context.
        configured_max = (
            self.config.max_tokens
            or compaction_settings.compaction_fallback_output_tokens
        )
        params["max_tokens"] = configured_max

        context_window = context_window_override or (
            self.config.conversation_token_limit_premium
            or self.config.conversation_token_limit
        )
        if context_window and params.get("max_tokens"):
            input_tokens = count_tokens(messages)
            available = context_window - input_tokens
            if available < params["max_tokens"]:
                capped = max(available, server_settings.min_output_tokens)
                params["max_tokens"] = capped

        return params

    def _set_inference_profile(self) -> str | None:
        inference_profile_key = self.config.inference_profile
        self.inference_profile = (
            os.getenv(inference_profile_key) if inference_profile_key else None
        )

    def _set_chat_template_kwargs(self) -> dict | None:
        self.chat_template_kwargs = {}
        if self.config.extra_body:
            self.chat_template_kwargs = self.config.extra_body.get(
                "chat_template_kwargs"
            )


def create_error_response(error_type: str, status_code: int) -> dict:
    return {
        "type": "error",
        "content": error_type,
        "code": status_code,
    }


async def yield_error_response(error_response: dict, model: str | None = None):
    """Async generator that yields a single error response as a ChatCompletionChunk."""
    # Convert error dict to ChatCompletionChunk for streaming
    error_chunk = ChatCompletionChunk(
        id="error",
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(
                    content=error_response["content"],
                    role="assistant",
                ),
                finish_reason="stop",
            )
        ],
    )
    logger.error(f"Model: {model}, Error: {error_response['content']}")
    yield error_chunk


def extract_message_from_bad_request_error(error: litellm.BadRequestError) -> str:
    error_str = str(error)

    json_match = re.search(r'\{[^}]*"message"\s*:\s*"([^"]*)"', error_str)
    if json_match:
        message = json_match.group(1)
        if "Already borrowed" in message:
            return "The model is temporarily busy. Please try again."
        return message

    if "Hosted_vllmException" in error_str:
        msg = error_str.split("Hosted_vllmException")[1]
        if "Unknown part type" in msg:
            return "This model does not support the file(s) attached in the conversation. Please try a different model."
        return msg

    if "BedrockException" in error_str:
        if "length limit exceeded" in error_str:
            return "Your request is too large."
        return error_str.split("BedrockException")[1]

    return error_str


def handle_litellm_error(
    error: Exception, is_streaming: bool = False, model: str | None = None
):
    """
    Unified error handler for litellm exceptions.

    Returns an error response dict (non-streaming) or an async generator
    that yields the error response (streaming).
    """
    if isinstance(error, litellm.ContextWindowExceededError):
        error_response = create_error_response(
            "Conversation length exceeded. Please shorten your request or start a new conversation.",
            ErrorCode.EXCEEDED_CONTEXT_LENGTH,
        )

    elif isinstance(
        error, (litellm.APIConnectionError, litellm.ServiceUnavailableError)
    ):
        error_response = create_error_response(
            "There was an issue connecting to the model.",
            ErrorCode.API_CONNECTION_ERROR,
        )

    elif isinstance(error, litellm.BadRequestError):
        error_response = create_error_response(
            extract_message_from_bad_request_error(error),
            ErrorCode.BAD_REQUEST_ERROR,
        )
    else:
        logger.error(f"Uncaught error: {str(error)[:400]}")
        error_response = create_error_response(
            "There was an issue connecting to the model.", ErrorCode.INTERNAL_ERROR
        )

    return (
        yield_error_response(error_response, model) if is_streaming else error_response
    )
