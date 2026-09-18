from pydantic_settings import BaseSettings


class ServerSettings(BaseSettings):
    log_level: str = "WARNING"
    env: str
    placeholder_model: str = " "

    # Sized above asyncio's default so slow synchronous calls on the shared default
    # executor can't starve other requests' completion() setup.
    litellm_executor_max_workers: int = 64

    # Overrides litellm's hardcoded 1-hour httpx-client cache TTL (see litellm.py). 3 days.
    litellm_httpx_client_ttl_seconds: int = 60 * 60 * 24 * 3

    # Minimum output token budget passed to the model. Prevents max_tokens from being
    # capped to near-zero when the conversation is long, which would cause models to
    # truncate tool call arguments mid-JSON. When available headroom is less than this
    # floor, the model will reject with a context-window error (surfaced as a clean
    # user-facing message) rather than silently truncating output.
    min_output_tokens: int = 512

    api_key_chat_enabled: bool = True
    # When a model has truncation_continuation in models.json: one follow-up LLM
    # turn if finish_reason=stop but the reply does not end like a full sentence.
    truncation_continuation_max_depth: int = 2


server_settings = ServerSettings()
