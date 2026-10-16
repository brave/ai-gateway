import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Coroutine
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from aichat.llm.metrics import API_KEY_USAGE_SETTLE_TOTAL, API_KEY_USAGE_TOKENS_TOTAL
from aichat.serve import internal_client
from aichat.serve.services.models import ModelConfig

logger = logging.getLogger(__name__)

USAGE_SHUTDOWN_TIMEOUT_SECONDS = 20
STREAM_BUFFER_SIZE = 32
STREAM_END = object()
_background_tasks: set[asyncio.Task] = set()


@dataclass(frozen=True)
class UsageContext:
    httpx_client: httpx.AsyncClient
    api_key: str
    transaction_id: str
    model: str
    provider: str


@dataclass(frozen=True)
class TokenUsageCounts:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0


def usage_provider(config: ModelConfig) -> str:
    if config.backend in ("bedrock", "bedrock_mantle"):
        return "bedrock"
    if config.api_base and "near.ai" in config.api_base:
        return "near"
    return "self_hosted"


def new_usage_context(
    httpx_client: httpx.AsyncClient, api_key: str, model: str, config: ModelConfig
) -> UsageContext:
    return UsageContext(
        httpx_client=httpx_client,
        api_key=api_key,
        transaction_id=f"txn_{uuid.uuid4().hex}",
        model=model,
        provider=usage_provider(config),
    )


def set_token_usage_counts(usage: Any) -> TokenUsageCounts | None:
    if usage is None:
        return None
    data = usage if isinstance(usage, dict) else usage.model_dump()
    prompt_details = data.get("prompt_tokens_details") or {}
    completion_details = data.get("completion_tokens_details") or {}
    prompt_tokens = int(data.get("prompt_tokens") or 0)
    completion_tokens = int(data.get("completion_tokens") or 0)
    cache_write_tokens = int(
        data.get("cache_creation_input_tokens")
        or prompt_details.get("cache_creation_tokens")
        or 0
    )
    cache_read_tokens = int(
        data.get("cache_read_input_tokens") or prompt_details.get("cached_tokens") or 0
    )
    reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
    return TokenUsageCounts(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
        reasoning_tokens=reasoning_tokens,
    )


# Makes request to internal_client.usage_settle to settle the token usage for
# the given api key and model.
async def settle_usage(ctx: UsageContext, usage: Any) -> None:
    try:
        counts: TokenUsageCounts | None = set_token_usage_counts(usage)
    except Exception:
        logger.exception("Failed to read usage for %s", ctx.transaction_id)
        counts = None

    if counts is None:
        API_KEY_USAGE_SETTLE_TOTAL.labels(outcome="missing_usage").inc()
        logger.error("No usage reported for %s", ctx.transaction_id)
        return

    token_counts: dict[str, int] = asdict(counts)
    for token_type, value in token_counts.items():
        if value:
            API_KEY_USAGE_TOKENS_TOTAL.labels(
                model=ctx.model, token_type=token_type
            ).inc(value)

    result = await internal_client.usage_settle(
        ctx.httpx_client,
        api_key=ctx.api_key,
        transaction_id=ctx.transaction_id,
        model=ctx.model,
        provider=ctx.provider,
        prompt_tokens=counts.prompt_tokens,
        completion_tokens=counts.completion_tokens,
        cache_write_tokens=counts.cache_write_tokens,
        cache_read_tokens=counts.cache_read_tokens,
        reasoning_tokens=counts.reasoning_tokens,
    )
    if result is None:
        API_KEY_USAGE_SETTLE_TOTAL.labels(outcome="unavailable").inc()
        logger.error("Usage settle failed for %s", ctx.transaction_id)
        return
    API_KEY_USAGE_SETTLE_TOTAL.labels(outcome="settled").inc()


async def converse_and_settle(converse: Awaitable, ctx: UsageContext | None) -> Any:
    response = await converse
    if ctx is not None and not isinstance(response, dict):
        spawn(settle_usage(ctx, getattr(response, "usage", None)))
    return response


async def read_stream_and_settle(
    response: AsyncIterator,
    chunks: asyncio.Queue,
    disconnected: asyncio.Event,
    ctx: UsageContext | None,
) -> None:
    usage = None
    try:
        try:
            async for chunk in response:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not disconnected.is_set():
                    await chunks.put(chunk)
        except Exception as e:
            if not disconnected.is_set():
                await chunks.put(e)
        if not disconnected.is_set():
            await chunks.put(STREAM_END)
    finally:
        if ctx is not None:
            await settle_usage(ctx, usage)


# The settle has to run as a detached background task, and Python needs a strong
# reference to that task so it isn't garbage-collected before it finishes.
def spawn(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def wait_for_background_tasks(timeout: float) -> None:
    if not _background_tasks:
        return
    _, pending = await asyncio.wait(set(_background_tasks), timeout=timeout)
    if pending:
        logger.warning("%d usage tasks still running at shutdown", len(pending))
