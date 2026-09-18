import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from aichat.serve.services.mcp.mcp_settings import mcp_settings

logger = logging.getLogger(__name__)


async def iter_events_with_keepalive(
    agen: AsyncGenerator[dict[str, Any], None],
    keepalive_interval: float = 20.0,
) -> AsyncGenerator[dict[str, Any], None]:
    """Wrap a deep-research event generator with keepalive ping injection.

    Forwards all events from *agen* unchanged.  If no event arrives within
    *keepalive_interval* seconds a synthetic ``{"event": "ping"}`` is yielded
    so that intermediate proxies (AWS ALB default idle-timeout is 60 s) do not
    close the streaming connection during long LLM calls.

    Uses an asyncio.Queue so that the underlying generator keeps running
    independently; ``asyncio.wait_for`` cancels only the *queue.get()* wait,
    never the generator itself.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    _DONE = object()

    async def _producer() -> None:
        try:
            async for event in agen:
                await queue.put(event)
        except Exception as exc:
            await queue.put(exc)
        finally:
            await queue.put(_DONE)

    producer_task = asyncio.create_task(_producer())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=keepalive_interval)
            except TimeoutError:
                yield {"event": "ping"}
                continue

            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("producer task ended with error", exc_info=True)


class DeepResearchStreamingExecutor:
    """Executor for deep research that supports streaming JSONL responses."""

    def __init__(
        self,
        deep_research_url: str | None = None,
        max_iterations: int | None = None,
        max_queries: int | None = None,
        max_seconds: int | None = None,
    ):
        """Initialize deep research streaming executor.

        Args:
            deep_research_url: URL of the deep research service
            max_iterations: Maximum research iterations
            max_queries: Maximum total queries
            max_seconds: Maximum time budget in seconds
        """
        self.deep_research_url = deep_research_url or mcp_settings.deep_research_url
        self.max_iterations = (
            max_iterations or mcp_settings.deep_research_max_iterations
        )
        self.max_queries = max_queries or mcp_settings.deep_research_max_queries
        self.max_seconds = max_seconds or mcp_settings.deep_research_max_seconds

    async def execute_streaming(
        self,
        query: str,
        country: str = "us",
        language: str = "en",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Execute deep research with streaming response.

        Args:
            query: The research question
            country: Search country code
            language: Search language

        Yields:
            Deep research event dictionaries
        """
        if not mcp_settings.deep_research_enabled:
            logger.error(
                "Deep research tool was invoked but deep_research_enabled=False. "
                "Enable the service or check capability gating."
            )
            yield {
                "type": "error",
                "error": "Deep research is not enabled on this server.",
            }
            return

        request_body = {
            "query": query,
            "country": country,
            "language": language,
            "config": {
                "maxIterations": self.max_iterations,
                "maxQueries": self.max_queries,
                "maxSeconds": self.max_seconds,
                "enableThinking": True,
                "enableCitations": True,
            },
        }

        url = f"{self.deep_research_url}/v1/research/stream"
        logger.info(f"Starting deep research stream to {url} with query: {query}")

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.max_seconds
                    + 180,  # 3-min buffer: accounts for per-iteration LLM overrun
                    write=30.0,
                    pool=30.0,
                )
            ) as client:
                logger.debug(
                    f"Sending request to deep research with body: {request_body}"
                )
                async with client.stream(
                    "POST",
                    url,
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    logger.info(
                        f"Deep research response status: {response.status_code}"
                    )
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(
                            f"Deep research service returned {response.status_code}: "
                            f"{error_text.decode()}"
                        )
                        yield {
                            "type": "error",
                            "error": f"Deep research service error: {response.status_code}",
                        }
                        return

                    # Stream JSONL responses
                    buffer = ""
                    logger.info("Starting to read deep research stream")
                    async for chunk in response.aiter_text():
                        logger.debug(f"Received chunk: {chunk[:100]}...")
                        buffer += chunk

                        # Process complete lines
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue

                            try:
                                event = json.loads(line)
                                logger.debug(
                                    f"Deep research event: {event.get('event')}"
                                )
                                yield event
                            except json.JSONDecodeError as e:
                                logger.warning(
                                    f"Failed to parse deep research event: {e}, "
                                    f"line: {line[:100]}..."
                                )
                                continue

                    # Process any remaining content in buffer
                    if buffer.strip():
                        try:
                            event = json.loads(buffer.strip())
                            yield event
                        except json.JSONDecodeError:
                            pass

        except httpx.TimeoutException as e:
            logger.error(
                f"Deep research request timed out after {self.max_seconds}s: {e}"
            )
            yield {
                "type": "error",
                "error": "Deep research request timed out",
            }
        except httpx.ConnectError as e:
            logger.error(
                f"Deep research request failed: could not connect to {url} - {e}"
            )
            yield {
                "type": "error",
                "error": "Deep research service is unreachable. Please try again later.",
            }
        except httpx.RequestError as e:
            logger.error(f"Deep research request to {url} failed: {e}")
            yield {
                "type": "error",
                "error": "Deep research service is unreachable. Please try again later.",
            }
        except Exception as e:
            logger.exception("Unexpected error in deep research")
            yield {
                "type": "error",
                "error": f"Unexpected error: {e!s}",
            }

    async def execute(
        self,
        query: str,
        country: str = "us",
        language: str = "en",
    ) -> dict[str, Any]:
        """Execute deep research and return the final result (non-streaming).

        This collects all events and returns only the final answer.
        Use execute_streaming() for real-time event streaming.

        Args:
            query: The research question
            country: Search country code
            language: Search language

        Returns:
            Final answer dict with answer and citations
        """
        final_answer = None
        last_error = None

        async for event in self.execute_streaming(query, country, language):
            event_type = event.get("event")

            if event_type == "answer" and event.get("final"):
                final_answer = event
            elif event_type == "error" or event.get("type") == "error":
                last_error = event

        if final_answer:
            return final_answer
        elif last_error:
            return last_error
        else:
            return {
                "type": "error",
                "error": "No final answer received from deep research",
            }
