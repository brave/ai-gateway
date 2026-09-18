"""Event transformers for MCP tool execution.

This module contains functions that transform MCP tool events into
Brave SSE format for streaming to clients.
"""

import json
import time
from enum import Enum
from typing import Any

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)


class DeepResearchEventType(str, Enum):
    """Event types emitted by the deep research service."""

    QUERIES = "queries"
    ANALYZING = "analyzing"
    THINKING = "thinking"
    ANSWER = "answer"
    PROGRESS = "progress"
    BLINDSPOTS = "blindspots"
    STOPPING_CONDITION = "stopping_condition"
    ERROR = "error"
    SEARCH_STARTED = "search_started"
    SEARCH_COMPLETED = "search_completed"
    FETCHING_URLS = "fetching_urls"
    LLM_ANALYSIS_STARTED = "llm_analysis_started"
    LLM_ANALYSIS_PROGRESS = "llm_analysis_progress"
    ITERATION_COMPLETE = "iteration_complete"
    PING = "ping"


def create_completion_chunk(
    content: str,
    conversation_log_id: str,
    model_name: str,
) -> str:
    """Create a standard chat.completion.chunk with delta content.

    Args:
        content: The text content for the completion delta
        conversation_log_id: Conversation log ID
        model_name: Model name

    Returns:
        SSE formatted completion chunk string
    """
    chunk = ChatCompletionChunk(
        id=conversation_log_id or "unknown",
        created=int(time.time()),
        model=model_name,
        object="chat.completion.chunk",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=content),
            )
        ],
    )
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


def _create_brave_event(object_type: str, data: dict) -> str:
    """Create a top-level brave-chat event.

    Args:
        object_type: The brave-chat.deepResearch.* object type
        data: The event data fields

    Returns:
        SSE formatted event string
    """
    event = {"object": object_type, **data}
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


def transform_deep_research_event(
    event: dict[str, Any],
    conversation_log_id: str,
    model_name: str,
) -> str | None:
    """Transform a deep research event into a brave SSE chunk.

    Args:
        event: Deep research event dictionary
        conversation_log_id: Conversation log ID
        model_name: Model name

    Returns:
        SSE formatted event string or None if event should be skipped
    """
    event_type = event.get("event")

    if event_type == DeepResearchEventType.QUERIES:
        return _create_brave_event(
            "brave-chat.deepResearch.queries",
            {"queries": event.get("queries", [])},
        )

    elif event_type == DeepResearchEventType.ANALYZING:
        return _create_brave_event(
            "brave-chat.deepResearch.analyzing",
            {
                "query": event.get("query", ""),
                "urls": event.get("urls", 0),
                "new_urls": event.get("newUrls", 0),
            },
        )

    elif event_type == DeepResearchEventType.THINKING:
        return _create_brave_event(
            "brave-chat.deepResearch.thinking",
            {
                "query": event.get("query", ""),
                "chunks_analyzed": event.get("chunksAnalyzed", 0),
                "chunks_selected": event.get("chunksSelected", 0),
                "urls_analyzed": event.get("urlsAnalyzed", 0),
                "urls_selected": event.get("urlsSelected", []),
                "urls_info": event.get("urlsInfo", []),
            },
        )

    # Answer events are handled by the streaming executor which emits them
    # as standard completion chunks and WebSources tool output, so we skip
    # them here to avoid redundant data.

    elif event_type == DeepResearchEventType.PROGRESS:
        return _create_brave_event(
            "brave-chat.deepResearch.progress",
            {
                "elapsed_seconds": event.get("elapsedSeconds", 0),
                "iterations": event.get("iterations", 0),
                "queries_count": event.get("queries", 0),
                "urls_analyzed": event.get("urlsAnalyzed", 0),
                "snippets_analyzed": event.get("snippetsAnalyzed", 0),
            },
        )

    elif event_type == DeepResearchEventType.BLINDSPOTS:
        return _create_brave_event(
            "brave-chat.deepResearch.blindspots",
            {"blindspots": event.get("blindspots", [])},
        )

    elif event_type == DeepResearchEventType.STOPPING_CONDITION:
        return _create_brave_event(
            "brave-chat.deepResearch.complete",
            {"reason": event.get("reason", "")},
        )

    elif event_type == DeepResearchEventType.ERROR:
        return _create_brave_event(
            "brave-chat.deepResearch.error",
            {"error": event.get("error", "Unknown error")},
        )

    # Granular progress events for UX improvement
    elif event_type == DeepResearchEventType.SEARCH_STARTED:
        return _create_brave_event(
            "brave-chat.deepResearch.searchStatus",
            {
                "status": "started",
                "query": event.get("query", ""),
                "queryIndex": event.get("queryIndex", 0),
                "totalQueries": event.get("totalQueries", 0),
            },
        )

    elif event_type == DeepResearchEventType.SEARCH_COMPLETED:
        return _create_brave_event(
            "brave-chat.deepResearch.searchStatus",
            {
                "status": "completed",
                "query": event.get("query", ""),
                "queryIndex": event.get("queryIndex", 0),
                "totalQueries": event.get("totalQueries", 0),
                "urlsFound": event.get("urlsFound", 0),
                "elapsedMs": event.get("elapsedMs", 0),
            },
        )

    elif event_type == DeepResearchEventType.FETCHING_URLS:
        return _create_brave_event(
            "brave-chat.deepResearch.fetchStatus",
            {
                "query": event.get("query", ""),
                "urlsTotal": event.get("urlsTotal", 0),
                "urlsFetched": event.get("urlsFetched", 0),
            },
        )

    elif event_type == DeepResearchEventType.LLM_ANALYSIS_STARTED:
        return _create_brave_event(
            "brave-chat.deepResearch.analysisStatus",
            {
                "status": "started",
                "query": event.get("query", ""),
                # Map chunksToAnalyze to chunksTotal for browser compatibility
                "chunksTotal": event.get("chunksToAnalyze", 0),
                "chunksAnalyzed": 0,
            },
        )

    elif event_type == DeepResearchEventType.LLM_ANALYSIS_PROGRESS:
        return _create_brave_event(
            "brave-chat.deepResearch.analysisStatus",
            {
                "status": "progress",
                "query": event.get("query", ""),
                "chunksAnalyzed": event.get("chunksAnalyzed", 0),
                "chunksTotal": event.get("chunksTotal", 0),
            },
        )

    elif event_type == DeepResearchEventType.ITERATION_COMPLETE:
        return _create_brave_event(
            "brave-chat.deepResearch.iterationComplete",
            {
                "iteration": event.get("iteration", 0),
                "totalIterations": event.get("totalIterations", 0),
                "queriesThisIteration": event.get("queriesThisIteration", 0),
                "urlsAnalyzed": event.get("urlsAnalyzed", 0),
                "blindspotsIdentified": event.get("blindspotsIdentified", 0),
            },
        )

    elif event_type == DeepResearchEventType.PING:
        # SSE comment — resets nginx proxy_read_timeout without producing a
        # visible event in the browser, keeping the connection alive during
        # long LLM calls that emit no other events.
        return ": keepalive\n\n"

    # Skip internal events like insights, timing, etc.
    return None
