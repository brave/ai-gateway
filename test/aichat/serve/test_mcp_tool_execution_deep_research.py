"""Tests for deep research functions in mcp_tool_execution.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from aichat.serve.mcp_tool_execution import (
    STREAMING_TOOLS,
    execute_streaming_tool_and_proxy_events,
)
from aichat.serve.services.mcp.handlers.deep_research import (
    transform_deep_research_event,
)


class TestStreamingToolsConstant:
    """Tests for STREAMING_TOOLS constant."""

    def test_deep_research_in_streaming_tools(self):
        """Test that 'deep_research' is in STREAMING_TOOLS."""
        assert "deep_research" in STREAMING_TOOLS

    def test_other_tools_not_in_streaming_tools(self):
        """Test that other tools are not in STREAMING_TOOLS."""
        assert "brave_web_search" not in STREAMING_TOOLS
        assert "search" not in STREAMING_TOOLS
        assert "" not in STREAMING_TOOLS


def _parse_brave_event(result):
    """Parse a brave-chat SSE event string into a dict."""
    assert result.startswith("data: ")
    assert result.endswith("\n\n")
    return json.loads(result[6:-2])


class TestTransformDeepResearchEvent:
    """Tests for transform_deep_research_event function.

    Events are now emitted as top-level brave-chat.deepResearch.* objects
    rather than nested inside chat.completion.chunk choices/delta.
    """

    def test_queries_event(self):
        """Test transformation of queries event."""
        event = {"event": "queries", "queries": ["query1", "query2"]}
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.queries"
        assert data["queries"] == ["query1", "query2"]

    def test_analyzing_event(self):
        """Test transformation of analyzing event."""
        event = {"event": "analyzing", "query": "test query", "urls": 10, "newUrls": 5}
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.analyzing"
        assert data["query"] == "test query"
        assert data["urls"] == 10
        assert data["new_urls"] == 5

    def test_thinking_event(self):
        """Test transformation of thinking event."""
        event = {
            "event": "thinking",
            "query": "test query",
            "chunksAnalyzed": 50,
            "chunksSelected": 10,
            "urlsAnalyzed": 5,
            "urlsSelected": ["url1", "url2"],
            "urlsInfo": [{"url": "url1", "title": "Title 1"}],
        }
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.thinking"
        assert data["query"] == "test query"
        assert data["chunks_analyzed"] == 50
        assert data["chunks_selected"] == 10
        assert data["urls_analyzed"] == 5
        assert data["urls_selected"] == ["url1", "url2"]

    def test_answer_event_returns_none(self):
        """Test that answer events return None (handled by streaming executor)."""
        event = {
            "event": "answer",
            "answer": "This is the answer",
            "final": True,
            "citations": [
                {"number": 1, "url": "https://example.com", "snippet": "Source snippet"}
            ],
        }
        result = transform_deep_research_event(event, "conv-123", "model-name")
        assert result is None

    def test_progress_event(self):
        """Test transformation of progress event."""
        event = {
            "event": "progress",
            "elapsedSeconds": 30,
            "iterations": 2,
            "queries": 5,
            "urlsAnalyzed": 15,
            "snippetsAnalyzed": 100,
        }
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.progress"
        assert data["elapsed_seconds"] == 30
        assert data["iterations"] == 2
        assert data["queries_count"] == 5

    def test_blindspots_event(self):
        """Test transformation of blindspots event."""
        event = {"event": "blindspots", "blindspots": ["blindspot1", "blindspot2"]}
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.blindspots"
        assert data["blindspots"] == ["blindspot1", "blindspot2"]

    def test_stopping_condition_event(self):
        """Test transformation of stopping_condition event."""
        event = {"event": "stopping_condition", "reason": "max_iterations_reached"}
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.complete"
        assert data["reason"] == "max_iterations_reached"

    def test_error_event(self):
        """Test transformation of error event."""
        event = {"event": "error", "error": "Something went wrong"}
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.error"
        assert data["error"] == "Something went wrong"

    def test_search_started_event(self):
        """Test transformation of search_started event."""
        event = {
            "event": "search_started",
            "query": "test query",
            "queryIndex": 1,
            "totalQueries": 5,
        }
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.searchStatus"
        assert data["status"] == "started"
        assert data["query"] == "test query"
        assert data["queryIndex"] == 1
        assert data["totalQueries"] == 5

    def test_search_completed_event(self):
        """Test transformation of search_completed event."""
        event = {
            "event": "search_completed",
            "query": "test query",
            "queryIndex": 1,
            "totalQueries": 5,
            "urlsFound": 10,
            "elapsedMs": 500,
        }
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.searchStatus"
        assert data["status"] == "completed"
        assert data["urlsFound"] == 10
        assert data["elapsedMs"] == 500

    def test_fetching_urls_event(self):
        """Test transformation of fetching_urls event."""
        event = {
            "event": "fetching_urls",
            "query": "test query",
            "urlsTotal": 20,
            "urlsFetched": 10,
        }
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.fetchStatus"
        assert data["query"] == "test query"
        assert data["urlsTotal"] == 20
        assert data["urlsFetched"] == 10

    def test_llm_analysis_started_event(self):
        """Test transformation of llm_analysis_started event."""
        event = {
            "event": "llm_analysis_started",
            "query": "test query",
            "chunksToAnalyze": 50,
        }
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.analysisStatus"
        assert data["status"] == "started"
        assert data["chunksTotal"] == 50
        assert data["chunksAnalyzed"] == 0

    def test_llm_analysis_progress_event(self):
        """Test transformation of llm_analysis_progress event."""
        event = {
            "event": "llm_analysis_progress",
            "query": "test query",
            "chunksAnalyzed": 25,
            "chunksTotal": 50,
        }
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.analysisStatus"
        assert data["status"] == "progress"
        assert data["chunksAnalyzed"] == 25
        assert data["chunksTotal"] == 50

    def test_iteration_complete_event(self):
        """Test transformation of iteration_complete event."""
        event = {
            "event": "iteration_complete",
            "iteration": 2,
            "totalIterations": 5,
            "queriesThisIteration": 3,
            "urlsAnalyzed": 15,
            "blindspotsIdentified": 2,
        }
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result is not None
        data = _parse_brave_event(result)
        assert data["object"] == "brave-chat.deepResearch.iterationComplete"
        assert data["iteration"] == 2
        assert data["totalIterations"] == 5
        assert data["queriesThisIteration"] == 3
        assert data["urlsAnalyzed"] == 15
        assert data["blindspotsIdentified"] == 2

    def test_unknown_event_returns_none(self):
        """Test that unknown events return None."""
        event = {"event": "unknown_event", "data": "something"}
        result = transform_deep_research_event(event, "conv-123", "model-name")
        assert result is None

    def test_internal_events_return_none(self):
        """Test that internal events like 'insights', 'timing' return None."""
        for event_type in ["insights", "timing", "internal"]:
            event = {"event": event_type, "data": "something"}
            result = transform_deep_research_event(event, "conv-123", "model-name")
            assert result is None

    def test_ping_returns_keepalive_comment(self):
        """Test that ping events return an SSE comment to keep connections alive."""
        event = {"event": "ping"}
        result = transform_deep_research_event(event, "conv-123", "model-name")
        assert result == ": keepalive\n\n"

    def test_sse_format_validation(self):
        """Test that output follows SSE format with brave-chat object type."""
        event = {"event": "queries", "queries": ["q1"]}
        result = transform_deep_research_event(event, "conv-123", "model-name")

        assert result.startswith("data: ")
        assert result.endswith("\n\n")

        # Verify JSON is valid with brave-chat object type
        data = _parse_brave_event(result)
        assert "object" in data
        assert data["object"] == "brave-chat.deepResearch.queries"


def _mock_handler_for_registry():
    """Create a mock registry with a deep_research handler."""
    from aichat.serve.services.mcp.handlers.deep_research.handler import (
        DeepResearchServerHandler,
    )

    handler = DeepResearchServerHandler()
    mock_registry = MagicMock()
    mock_registry.handlers = {"deep_research": handler}
    return mock_registry


class TestExecuteStreamingToolAndProxyEvents:
    """Tests for execute_streaming_tool_and_proxy_events function."""

    @pytest.fixture
    def mock_tool_call(self):
        """Create a mock tool call."""
        mock = MagicMock()
        mock.function_name = "deep_research"
        mock.arguments = json.dumps({"query": "test research query"})
        mock.id = "call_123"
        mock.index = 0
        return mock

    @pytest.mark.asyncio
    async def test_tool_start_event_emitted(self, mock_tool_call):
        """Test that tool start event is emitted without hint message."""
        mock_executor = MagicMock()

        async def mock_execute_streaming(*args, **kwargs):
            yield {"event": "answer", "final": True, "answer": "Answer"}

        mock_executor.execute_streaming = mock_execute_streaming

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.DeepResearchStreamingExecutor",
                return_value=mock_executor,
            ),
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.get_global_registry",
                return_value=_mock_handler_for_registry(),
            ),
        ):
            events = []
            async for event, msg in execute_streaming_tool_and_proxy_events(
                mock_tool_call, "conv-123", "model-name"
            ):
                events.append((event, msg))

            # First event should be tool start
            first_event = events[0][0]
            assert "brave-chat.toolStart" in first_event
            first_event_data = json.loads(first_event[6:-2])
            assert first_event_data["object"] == "brave-chat.toolStart"
            assert "message" not in first_event_data

    @pytest.mark.asyncio
    async def test_event_transformation(self, mock_tool_call):
        """Test that deep research events are transformed as brave-chat objects."""
        mock_executor = MagicMock()

        async def mock_execute_streaming(*args, **kwargs):
            yield {"event": "queries", "queries": ["q1", "q2"]}
            yield {"event": "answer", "final": True, "answer": "Final"}

        mock_executor.execute_streaming = mock_execute_streaming

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.DeepResearchStreamingExecutor",
                return_value=mock_executor,
            ),
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.get_global_registry",
                return_value=_mock_handler_for_registry(),
            ),
        ):
            events = []
            async for event, _msg in execute_streaming_tool_and_proxy_events(
                mock_tool_call, "conv-123", "model-name"
            ):
                if event:
                    events.append(event)

            # Should have: tool_start, queries, completion content, tool_end (+ possible WebSources)
            assert len(events) >= 3
            # Check that queries event was transformed as brave-chat object
            queries_found = any("brave-chat.deepResearch.queries" in e for e in events)
            assert queries_found

    @pytest.mark.asyncio
    async def test_final_answer_tracking(self, mock_tool_call):
        """Test that final answer is tracked and included in tool message."""
        mock_executor = MagicMock()

        async def mock_execute_streaming(*args, **kwargs):
            yield {
                "event": "answer",
                "final": True,
                "answer": "Final research answer",
                "citations": [
                    {"number": 1, "url": "https://example.com", "snippet": "Source"}
                ],
            }

        mock_executor.execute_streaming = mock_execute_streaming

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.DeepResearchStreamingExecutor",
                return_value=mock_executor,
            ),
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.get_global_registry",
                return_value=_mock_handler_for_registry(),
            ),
        ):
            tool_messages = []
            async for _event, msg in execute_streaming_tool_and_proxy_events(
                mock_tool_call, "conv-123", "model-name"
            ):
                if msg:
                    tool_messages.append(msg)

            # Should have exactly one tool message at the end
            assert len(tool_messages) == 1
            content = tool_messages[0].content
            # With handler, content is a list of content parts when citations exist
            if isinstance(content, list):
                text_parts = [p for p in content if p.type == "text"]
                assert any("Final research answer" in p.text for p in text_parts)
            else:
                assert "Final research answer" in content

    @pytest.mark.asyncio
    async def test_citation_formatting(self, mock_tool_call):
        """Test that citations are formatted in tool message via handler."""
        mock_executor = MagicMock()

        async def mock_execute_streaming(*args, **kwargs):
            yield {
                "event": "answer",
                "final": True,
                "answer": "Answer with citation [1]",
                "citations": [
                    {"number": 1, "url": "https://example.com", "snippet": "Source"}
                ],
            }

        mock_executor.execute_streaming = mock_execute_streaming

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.DeepResearchStreamingExecutor",
                return_value=mock_executor,
            ),
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.get_global_registry",
                return_value=_mock_handler_for_registry(),
            ),
        ):
            tool_messages = []
            async for _event, msg in execute_streaming_tool_and_proxy_events(
                mock_tool_call, "conv-123", "model-name"
            ):
                if msg:
                    tool_messages.append(msg)

            assert len(tool_messages) == 1
            content = tool_messages[0].content
            # With handler + citations, content is a list with TextContentPart
            # and WebSourcesContentPart
            if isinstance(content, list):
                text_parts = [p for p in content if p.type == "text"]
                web_source_parts = [
                    p for p in content if p.type == "brave-chat.webSources"
                ]
                assert len(text_parts) >= 1
                assert len(web_source_parts) == 1
                # Text part should contain Sources info
                full_text = " ".join(p.text for p in text_parts)
                assert "Sources:" in full_text
                assert "https://example.com" in full_text
            else:
                assert "Sources:" in content
                assert "https://example.com" in content

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_tool_call):
        """Test that error events are handled."""
        mock_executor = MagicMock()

        async def mock_execute_streaming(*args, **kwargs):
            yield {"event": "error", "error": "Test error message"}

        mock_executor.execute_streaming = mock_execute_streaming

        with patch(
            "aichat.serve.services.mcp.handlers.deep_research.service.DeepResearchStreamingExecutor",
            return_value=mock_executor,
        ):
            events = []
            async for event, _msg in execute_streaming_tool_and_proxy_events(
                mock_tool_call, "conv-123", "model-name"
            ):
                if event:
                    events.append(event)

            # Error events are emitted both as brave-chat.deepResearch.error
            # (from the transformer) and as brave-chat.toolError (from the executor)
            error_found = any(
                "brave-chat.deepResearch.error" in e or "brave-chat.toolError" in e
                for e in events
            )
            assert error_found

    @pytest.mark.asyncio
    async def test_tool_end_event(self, mock_tool_call):
        """Test that tool end event is emitted."""
        mock_executor = MagicMock()

        async def mock_execute_streaming(*args, **kwargs):
            yield {"event": "answer", "final": True, "answer": "Answer"}

        mock_executor.execute_streaming = mock_execute_streaming

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.DeepResearchStreamingExecutor",
                return_value=mock_executor,
            ),
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.get_global_registry",
                return_value=_mock_handler_for_registry(),
            ),
        ):
            events = []
            async for event, _msg in execute_streaming_tool_and_proxy_events(
                mock_tool_call, "conv-123", "model-name"
            ):
                if event:
                    events.append(event)

            # Last event should be tool end
            last_event = events[-1]
            assert "brave-chat.toolEnd" in last_event

    @pytest.mark.asyncio
    async def test_tool_message_content(self, mock_tool_call):
        """Test ToolMessage content structure."""
        mock_executor = MagicMock()

        async def mock_execute_streaming(*args, **kwargs):
            yield {"event": "answer", "final": True, "answer": "Final answer text"}

        mock_executor.execute_streaming = mock_execute_streaming

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.DeepResearchStreamingExecutor",
                return_value=mock_executor,
            ),
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.get_global_registry",
                return_value=_mock_handler_for_registry(),
            ),
        ):
            tool_messages = []
            async for _event, msg in execute_streaming_tool_and_proxy_events(
                mock_tool_call, "conv-123", "model-name"
            ):
                if msg:
                    tool_messages.append(msg)

            assert len(tool_messages) == 1
            tool_msg = tool_messages[0]
            assert tool_msg.role == "tool"
            assert tool_msg.tool_call_id == "call_123"
            # Without citations, handler returns plain string
            content = tool_msg.content
            if isinstance(content, str):
                assert "Final answer text" in content
            else:
                text_parts = [p for p in content if p.type == "text"]
                assert any("Final answer text" in p.text for p in text_parts)

    @pytest.mark.asyncio
    async def test_missing_query_error(self):
        """Test error when query is missing."""
        mock_tool_call = MagicMock()
        mock_tool_call.function_name = "deep_research"
        mock_tool_call.arguments = json.dumps({})  # No query
        mock_tool_call.id = "call_123"
        mock_tool_call.index = 0

        events = []
        async for event, _msg in execute_streaming_tool_and_proxy_events(
            mock_tool_call, "conv-123", "model-name"
        ):
            if event:
                events.append(event)

        # Should have error event about missing query
        error_found = any("brave-chat.toolError" in e for e in events)
        assert error_found
        error_event = next(e for e in events if "brave-chat.toolError" in e)
        assert "query" in error_event.lower()

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self):
        """Test error for unknown streaming tool."""
        mock_tool_call = MagicMock()
        mock_tool_call.function_name = "unknown_streaming_tool"
        mock_tool_call.arguments = json.dumps({"query": "test"})
        mock_tool_call.id = "call_123"
        mock_tool_call.index = 0

        events = []
        async for event, _msg in execute_streaming_tool_and_proxy_events(
            mock_tool_call, "conv-123", "model-name"
        ):
            if event:
                events.append(event)

        # Should have error event about unknown tool
        error_found = any("brave-chat.toolError" in e for e in events)
        assert error_found

    @pytest.mark.asyncio
    async def test_no_final_answer_message(self, mock_tool_call):
        """Test tool message when no final answer is received."""
        mock_executor = MagicMock()

        async def mock_execute_streaming(*args, **kwargs):
            yield {"event": "queries", "queries": ["q1"]}
            # No final answer

        mock_executor.execute_streaming = mock_execute_streaming

        with patch(
            "aichat.serve.services.mcp.handlers.deep_research.service.DeepResearchStreamingExecutor",
            return_value=mock_executor,
        ):
            tool_messages = []
            async for _event, msg in execute_streaming_tool_and_proxy_events(
                mock_tool_call, "conv-123", "model-name"
            ):
                if msg:
                    tool_messages.append(msg)

            # Should still get a tool message with fallback content
            assert len(tool_messages) == 1
            assert "no final answer" in tool_messages[0].content.lower()

    @pytest.mark.asyncio
    async def test_websources_emitted_with_citations(self, mock_tool_call):
        """Test that WebSources output chunk is emitted when citations exist."""
        mock_executor = MagicMock()

        async def mock_execute_streaming(*args, **kwargs):
            yield {
                "event": "answer",
                "final": True,
                "answer": "Answer with sources",
                "citations": [
                    {"number": 1, "url": "https://example.com", "snippet": "Source"}
                ],
            }

        mock_executor.execute_streaming = mock_execute_streaming

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.DeepResearchStreamingExecutor",
                return_value=mock_executor,
            ),
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.service.get_global_registry",
                return_value=_mock_handler_for_registry(),
            ),
        ):
            events = []
            async for event, _msg in execute_streaming_tool_and_proxy_events(
                mock_tool_call, "conv-123", "model-name"
            ):
                if event:
                    events.append(event)

            # Should have a WebSources output chunk
            web_sources_found = any("brave-chat.webSources" in e for e in events)
            assert web_sources_found
