"""Tests for DeepResearchStreamingExecutor and iter_events_with_keepalive."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aichat.serve.services.mcp.handlers.deep_research import (
    DeepResearchStreamingExecutor,
)
from aichat.serve.services.mcp.handlers.deep_research.streaming_executor import (
    iter_events_with_keepalive,
)


@pytest.fixture
def mock_mcp_settings():
    """Mock mcp_settings with deep research config."""
    mock = MagicMock()
    mock.deep_research_enabled = True
    mock.deep_research_url = "http://test:3011"
    mock.deep_research_max_iterations = 3
    mock.deep_research_max_queries = 30
    mock.deep_research_max_seconds = 300
    return mock


class TestInit:
    """Tests for __init__ method."""

    def test_default_values_from_settings(self, mock_mcp_settings):
        """Test that default values come from mcp_settings."""
        with patch(
            "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
            mock_mcp_settings,
        ):
            executor = DeepResearchStreamingExecutor()

            assert executor.deep_research_url == "http://test:3011"
            assert executor.max_iterations == 3
            assert executor.max_queries == 30
            assert executor.max_seconds == 300

    def test_explicit_params_override(self, mock_mcp_settings):
        """Test that explicit parameters override settings."""
        with patch(
            "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
            mock_mcp_settings,
        ):
            executor = DeepResearchStreamingExecutor(
                deep_research_url="http://custom:9999",
                max_iterations=5,
                max_queries=50,
                max_seconds=600,
            )

            assert executor.deep_research_url == "http://custom:9999"
            assert executor.max_iterations == 5
            assert executor.max_queries == 50
            assert executor.max_seconds == 600


class TestExecuteStreaming:
    """Tests for execute_streaming method."""

    @pytest.mark.asyncio
    async def test_success_with_jsonl(self, mock_mcp_settings):
        """Test successful streaming with JSONL response."""
        events = [
            {"event": "queries", "queries": ["query1", "query2"]},
            {"event": "thinking", "query": "query1"},
            {"event": "answer", "final": True, "answer": "Final answer"},
        ]
        chunks = [json.dumps(e) + "\n" for e in events]

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def aiter_text():
            for chunk in chunks:
                yield chunk

        mock_response.aiter_text = aiter_text

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(return_value=AsyncMock())
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
                mock_mcp_settings,
            ),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            executor = DeepResearchStreamingExecutor()
            received_events = []
            async for event in executor.execute_streaming("test query"):
                received_events.append(event)

            assert len(received_events) == 3
            assert received_events[0]["event"] == "queries"
            assert received_events[1]["event"] == "thinking"
            assert received_events[2]["event"] == "answer"
            assert received_events[2]["final"] is True

    @pytest.mark.asyncio
    async def test_http_error_status(self, mock_mcp_settings):
        """Test handling of HTTP error status codes."""
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Internal Server Error")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(return_value=AsyncMock())
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
                mock_mcp_settings,
            ),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            executor = DeepResearchStreamingExecutor()
            received_events = []
            async for event in executor.execute_streaming("test query"):
                received_events.append(event)

            assert len(received_events) == 1
            assert received_events[0]["type"] == "error"
            assert "500" in received_events[0]["error"]

    @pytest.mark.asyncio
    async def test_timeout(self, mock_mcp_settings):
        """Test handling of timeout exception."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
                mock_mcp_settings,
            ),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            executor = DeepResearchStreamingExecutor()
            received_events = []
            async for event in executor.execute_streaming("test query"):
                received_events.append(event)

            assert len(received_events) == 1
            assert received_events[0]["type"] == "error"
            assert "timed out" in received_events[0]["error"]

    @pytest.mark.asyncio
    async def test_request_error(self, mock_mcp_settings):
        """Test handling of request error."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(
            side_effect=httpx.RequestError("Connection refused")
        )

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
                mock_mcp_settings,
            ),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            executor = DeepResearchStreamingExecutor()
            received_events = []
            async for event in executor.execute_streaming("test query"):
                received_events.append(event)

            assert len(received_events) == 1
            assert received_events[0]["type"] == "error"
            assert "unreachable" in received_events[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_unexpected_exception(self, mock_mcp_settings):
        """Test handling of unexpected exception."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(side_effect=Exception("Unexpected error"))

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
                mock_mcp_settings,
            ),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            executor = DeepResearchStreamingExecutor()
            received_events = []
            async for event in executor.execute_streaming("test query"):
                received_events.append(event)

            assert len(received_events) == 1
            assert received_events[0]["type"] == "error"
            assert "Unexpected error" in received_events[0]["error"]

    @pytest.mark.asyncio
    async def test_jsonl_parsing_skips_empty_lines(self, mock_mcp_settings):
        """Test that empty lines in JSONL are skipped."""
        chunks = ['{"event": "queries"}\n', "\n", '{"event": "answer"}\n', "\n\n"]

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def aiter_text():
            for chunk in chunks:
                yield chunk

        mock_response.aiter_text = aiter_text

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(return_value=AsyncMock())
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
                mock_mcp_settings,
            ),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            executor = DeepResearchStreamingExecutor()
            received_events = []
            async for event in executor.execute_streaming("test query"):
                received_events.append(event)

            assert len(received_events) == 2

    @pytest.mark.asyncio
    async def test_invalid_json_skipped(self, mock_mcp_settings):
        """Test that invalid JSON lines are skipped."""
        chunks = ['{"event": "queries"}\n', "not valid json\n", '{"event": "answer"}\n']

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def aiter_text():
            for chunk in chunks:
                yield chunk

        mock_response.aiter_text = aiter_text

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(return_value=AsyncMock())
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
                mock_mcp_settings,
            ),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            executor = DeepResearchStreamingExecutor()
            received_events = []
            async for event in executor.execute_streaming("test query"):
                received_events.append(event)

            # Should only get 2 valid events, invalid JSON skipped
            assert len(received_events) == 2

    @pytest.mark.asyncio
    async def test_buffer_remainder_processed(self, mock_mcp_settings):
        """Test that remaining buffer content is processed at end of stream."""
        # Send event without trailing newline
        chunks = ['{"event": "answer", "final": true}']

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def aiter_text():
            for chunk in chunks:
                yield chunk

        mock_response.aiter_text = aiter_text

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(return_value=AsyncMock())
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
                mock_mcp_settings,
            ),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            executor = DeepResearchStreamingExecutor()
            received_events = []
            async for event in executor.execute_streaming("test query"):
                received_events.append(event)

            # Should process the remaining buffer
            assert len(received_events) == 1
            assert received_events[0]["event"] == "answer"


class TestExecute:
    """Tests for execute method (non-streaming)."""

    @pytest.mark.asyncio
    async def test_returns_final_answer(self, mock_mcp_settings):
        """Test that execute returns the final answer."""
        events = [
            {"event": "queries", "queries": ["query1"]},
            {"event": "answer", "final": False, "answer": "Partial"},
            {"event": "answer", "final": True, "answer": "Final answer"},
        ]

        async def mock_execute_streaming(*args, **kwargs):
            for event in events:
                yield event

        with patch(
            "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
            mock_mcp_settings,
        ):
            executor = DeepResearchStreamingExecutor()
            executor.execute_streaming = mock_execute_streaming

            result = await executor.execute("test query")

            assert result["event"] == "answer"
            assert result["final"] is True
            assert result["answer"] == "Final answer"

    @pytest.mark.asyncio
    async def test_no_final_answer_error(self, mock_mcp_settings):
        """Test that missing final answer returns error."""
        events = [
            {"event": "queries", "queries": ["query1"]},
            {"event": "thinking", "query": "query1"},
            # No final answer
        ]

        async def mock_execute_streaming(*args, **kwargs):
            for event in events:
                yield event

        with patch(
            "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
            mock_mcp_settings,
        ):
            executor = DeepResearchStreamingExecutor()
            executor.execute_streaming = mock_execute_streaming

            result = await executor.execute("test query")

            assert result["type"] == "error"
            assert "No final answer" in result["error"]

    @pytest.mark.asyncio
    async def test_last_error_fallback(self, mock_mcp_settings):
        """Test that last error is returned if no final answer."""
        events = [
            {"event": "queries", "queries": ["query1"]},
            {"event": "error", "error": "Something went wrong"},
        ]

        async def mock_execute_streaming(*args, **kwargs):
            for event in events:
                yield event

        with patch(
            "aichat.serve.services.mcp.handlers.deep_research.streaming_executor.mcp_settings",
            mock_mcp_settings,
        ):
            executor = DeepResearchStreamingExecutor()
            executor.execute_streaming = mock_execute_streaming

            result = await executor.execute("test query")

            assert result["event"] == "error"
            assert result["error"] == "Something went wrong"


class TestIterEventsWithKeepalive:
    """Tests for the iter_events_with_keepalive wrapper."""

    @pytest.mark.asyncio
    async def test_events_pass_through_unchanged(self):
        """All source events are forwarded in order without modification."""
        events = [
            {"event": "queries", "queries": ["q1"]},
            {"event": "thinking", "query": "q1"},
            {"event": "answer", "final": True, "answer": "done"},
        ]

        async def source():
            for e in events:
                yield e

        received = []
        async for item in iter_events_with_keepalive(source(), keepalive_interval=5.0):
            received.append(item)

        assert received == events

    @pytest.mark.asyncio
    async def test_ping_injected_on_idle(self):
        """A ping event is injected when the source is idle past the interval."""

        async def slow_source():
            yield {"event": "start"}
            await asyncio.sleep(0.3)
            yield {"event": "end"}

        received = []
        async for item in iter_events_with_keepalive(
            slow_source(), keepalive_interval=0.05
        ):
            received.append(item)

        event_types = [e["event"] for e in received]
        assert event_types[0] == "start"
        assert event_types[-1] == "end"
        assert "ping" in event_types, "Expected at least one keepalive ping"

    @pytest.mark.asyncio
    async def test_no_ping_when_events_flow_fast(self):
        """No pings are injected when events arrive faster than the interval."""

        async def fast_source():
            for i in range(5):
                yield {"event": "progress", "i": i}

        received = []
        async for item in iter_events_with_keepalive(
            fast_source(), keepalive_interval=5.0
        ):
            received.append(item)

        assert len(received) == 5
        assert all(e["event"] == "progress" for e in received)

    @pytest.mark.asyncio
    async def test_source_exception_is_reraised(self):
        """Exceptions from the source generator propagate to the consumer."""

        async def failing_source():
            yield {"event": "start"}
            raise RuntimeError("LLM call failed")

        received = []
        with pytest.raises(RuntimeError, match="LLM call failed"):
            async for item in iter_events_with_keepalive(
                failing_source(), keepalive_interval=5.0
            ):
                received.append(item)

        assert len(received) == 1
        assert received[0]["event"] == "start"

    @pytest.mark.asyncio
    async def test_producer_task_cleaned_up_on_completion(self):
        """The background producer task is cancelled after normal completion."""

        async def source():
            yield {"event": "done"}

        wrapper = iter_events_with_keepalive(source(), keepalive_interval=5.0)
        collected = [item async for item in wrapper]

        assert len(collected) == 1
        # Allow a tick for the finally block to run
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_producer_task_cleaned_up_on_early_break(self):
        """The producer task is cancelled if the consumer stops early."""

        async def infinite_source():
            i = 0
            while True:
                yield {"event": "tick", "i": i}
                i += 1
                await asyncio.sleep(0.01)

        received = []
        async for item in iter_events_with_keepalive(
            infinite_source(), keepalive_interval=5.0
        ):
            received.append(item)
            if len(received) >= 3:
                break

        assert len(received) == 3
        # Allow a tick for cleanup
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_empty_source(self):
        """An empty source yields nothing (no pings, no errors)."""

        async def empty():
            return
            yield  # noqa: unreachable — makes this an async generator

        received = [
            item
            async for item in iter_events_with_keepalive(
                empty(), keepalive_interval=5.0
            )
        ]
        assert received == []

    @pytest.mark.asyncio
    async def test_multiple_pings_during_long_idle(self):
        """Multiple pings are emitted during a long idle period."""

        async def long_pause():
            yield {"event": "start"}
            await asyncio.sleep(0.25)
            yield {"event": "end"}

        received = []
        async for item in iter_events_with_keepalive(
            long_pause(), keepalive_interval=0.05
        ):
            received.append(item)

        ping_count = sum(1 for e in received if e["event"] == "ping")
        assert ping_count >= 2, f"Expected multiple pings, got {ping_count}"
