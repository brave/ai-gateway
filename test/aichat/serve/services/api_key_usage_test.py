import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm.types.utils import Usage
from prometheus_client import REGISTRY

from aichat.serve.services import api_key_usage

USAGE_CTX = api_key_usage.UsageContext(
    httpx_client=MagicMock(),
    api_key="k",
    transaction_id="txn_1",
    model="test-model",
    provider="bedrock",
)


class TestSetTokenUsageCounts:
    def test_none(self):
        assert api_key_usage.set_token_usage_counts(None) is None

    def test_litellm_usage(self):
        usage = Usage(
            prompt_tokens=13,
            completion_tokens=12,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=4,
            reasoning_tokens=5,
        )
        assert api_key_usage.set_token_usage_counts(
            usage
        ) == api_key_usage.TokenUsageCounts(
            prompt_tokens=13,
            completion_tokens=12,
            cache_write_tokens=3,
            cache_read_tokens=4,
            reasoning_tokens=5,
        )

    def test_falls_back_to_token_details(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 6, "cache_creation_tokens": 1},
            "completion_tokens_details": None,
        }
        assert api_key_usage.set_token_usage_counts(
            usage
        ) == api_key_usage.TokenUsageCounts(
            prompt_tokens=10,
            completion_tokens=2,
            cache_write_tokens=1,
            cache_read_tokens=6,
            reasoning_tokens=0,
        )

    def test_missing_fields_are_zero(self):
        assert (
            api_key_usage.set_token_usage_counts({}) == api_key_usage.TokenUsageCounts()
        )


class TestSettleUsage:
    @pytest.mark.asyncio
    async def test_records_token_metrics(self, monkeypatch):
        settle = AsyncMock(return_value={})
        monkeypatch.setattr(api_key_usage.internal_client, "usage_settle", settle)
        labels = {"model": "test-model", "token_type": "cache_read_tokens"}
        before = REGISTRY.get_sample_value("api_key_usage_tokens_total", labels) or 0.0
        await api_key_usage.settle_usage(
            USAGE_CTX, {"prompt_tokens": 1, "cache_read_input_tokens": 7}
        )
        assert REGISTRY.get_sample_value("api_key_usage_tokens_total", labels) == (
            before + 7
        )
        settle.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unreadable_usage_is_not_settled(self, monkeypatch):
        settle = AsyncMock()
        monkeypatch.setattr(api_key_usage.internal_client, "usage_settle", settle)
        await api_key_usage.settle_usage(USAGE_CTX, object())
        settle.assert_not_awaited()


class TestConverseAndSettle:
    @pytest.mark.asyncio
    async def test_settles_response_usage_in_background(self, monkeypatch):
        settle = AsyncMock()
        monkeypatch.setattr(api_key_usage, "settle_usage", settle)
        response = SimpleNamespace(usage={"prompt_tokens": 3})

        async def converse():
            return response

        result = await api_key_usage.converse_and_settle(converse(), USAGE_CTX)
        assert result is response
        await api_key_usage.wait_for_background_tasks(1)
        settle.assert_awaited_once_with(USAGE_CTX, {"prompt_tokens": 3})

    @pytest.mark.asyncio
    async def test_error_response_is_not_settled(self, monkeypatch):
        settle = AsyncMock()
        monkeypatch.setattr(api_key_usage, "settle_usage", settle)

        async def converse():
            return {"type": "error", "code": 50000}

        await api_key_usage.converse_and_settle(converse(), USAGE_CTX)
        await api_key_usage.wait_for_background_tasks(1)
        settle.assert_not_awaited()


class TestReadStreamAndSettle:
    @pytest.mark.asyncio
    async def test_queues_chunks_and_settles_final_usage(self, monkeypatch):
        settle = AsyncMock()
        monkeypatch.setattr(api_key_usage, "settle_usage", settle)
        content = SimpleNamespace(usage=None)
        final = SimpleNamespace(usage={"prompt_tokens": 3})

        async def gen():
            yield content
            yield final

        chunks: asyncio.Queue = asyncio.Queue()
        await api_key_usage.read_stream_and_settle(
            gen(), chunks, asyncio.Event(), USAGE_CTX
        )
        assert [chunks.get_nowait() for _ in range(3)] == [
            content,
            final,
            api_key_usage.STREAM_END,
        ]
        settle.assert_awaited_once_with(USAGE_CTX, {"prompt_tokens": 3})

    @pytest.mark.asyncio
    async def test_upstream_error_is_forwarded_and_settles_without_usage(
        self, monkeypatch
    ):
        settle = AsyncMock()
        monkeypatch.setattr(api_key_usage, "settle_usage", settle)
        error = RuntimeError("upstream failed")

        async def gen():
            raise error
            yield

        chunks: asyncio.Queue = asyncio.Queue()
        await api_key_usage.read_stream_and_settle(
            gen(), chunks, asyncio.Event(), USAGE_CTX
        )
        assert chunks.get_nowait() is error
        assert chunks.get_nowait() is api_key_usage.STREAM_END
        settle.assert_awaited_once_with(USAGE_CTX, None)

    @pytest.mark.asyncio
    async def test_full_buffer_pauses_upstream_reads(self, monkeypatch):
        monkeypatch.setattr(api_key_usage, "settle_usage", AsyncMock())
        read = []

        async def gen():
            for i in range(10):
                read.append(i)
                yield SimpleNamespace(usage=None)

        chunks: asyncio.Queue = asyncio.Queue(maxsize=2)
        reader = api_key_usage.spawn(
            api_key_usage.read_stream_and_settle(
                gen(), chunks, asyncio.Event(), USAGE_CTX
            )
        )
        await asyncio.sleep(0.01)
        assert len(read) == 3
        assert not reader.done()
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader

    @pytest.mark.asyncio
    async def test_disconnect_stops_buffering_and_still_settles(self, monkeypatch):
        settle = AsyncMock()
        monkeypatch.setattr(api_key_usage, "settle_usage", settle)

        async def gen():
            for _ in range(10):
                yield SimpleNamespace(usage=None)
            yield SimpleNamespace(usage={"prompt_tokens": 3})

        chunks: asyncio.Queue = asyncio.Queue(maxsize=2)
        disconnected = asyncio.Event()
        reader = api_key_usage.spawn(
            api_key_usage.read_stream_and_settle(gen(), chunks, disconnected, USAGE_CTX)
        )
        await asyncio.sleep(0.01)
        assert chunks.full()
        disconnected.set()
        while not chunks.empty():
            chunks.get_nowait()
        await reader
        assert chunks.qsize() <= 1
        settle.assert_awaited_once_with(USAGE_CTX, {"prompt_tokens": 3})


class TestWaitForBackgroundTasks:
    @pytest.mark.asyncio
    async def test_waits_for_pending_tasks(self):
        done = asyncio.Event()

        async def work():
            await asyncio.sleep(0.01)
            done.set()

        api_key_usage.spawn(work())
        await api_key_usage.wait_for_background_tasks(1)
        assert done.is_set()

    @pytest.mark.asyncio
    async def test_gives_up_after_timeout(self):
        task = api_key_usage.spawn(asyncio.sleep(10))
        await api_key_usage.wait_for_background_tasks(0.01)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestNewUsageContext:
    def test_builds_unique_transaction_ids(self):
        config = SimpleNamespace(backend="bedrock", api_base=None)
        first = api_key_usage.new_usage_context(MagicMock(), "k", "m", config)
        second = api_key_usage.new_usage_context(MagicMock(), "k", "m", config)
        assert first.transaction_id.startswith("txn_")
        assert first.transaction_id != second.transaction_id
        assert first.provider == "bedrock"
