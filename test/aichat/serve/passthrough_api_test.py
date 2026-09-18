"""Tests for passthrough_api."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from aichat.serve import passthrough_api


@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.json = AsyncMock()
    return request


class TestV1Passthrough:
    @pytest.mark.asyncio
    async def test_invalid_body_raises_400(self, mock_request):
        mock_request.json.return_value = {"model": 1}
        with pytest.raises(HTTPException) as exc:
            await passthrough_api.v1_passthrough(mock_request)
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_backend_raises_400(self, mock_request):
        mock_request.json.return_value = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with (
            patch(
                "aichat.serve.passthrough_api.get_backend",
                side_effect=ValueError("unknown model"),
            ),
            pytest.raises(HTTPException) as exc,
        ):
            await passthrough_api.v1_passthrough(mock_request)
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_success_non_stream(self, mock_request):
        mock_request.json.return_value = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }
        mock_response = MagicMock()
        with (
            patch("aichat.serve.passthrough_api.get_backend") as mock_get_backend,
            patch("aichat.serve.passthrough_api.apply_claude_upstream_sampling_params"),
        ):
            backend = mock_get_backend.return_value
            backend.config.upstream_model = "claude"
            backend.build_params.return_value = {"temperature": 1}
            backend.converse = AsyncMock(return_value=mock_response)
            resp = await passthrough_api.v1_passthrough(mock_request)
        assert resp is mock_response

    @pytest.mark.asyncio
    async def test_error_dict_raises_http(self, mock_request):
        mock_request.json.return_value = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }
        error_dict = {"type": "error", "code": 50000, "content": "boom"}
        with (
            patch("aichat.serve.passthrough_api.get_backend") as mock_get_backend,
            patch("aichat.serve.passthrough_api.apply_claude_upstream_sampling_params"),
        ):
            backend = mock_get_backend.return_value
            backend.config.upstream_model = "claude"
            backend.build_params.return_value = {}
            backend.converse = AsyncMock(return_value=error_dict)
            with pytest.raises(HTTPException) as exc:
                await passthrough_api.v1_passthrough(mock_request)
        assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_streaming(self, mock_request):
        mock_request.json.return_value = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        async def gen():
            yield MagicMock()

        with (
            patch("aichat.serve.passthrough_api.get_backend") as mock_get_backend,
            patch("aichat.serve.passthrough_api.apply_claude_upstream_sampling_params"),
        ):
            backend = mock_get_backend.return_value
            backend.config.upstream_model = "claude"
            backend.build_params.return_value = {}
            backend.converse = AsyncMock(return_value=gen())
            resp = await passthrough_api.v1_passthrough(mock_request)
        assert resp.media_type == "text/event-stream"


class TestStreamChunks:
    @pytest.mark.asyncio
    async def test_yields_chunks_and_done(self):
        chunk = MagicMock()
        chunk.model_dump_json.return_value = '{"id": "1"}'

        async def gen():
            yield chunk

        out = [c async for c in passthrough_api._stream_chunks(gen())]
        assert out[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_unserializable_chunk_logs_warning(self):
        chunk = MagicMock()
        chunk.model_dump_json.side_effect = TypeError("no")

        async def gen():
            yield chunk

        out = [c async for c in passthrough_api._stream_chunks(gen())]
        assert out == ["data: [DONE]\n\n"]
