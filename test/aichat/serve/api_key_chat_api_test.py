"""Tests for api_key_chat_api."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from aichat.serve import api_key_chat_api


def test_openai_error_response_shape():
    resp = api_key_chat_api.openai_error_response(401, "bad", "invalid_api_key")
    assert resp.status_code == 401


def test_is_valid_api_key():
    assert api_key_chat_api.is_valid_api_key("brv_live_" + "a" * 32)
    assert not api_key_chat_api.is_valid_api_key("brv_live_" + "a" * 10)
    assert not api_key_chat_api.is_valid_api_key("other_" + "a" * 40)
    assert not api_key_chat_api.is_valid_api_key("brv_live_" + "a" * 70)


@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.json = AsyncMock()
    request.body = AsyncMock(return_value=b"{}")
    request.headers = {}
    return request


class TestHandleChatCompletions:
    @pytest.mark.asyncio
    async def test_invalid_body_returns_400(self, mock_request):
        mock_request.json.return_value = {"model": 1}
        resp = await api_key_chat_api.handle_chat_completions(mock_request)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_success_non_stream(self, mock_request):
        mock_request.json.return_value = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"id": "x"}
        with (
            patch("aichat.serve.api_key_chat_api.get_backend") as mock_get_backend,
            patch(
                "aichat.serve.api_key_chat_api.apply_claude_upstream_sampling_params"
            ),
        ):
            backend = mock_get_backend.return_value
            backend.config.upstream_model = "claude"
            backend.build_params.return_value = {"temperature": 1}
            backend.converse = AsyncMock(return_value=mock_response)
            resp = await api_key_chat_api.handle_chat_completions(mock_request)
        assert resp.body is not None

    @pytest.mark.asyncio
    async def test_backend_error_returns_500(self, mock_request):
        mock_request.json.return_value = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with patch(
            "aichat.serve.api_key_chat_api.get_backend",
            side_effect=ValueError("no backend"),
        ):
            resp = await api_key_chat_api.handle_chat_completions(mock_request)
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_error_dict_response(self, mock_request):
        mock_request.json.return_value = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }
        error_dict = {"type": "error", "code": 50000, "content": "boom"}
        with (
            patch("aichat.serve.api_key_chat_api.get_backend") as mock_get_backend,
            patch(
                "aichat.serve.api_key_chat_api.apply_claude_upstream_sampling_params"
            ),
        ):
            backend = mock_get_backend.return_value
            backend.config.upstream_model = "claude"
            backend.build_params.return_value = {}
            backend.converse = AsyncMock(return_value=error_dict)
            resp = await api_key_chat_api.handle_chat_completions(mock_request)
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_streaming_response(self, mock_request):
        mock_request.json.return_value = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        async def gen():
            yield MagicMock()

        with (
            patch("aichat.serve.api_key_chat_api.get_backend") as mock_get_backend,
            patch(
                "aichat.serve.api_key_chat_api.apply_claude_upstream_sampling_params"
            ),
        ):
            backend = mock_get_backend.return_value
            backend.config.upstream_model = "claude"
            backend.build_params.return_value = {}
            backend.converse = AsyncMock(return_value=gen())
            resp = await api_key_chat_api.handle_chat_completions(mock_request)
        assert resp.media_type == "text/event-stream"


class TestStreamChunks:
    @pytest.mark.asyncio
    async def test_yields_chunks_and_done(self):
        chunk = MagicMock()
        chunk.model_dump_json.return_value = '{"id": "1"}'

        async def gen():
            yield chunk

        out = [chunk_out async for chunk_out in api_key_chat_api._stream_chunks(gen())]
        assert out[-1] == "data: [DONE]\n\n"
        assert out[0] == 'data: {"id": "1"}\n\n'

    @pytest.mark.asyncio
    async def test_unserializable_chunk_logs_warning(self):
        chunk = MagicMock()
        chunk.model_dump_json.side_effect = TypeError("no")

        async def gen():
            yield chunk

        out = [chunk_out async for chunk_out in api_key_chat_api._stream_chunks(gen())]
        assert out == ["data: [DONE]\n\n"]
