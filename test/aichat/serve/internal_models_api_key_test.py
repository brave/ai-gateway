"""Tests for INTERNAL_MODELS_API_KEY gating on model passthrough routes."""

import os

os.environ.setdefault("ENV", "test")

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

from aichat.protocol.open_ai_protocol import ErrorCode
from aichat.serve.common_api import (
    is_valid_internal_models_api_key,
    require_internal_models_api_key,
)
from aichat.serve.embeddings_api import v1_embeddings


class TestInternalModelsApiKeyHelpers:
    def test_unconfigured_allows_anything(self):
        with patch("aichat.serve.common_api.security_settings") as mock_settings:
            mock_settings.internal_models_api_key = ""
            assert is_valid_internal_models_api_key(None)
            assert require_internal_models_api_key(None) is None

    def test_configured_requires_bearer(self):
        with patch("aichat.serve.common_api.security_settings") as mock_settings:
            mock_settings.internal_models_api_key = "team-secret"
            assert not is_valid_internal_models_api_key(None)
            assert not is_valid_internal_models_api_key("Basic abc")
            assert is_valid_internal_models_api_key("Bearer team-secret")
            err = require_internal_models_api_key("Bearer wrong")
            assert err is not None
            assert err.status_code == int(ErrorCode.INVALID_AUTH_KEY / 100)


class TestEmbeddingsApiKeyGate:
    @pytest.fixture
    def mock_request(self):
        request = MagicMock(spec=Request)
        request.json = AsyncMock()
        request.headers = {}
        return request

    @pytest.mark.asyncio
    async def test_rejects_missing_key_when_configured(self, mock_request):
        mock_request.json.return_value = {
            "model": "text_embedding",
            "input": "hello",
        }
        with patch("aichat.serve.common_api.security_settings") as mock_settings:
            mock_settings.internal_models_api_key = "team-secret"
            with patch("aichat.serve.embeddings_api.model_settings") as ms:
                ms.models = {"text_embedding": {"type": "embedding"}}
                resp = await v1_embeddings(mock_request)
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_accepts_valid_bearer(self, mock_request):
        mock_request.headers = {"Authorization": "Bearer team-secret"}
        mock_request.json.return_value = {
            "model": "text_embedding",
            "input": "hello",
        }
        with patch("aichat.serve.common_api.security_settings") as mock_settings:
            mock_settings.internal_models_api_key = "team-secret"
            with patch("aichat.serve.embeddings_api.model_settings") as ms:
                ms.models = {"text_embedding": {"type": "embedding"}}
                with patch(
                    "aichat.serve.embeddings_api.generate_embeddings",
                    new_callable=AsyncMock,
                ) as mock_gen:
                    mock_gen.return_value = {
                        "object": "list",
                        "data": [],
                        "model": "text_embedding",
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                    resp = await v1_embeddings(mock_request)
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 200
