"""Tests for System One /v1/systemone API."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

from aichat.serve.system_one_api import v1_systemone

MODEL_KEY = "system_one_triton"


class TestSystemOneAPI:
    @pytest.fixture
    def mock_request(self):
        request = MagicMock(spec=Request)
        request.json = AsyncMock()
        request.headers = {}
        return request

    @pytest.mark.asyncio
    async def test_successful_systemone(self, mock_request):
        mock_request.json.return_value = {
            "model": MODEL_KEY,
            "state": {"body": "refund please"},
            "questions": {
                "refund": {
                    "type": "noul",
                    "instructions": "Refund requested?",
                }
            },
        }

        with patch("aichat.serve.system_one_api.model_settings") as mock_settings:
            mock_settings.models = {MODEL_KEY: {"type": "system_one"}}
            with patch(
                "aichat.serve.system_one_api.run_system_one",
                new_callable=AsyncMock,
            ) as mock_run:
                mock_run.return_value = {
                    "model": MODEL_KEY,
                    "answers": {"refund": {"type": "noul", "noul": 0.95}},
                    "usage": {"input_tokens": 12, "output_tokens": 0},
                }
                response = await v1_systemone(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 200
        mock_run.assert_awaited_once()
        assert mock_run.call_args.kwargs["model_id"] == MODEL_KEY

    @pytest.mark.asyncio
    async def test_missing_state(self, mock_request):
        mock_request.json.return_value = {
            "model": MODEL_KEY,
            "questions": {},
        }

        with patch("aichat.serve.system_one_api.model_settings") as mock_settings:
            mock_settings.models = {MODEL_KEY: {"type": "system_one"}}
            response = await v1_systemone(mock_request)

        assert response.status_code == 400
        assert "state" in response.body.decode()

    @pytest.mark.asyncio
    async def test_wrong_model_type(self, mock_request):
        mock_request.json.return_value = {
            "model": MODEL_KEY,
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "y"}},
        }

        with patch("aichat.serve.system_one_api.model_settings") as mock_settings:
            mock_settings.models = {MODEL_KEY: {"type": "embedding"}}
            response = await v1_systemone(mock_request)

        assert response.status_code == 404
