"""Tests for embeddings API endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

from aichat.serve.embeddings_api import v1_embeddings


class TestEmbeddingsAPI:
    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI Request object"""
        request = MagicMock(spec=Request)
        request.json = AsyncMock()
        return request

    @pytest.mark.asyncio
    async def test_successful_embeddings_generation(self, mock_request):
        """Test successful embeddings generation request"""
        mock_request.json.return_value = {
            "model": "text_embedding",
            "input": ["hello how are you", "what is your name"],
        }

        with patch("aichat.serve.embeddings_api.model_settings") as mock_settings:
            mock_settings.models = {"text_embedding": {"type": "embedding"}}

            with patch(
                "aichat.serve.embeddings_api.generate_embeddings"
            ) as mock_generate:
                mock_generate.return_value = {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": [0.1] * 384,
                            "index": 0,
                        },
                        {
                            "object": "embedding",
                            "embedding": [0.2] * 384,
                            "index": 1,
                        },
                    ],
                    "model": "text_embedding",
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                }

                response = await v1_embeddings(mock_request)

                assert isinstance(response, JSONResponse)
                assert response.status_code == 200
                mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_model_parameter(self, mock_request):
        """Test request without model parameter"""
        mock_request.json.return_value = {
            "input": "hello",
        }

        response = await v1_embeddings(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        response_data = response.body.decode()
        assert "model parameter is required" in response_data

    @pytest.mark.asyncio
    async def test_missing_input_parameter(self, mock_request):
        """Test request without input parameter"""
        mock_request.json.return_value = {
            "model": "text_embedding",
        }

        response = await v1_embeddings(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        response_data = response.body.decode()
        assert "input parameter is required" in response_data

    @pytest.mark.asyncio
    async def test_model_not_found(self, mock_request):
        """Test request with non-existent model"""
        mock_request.json.return_value = {
            "model": "non-existent-model",
            "input": "hello",
        }

        with patch("aichat.serve.embeddings_api.model_settings") as mock_settings:
            mock_settings.models = {"text_embedding": {}}

            response = await v1_embeddings(mock_request)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 404
            response_data = response.body.decode()
            assert "model is not supported" in response_data

    @pytest.mark.asyncio
    async def test_invalid_json(self, mock_request):
        """Test request with invalid JSON"""
        mock_request.json.side_effect = ValueError("Invalid JSON")

        response = await v1_embeddings(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        response_data = response.body.decode()
        assert "Invalid JSON" in response_data

    @pytest.mark.asyncio
    async def test_embeddings_generation_error(self, mock_request):
        """Test embeddings generation failure"""
        mock_request.json.return_value = {
            "model": "text_embedding",
            "input": "hello",
        }

        with patch("aichat.serve.embeddings_api.model_settings") as mock_settings:
            mock_settings.models = {"text_embedding": {"type": "embedding"}}

            with patch(
                "aichat.serve.embeddings_api.generate_embeddings"
            ) as mock_generate:
                mock_generate.side_effect = Exception("Embeddings generation failed")

                response = await v1_embeddings(mock_request)

                assert isinstance(response, JSONResponse)
                assert response.status_code == 500
                response_data = response.body.decode()
                assert "Embeddings generation failed" in response_data

    @pytest.mark.asyncio
    async def test_passthrough_parameters_list(self, mock_request):
        """Test that input and model are passed correctly to generate_embeddings with list input"""
        request_body = {
            "model": "text_embedding",
            "input": ["hello how are you", "what is your name"],
        }
        mock_request.json.return_value = request_body

        with patch("aichat.serve.embeddings_api.model_settings") as mock_settings:
            mock_settings.models = {"text_embedding": {"type": "embedding"}}

            with patch(
                "aichat.serve.embeddings_api.generate_embeddings"
            ) as mock_generate:
                mock_generate.return_value = {
                    "object": "list",
                    "data": [],
                    "model": "text_embedding",
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                }

                await v1_embeddings(mock_request)

                # Verify model and input are passed correctly
                mock_generate.assert_called_once()
                call_kwargs = mock_generate.call_args[1]
                assert call_kwargs["model"] == "text_embedding"
                assert call_kwargs["input"] == [
                    "hello how are you",
                    "what is your name",
                ]

    @pytest.mark.asyncio
    async def test_passthrough_parameters_string(self, mock_request):
        """Test that single string input is passed correctly to generate_embeddings"""
        request_body = {
            "model": "text_embedding",
            "input": "hello how are you",
        }
        mock_request.json.return_value = request_body

        with patch("aichat.serve.embeddings_api.model_settings") as mock_settings:
            mock_settings.models = {"text_embedding": {"type": "embedding"}}

            with patch(
                "aichat.serve.embeddings_api.generate_embeddings"
            ) as mock_generate:
                mock_generate.return_value = {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": [0.1] * 384,
                            "index": 0,
                        }
                    ],
                    "model": "text_embedding",
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                }

                await v1_embeddings(mock_request)

                # Verify model and input are passed correctly
                mock_generate.assert_called_once()
                call_kwargs = mock_generate.call_args[1]
                assert call_kwargs["model"] == "text_embedding"
                assert call_kwargs["input"] == "hello how are you"


class TestEmbeddingsAPIIntegration:
    """Integration tests using TestClient."""

    @pytest.fixture
    def client(self):
        """Test client fixture"""
        from fastapi.testclient import TestClient

        from aichat.serve.api_server import app
        from aichat.serve.auth import check_x_brave_key

        app.dependency_overrides[check_x_brave_key] = lambda: None
        with TestClient(app) as test_client:
            yield test_client
        app.dependency_overrides.clear()

    def test_endpoint_missing_model(self, client):
        """Test endpoint with missing model parameter"""
        response = client.post(
            "/v1/embeddings",
            json={"input": "hello"},
        )

        assert response.status_code == 400

    def test_endpoint_missing_input(self, client):
        """Test endpoint with missing input parameter"""
        response = client.post(
            "/v1/embeddings",
            json={"model": "text_embedding"},
        )

        assert response.status_code == 400

    def test_endpoint_model_not_found(self, client):
        """Test endpoint with non-existent model"""
        with patch("aichat.serve.embeddings_api.model_settings") as mock_settings:
            mock_settings.models = {}

            response = client.post(
                "/v1/embeddings",
                json={"model": "non-existent", "input": "hello"},
            )

            assert response.status_code == 404
