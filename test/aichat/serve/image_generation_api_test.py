"""Tests for image generation API endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from aichat.serve.image_generation_api import v1_images_generations


class TestImageGenerationAPI:
    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI Request object"""
        request = MagicMock(spec=Request)
        request.json = AsyncMock()
        return request

    @pytest.fixture
    def mock_image_response(self):
        """Mock ImageResponse from litellm"""
        response = MagicMock()
        response.created = 1234567890
        response.data = [
            MagicMock(url="https://example.com/image.png", revised_prompt="A library")
        ]
        return response

    @pytest.mark.asyncio
    async def test_successful_image_generation(self, mock_request, mock_image_response):
        """Test successful image generation request"""
        mock_request.json.return_value = {
            "model": "test-image-model",
            "prompt": "A library where the books are alive",
            "size": "1024x1024",
        }

        with patch("aichat.serve.image_generation_api.model_settings") as mock_settings:
            mock_settings.models = {"test-image-model": {"type": "image_generation"}}

            with patch(
                "aichat.serve.image_generation_api.generate_image"
            ) as mock_generate:
                mock_generate.return_value = mock_image_response

                response = await v1_images_generations(mock_request)

                # Response should be the ImageResponse object (passthrough)
                assert response == mock_image_response
                mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_model_parameter(self, mock_request):
        """Test request without model parameter"""
        mock_request.json.return_value = {
            "prompt": "A library where the books are alive",
        }

        response = await v1_images_generations(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        response_data = response.body.decode()
        assert "model parameter is required" in response_data

    @pytest.mark.asyncio
    async def test_model_not_found(self, mock_request):
        """Test request with non-existent model"""
        mock_request.json.return_value = {
            "model": "non-existent-model",
            "prompt": "A library where the books are alive",
        }

        with patch("aichat.serve.image_generation_api.model_settings") as mock_settings:
            mock_settings.models = {"test-image-model": {}}

            response = await v1_images_generations(mock_request)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 404
            response_data = response.body.decode()
            assert "model is not supported" in response_data

    @pytest.mark.asyncio
    async def test_invalid_json(self, mock_request):
        """Test request with invalid JSON"""
        mock_request.json.side_effect = ValueError("Invalid JSON")

        response = await v1_images_generations(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        response_data = response.body.decode()
        assert "Invalid JSON" in response_data

    @pytest.mark.asyncio
    async def test_image_generation_error(self, mock_request):
        """Test image generation failure"""
        mock_request.json.return_value = {
            "model": "test-image-model",
            "prompt": "A library where the books are alive",
        }

        with patch("aichat.serve.image_generation_api.model_settings") as mock_settings:
            mock_settings.models = {"test-image-model": {"type": "image_generation"}}

            with patch(
                "aichat.serve.image_generation_api.generate_image"
            ) as mock_generate:
                mock_generate.side_effect = Exception("Image generation failed")

                response = await v1_images_generations(mock_request)

                assert isinstance(response, JSONResponse)
                assert response.status_code == 500
                response_data = response.body.decode()
                assert "Image generation failed" in response_data

    @pytest.mark.asyncio
    async def test_passthrough_parameters(self, mock_request, mock_image_response):
        """Test that all parameters are passed through to generate_image"""
        request_body = {
            "model": "test-image-model",
            "prompt": "A futuristic city",
            "size": "1024x1024",
            "num_inference_steps": 8,
            "seed": 42,
            "n": 2,
            "response_format": "url",
            "quality": "hd",
            "style": "vivid",
        }
        mock_request.json.return_value = request_body

        with patch("aichat.serve.image_generation_api.model_settings") as mock_settings:
            mock_settings.models = {"test-image-model": {"type": "image_generation"}}

            with patch(
                "aichat.serve.image_generation_api.generate_image"
            ) as mock_generate:
                mock_generate.return_value = mock_image_response

                await v1_images_generations(mock_request)

                # Verify all parameters were passed through
                mock_generate.assert_called_once_with(**request_body)

    @pytest.mark.asyncio
    @patch("aichat.serve.rate_limiting.rate_limiting_settings")
    @patch("aichat.serve.rate_limiting.check_route_rate_limit", new_callable=AsyncMock)
    @patch("aichat.serve.rate_limiting.get_real_ip")
    async def test_rate_limit_exceeded(
        self, mock_get_ip, mock_check_limit, mock_settings, mock_request
    ):
        mock_settings.rate_limiting_enabled = True
        mock_get_ip.return_value = "192.168.1.100"
        mock_check_limit.return_value = False
        mock_request.headers.get.return_value = "192.168.1.100"
        mock_request.url.path = "/v1/images/generations"
        mock_request.json.return_value = {
            "model": "test-image-model",
            "prompt": "A library where the books are alive",
        }

        with pytest.raises(HTTPException) as exc_info:
            await v1_images_generations(mock_request)

        assert exc_info.value.status_code == 429
        assert "image generation" in exc_info.value.detail.lower()
        mock_check_limit.assert_awaited_once_with(
            "/images/generations",
            "192.168.1.100",
            daily_limit=None,
            httpx_client=mock_request.state.httpx_client,
            config_key="image_generation",
        )


class TestImageGenerationAPIIntegration:
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

    @pytest.fixture
    def mock_image_response(self):
        """Mock ImageResponse from litellm"""
        response = MagicMock()
        response.created = 1234567890
        response.data = [{"url": "https://example.com/image.png"}]
        # Make it JSON serializable
        response.model_dump = MagicMock(
            return_value={
                "created": 1234567890,
                "data": [{"url": "https://example.com/image.png"}],
            }
        )
        return response

    def test_endpoint_missing_model(self, client):
        """Test endpoint with missing model parameter"""
        response = client.post(
            "/v1/images/generations",
            json={"prompt": "A library where books are alive"},
        )

        assert response.status_code == 400

    def test_endpoint_model_not_found(self, client):
        """Test endpoint with non-existent model"""
        with patch("aichat.serve.image_generation_api.model_settings") as mock_settings:
            mock_settings.models = {}

            response = client.post(
                "/v1/images/generations",
                json={
                    "model": "non-existent",
                    "prompt": "A library where books are alive",
                },
            )

            assert response.status_code == 404
