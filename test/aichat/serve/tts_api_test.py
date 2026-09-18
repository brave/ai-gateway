"""Tests for TTS API endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from aichat.serve.tts_api import v1_audio_speech


class TestTTSAPI:
    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI Request object"""
        request = MagicMock(spec=Request)
        request.json = AsyncMock()
        return request

    @pytest.fixture
    def mock_speech_response(self):
        """Mock speech response from litellm"""
        response = MagicMock()
        response.content = b"fake audio content"
        return response

    @pytest.mark.asyncio
    async def test_successful_speech_generation(
        self, mock_request, mock_speech_response
    ):
        """Test successful speech generation request"""
        mock_request.json.return_value = {
            "model": "test-tts-model",
            "input": "Hello, how are you?",
            "voice": "af_heart",
        }

        with patch("aichat.serve.tts_api.model_settings") as mock_settings:
            mock_settings.models = {"test-tts-model": {"type": "tts"}}

            with patch("aichat.serve.tts_api.generate_speech") as mock_generate:
                mock_generate.return_value = mock_speech_response

                response = await v1_audio_speech(mock_request)

                # Response should be a Response object with audio content
                assert isinstance(response, Response)
                assert response.body == b"fake audio content"
                assert response.media_type == "audio/mpeg"  # Default format
                mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_speech_generation_with_format(
        self, mock_request, mock_speech_response
    ):
        """Test speech generation with different response format"""
        mock_request.json.return_value = {
            "model": "test-tts-model",
            "input": "Hello, how are you?",
            "voice": "af_heart",
            "response_format": "opus",
        }

        with patch("aichat.serve.tts_api.model_settings") as mock_settings:
            mock_settings.models = {"test-tts-model": {"type": "tts"}}

            with patch("aichat.serve.tts_api.generate_speech") as mock_generate:
                mock_speech_response.content = b"fake opus audio"
                mock_generate.return_value = mock_speech_response

                response = await v1_audio_speech(mock_request)

                assert isinstance(response, Response)
                assert response.body == b"fake opus audio"
                assert response.media_type == "audio/opus"
                mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_model_parameter(self, mock_request):
        """Test request without model parameter"""
        mock_request.json.return_value = {
            "input": "Hello, how are you?",
            "voice": "af_heart",
        }

        response = await v1_audio_speech(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        response_data = response.body.decode()
        assert "model parameter is required" in response_data

    @pytest.mark.asyncio
    async def test_model_not_found(self, mock_request):
        """Test request with non-existent model"""
        mock_request.json.return_value = {
            "model": "non-existent-model",
            "input": "Hello, how are you?",
            "voice": "af_heart",
        }

        with patch("aichat.serve.tts_api.model_settings") as mock_settings:
            mock_settings.models = {"test-tts-model": {}}

            response = await v1_audio_speech(mock_request)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 404
            response_data = response.body.decode()
            assert "model is not supported" in response_data

    @pytest.mark.asyncio
    async def test_invalid_json(self, mock_request):
        """Test request with invalid JSON"""
        mock_request.json.side_effect = ValueError("Invalid JSON")

        response = await v1_audio_speech(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        response_data = response.body.decode()
        assert "Invalid JSON" in response_data

    @pytest.mark.asyncio
    async def test_speech_generation_error(self, mock_request):
        """Test speech generation failure"""
        mock_request.json.return_value = {
            "model": "test-tts-model",
            "input": "Hello, how are you?",
            "voice": "af_heart",
        }

        with patch("aichat.serve.tts_api.model_settings") as mock_settings:
            mock_settings.models = {"test-tts-model": {"type": "tts"}}

            with patch("aichat.serve.tts_api.generate_speech") as mock_generate:
                mock_generate.side_effect = Exception("Speech generation failed")

                response = await v1_audio_speech(mock_request)

                assert isinstance(response, JSONResponse)
                assert response.status_code == 500
                response_data = response.body.decode()
                assert "Speech generation failed" in response_data

    @pytest.mark.asyncio
    async def test_invalid_response_format(self, mock_request):
        """Test response without content attribute"""
        mock_request.json.return_value = {
            "model": "test-tts-model",
            "input": "Hello, how are you?",
            "voice": "af_heart",
        }

        with patch("aichat.serve.tts_api.model_settings") as mock_settings:
            mock_settings.models = {"test-tts-model": {"type": "tts"}}

            with patch("aichat.serve.tts_api.generate_speech") as mock_generate:
                # Mock response without content attribute
                invalid_response = MagicMock()
                del invalid_response.content
                mock_generate.return_value = invalid_response

                response = await v1_audio_speech(mock_request)

                assert isinstance(response, JSONResponse)
                assert response.status_code == 500
                response_data = response.body.decode()
                assert "Invalid response format" in response_data

    @pytest.mark.asyncio
    async def test_passthrough_parameters(self, mock_request, mock_speech_response):
        """Test that all parameters are passed through to generate_speech"""
        request_body = {
            "model": "test-tts-model",
            "input": "Hello, how are you?",
            "voice": "af_heart",
            "response_format": "mp3",
            "speed": 1.0,
        }
        mock_request.json.return_value = request_body

        with patch("aichat.serve.tts_api.model_settings") as mock_settings:
            mock_settings.models = {"test-tts-model": {"type": "tts"}}

            with patch("aichat.serve.tts_api.generate_speech") as mock_generate:
                mock_generate.return_value = mock_speech_response

                await v1_audio_speech(mock_request)

                # Verify all parameters were passed through
                mock_generate.assert_called_once_with(**request_body)

    @pytest.mark.asyncio
    async def test_streaming_speech_returns_streaming_response(self, mock_request):
        """Test that stream=true returns a StreamingResponse"""
        mock_request.json.return_value = {
            "model": "test-tts-model",
            "input": "Hello, how are you?",
            "voice": "af_heart",
            "stream": True,
        }

        async def fake_chunks():
            yield b"chunk1"
            yield b"chunk2"

        with patch("aichat.serve.tts_api.model_settings") as mock_settings:
            mock_settings.models = {"test-tts-model": {"type": "tts"}}

            with patch(
                "aichat.serve.tts_api.stream_speech",
                new_callable=AsyncMock,
                return_value=fake_chunks(),
            ) as mock_stream:
                response = await v1_audio_speech(mock_request)

                assert isinstance(response, StreamingResponse)
                assert response.media_type == "audio/mpeg"
                mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_audio_formats(self, mock_request, mock_speech_response):
        """Test different audio format content types"""
        formats = [
            ("mp3", "audio/mpeg"),
            ("opus", "audio/opus"),
            ("aac", "audio/aac"),
            ("flac", "audio/flac"),
        ]

        for response_format, expected_content_type in formats:
            mock_request.json.return_value = {
                "model": "test-tts-model",
                "input": "Hello",
                "voice": "af_heart",
                "response_format": response_format,
            }

            with patch("aichat.serve.tts_api.model_settings") as mock_settings:
                mock_settings.models = {"test-tts-model": {"type": "tts"}}

                with patch("aichat.serve.tts_api.generate_speech") as mock_generate:
                    mock_generate.return_value = mock_speech_response

                    response = await v1_audio_speech(mock_request)

                    assert isinstance(response, Response)
                    assert response.media_type == expected_content_type


class TestTTSAPIIntegration:
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
    def mock_speech_response(self):
        """Mock speech response from litellm"""
        response = MagicMock()
        response.content = b"fake audio content"
        return response

    def test_endpoint_missing_model(self, client):
        """Test endpoint with missing model parameter"""
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "Hello, how are you?",
                "voice": "af_heart",
            },
        )

        assert response.status_code == 400

    def test_endpoint_model_not_found(self, client):
        """Test endpoint with non-existent model"""
        with patch("aichat.serve.tts_api.model_settings") as mock_settings:
            mock_settings.models = {}

            response = client.post(
                "/v1/audio/speech",
                json={
                    "model": "non-existent",
                    "input": "Hello, how are you?",
                    "voice": "af_heart",
                },
            )

            assert response.status_code == 404
