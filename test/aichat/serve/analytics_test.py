from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aichat.serve.analytics import send_analytics_request


@pytest.fixture
def mock_httpx_async_client():
    """Mock httpx async client context manager for analytics requests"""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client.post.return_value = mock_response

    # Mock async context manager
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None

    return mock_context


@pytest.fixture
def mock_app_settings():
    """Mock app_settings for analytics tests"""
    settings = MagicMock()
    settings.analytics_model_address = "http://test.com"
    return settings


class TestAnalyticsEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_extracts_from_openai_messages(
        self, mock_httpx_async_client, mock_app_settings
    ):
        """Should extract text from OpenAI-style messages parameter"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        with (
            patch(
                "aichat.serve.analytics.httpx.AsyncClient",
                return_value=mock_httpx_async_client,
            ),
            patch(
                "aichat.serve.analytics.external_service_settings", mock_app_settings
            ),
        ):
            await send_analytics_request(messages=messages, model="test-model")

        mock_client = await mock_httpx_async_client.__aenter__()
        call_args = mock_client.post.call_args
        analytics_data = call_args[1]["json"]
        assert analytics_data["text"] == "How are you?"
        assert analytics_data["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_extracts_from_openai_content_parts(
        self, mock_httpx_async_client, mock_app_settings
    ):
        """Should join text from OpenAI-style content part dicts"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "write a python script to parse CSV files",
                    },
                    {
                        "type": "text",
                        "text": "This is the text of a web page: <page>some page context</page>.",
                    },
                ],
            }
        ]

        with (
            patch(
                "aichat.serve.analytics.httpx.AsyncClient",
                return_value=mock_httpx_async_client,
            ),
            patch(
                "aichat.serve.analytics.external_service_settings", mock_app_settings
            ),
        ):
            await send_analytics_request(messages=messages, model="test-model")

        mock_client = await mock_httpx_async_client.__aenter__()
        call_args = mock_client.post.call_args
        analytics_data = call_args[1]["json"]
        expected = (
            "write a python script to parse CSV files\n\n"
            "This is the text of a web page: <page>some page context</page>."
        )
        assert analytics_data["text"] == expected
        assert isinstance(analytics_data["text"], str)
