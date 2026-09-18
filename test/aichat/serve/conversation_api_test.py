import json

import pytest
from fastapi.testclient import TestClient

from aichat.serve.api_server import app
from aichat.serve.constants import CONVERSATION_API_DEPRECATION_MESSAGES
from aichat.serve.conversation_api import get_conversation_deprecation_message


class CustomTestClient(TestClient):
    def request(self, *args, **kwargs):
        # Set a default for x-forwarded-for
        default_headers = {
            "x-forwarded-for": "127.0.0.1",
        }
        headers = kwargs.pop("headers", {})
        default_headers.update(headers)
        return super().request(*args, headers=default_headers, **kwargs)


@pytest.fixture
def client():
    with CustomTestClient(app) as test_client:
        yield test_client


class TestConversationDeprecationMessage:
    """
    /v1/conversation is deprecated: it always returns a static "please upgrade"
    notice instead of running auth, rate limiting, model triage, or the LLM.
    """

    def test_defaults_to_english_when_no_system_language(self):
        message = get_conversation_deprecation_message(None)
        assert message == CONVERSATION_API_DEPRECATION_MESSAGES["en"]

    def test_defaults_to_english_for_english_system_language(self):
        message = get_conversation_deprecation_message("en-US")
        assert message == CONVERSATION_API_DEPRECATION_MESSAGES["en"]

    def test_localizes_known_language_and_still_includes_english(self):
        message = get_conversation_deprecation_message("fr-FR")
        assert message == (
            f"{CONVERSATION_API_DEPRECATION_MESSAGES['fr']}\n\n"
            f"{CONVERSATION_API_DEPRECATION_MESSAGES['en']}"
        )

    def test_falls_back_to_english_for_unrecognized_language(self):
        message = get_conversation_deprecation_message("xx-YY")
        assert message == CONVERSATION_API_DEPRECATION_MESSAGES["en"]

    def test_non_streaming_request_returns_static_notice(self, client):
        payload = {
            "model": "automatic",
            "system_language": "de",
            "stream": False,
            "events": [
                {"role": "user", "type": "chatMessage", "content": "hello"},
            ],
        }

        response = client.post("/v1/conversation", json=payload, headers={})

        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "completion"
        assert body["model"] == "automatic"
        assert body["stop_reason"] == "stop_sequence"
        assert body["completion"] == (
            f"{CONVERSATION_API_DEPRECATION_MESSAGES['de']}\n\n"
            f"{CONVERSATION_API_DEPRECATION_MESSAGES['en']}"
        )

    def test_streaming_request_returns_single_sse_event(self, client):
        payload = {
            "model": "mixtral-8x7b-instruct",
            "stream": True,
            "events": [
                {"role": "user", "type": "chatMessage", "content": "hello"},
            ],
        }

        response = client.post("/v1/conversation", json=payload, headers={})

        assert response.status_code == 200
        raw_lines = [line for line in response.text.split("\n\n") if line]
        assert raw_lines[-1] == "data: [DONE]"

        data = json.loads(raw_lines[0][len("data: ") :])
        assert data["type"] == "completion"
        assert data["model"] == "mixtral-8x7b-instruct"
        assert data["stop_reason"] == "stop_sequence"
        assert data["completion"] == CONVERSATION_API_DEPRECATION_MESSAGES["en"]

    def test_does_not_require_auth_headers_or_credentials(self, client):
        """The deprecated endpoint responds without a Brave services key or
        premium credential, since it no longer performs any auth/rate-limit
        gating or model resolution."""
        payload = {
            "model": "claude-3-sonnet",
            "stream": False,
            "events": [
                {"role": "user", "type": "chatMessage", "content": "hello"},
            ],
        }

        response = client.post("/v1/conversation", json=payload, headers={})

        assert response.status_code == 200
        assert (
            response.json()["completion"] == CONVERSATION_API_DEPRECATION_MESSAGES["en"]
        )
