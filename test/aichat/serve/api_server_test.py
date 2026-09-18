from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aichat.serve import api_server
from aichat.serve.api_server import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_ready_when_ready(client: TestClient):
    """Test /health/ready returns 200 after normal app startup."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_health_ready_when_not_ready(client: TestClient):
    """Test /health/ready returns 503 when app is not ready."""
    app.state.ready = False
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not ready"}


def test_info_returns_version(client: TestClient):
    """/info returns 200 with a version key."""
    with patch.object(api_server, "VERSION", "abc1234"):
        response = client.get("/info")
    assert response.status_code == 200
    assert response.json() == {"version": "abc1234"}


def test_info_does_not_depend_on_readiness(client: TestClient):
    """/info returns 200 even when /health/ready would return 503."""
    app.state.ready = False
    response = client.get("/info")
    assert response.status_code == 200
    assert "version" in response.json()


@pytest.mark.parametrize(
    "content_part",
    [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        },
        {
            "type": "video_url",
            "video_url": {"url": "https://example.com/video.mp4"},
        },
        {
            "type": "file_url",
            "file_url": {"url": "https://example.com/document.pdf"},
        },
        {
            "type": "file",
            "file": {
                "filename": "document.pdf",
                "file_data": "https://example.com/document.pdf",
            },
        },
    ],
)
def test_attachment_validation_returns_specific_error(
    client: TestClient, content_part: dict
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [content_part],
                }
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "Message attachments must use inline data URLs"
    )
