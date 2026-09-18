from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException, Request

from aichat.serve import brave_search_api


def _request(origin=None):
    req = MagicMock(spec=Request)
    req.headers = {"Origin": origin} if origin else {}
    req.query_params = "q=test"
    req.state = MagicMock()
    return req


@pytest.fixture
def allowed(monkeypatch):
    monkeypatch.setattr(
        "aichat.serve.brave_search_api.search_settings.search_api_allowed_cors_origins",
        '["https://allowed.example.com"]',
    )


def test_get_cors_headers_no_origin(allowed):
    assert brave_search_api.get_cors_headers(_request()) is None


def test_get_cors_headers_allowed_origin(allowed):
    headers = brave_search_api.get_cors_headers(
        _request(origin="https://allowed.example.com")
    )
    assert headers["Access-Control-Allow-Origin"] == "https://allowed.example.com"
    assert headers["Access-Control-Allow-Methods"] == "GET, POST, OPTIONS"
    assert headers["Access-Control-Allow-Headers"] == "Content-Type"


def test_get_cors_headers_disallowed_origin(allowed):
    assert (
        brave_search_api.get_cors_headers(_request(origin="https://evil.com")) is None
    )


def test_get_cors_headers_empty_settings(monkeypatch):
    monkeypatch.setattr(
        "aichat.serve.brave_search_api.search_settings.search_api_allowed_cors_origins",
        "",
    )
    assert brave_search_api.get_cors_headers(_request(origin="https://x.com")) is None


@pytest.mark.asyncio
async def test_options_rhfetch_returns_cors_response(allowed):
    response = await brave_search_api.options_rhfetch(
        _request(origin="https://allowed.example.com")
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_rhfetch_success(allowed):
    request = _request()
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {"results": []}
    request.state.httpx_client = AsyncMock()
    request.state.httpx_client.get = AsyncMock(return_value=response_mock)
    result = await brave_search_api.get_rhfetch(request, "some/path")
    assert result.status_code == 200
    assert result.body == b'{"results": []}' or b"results" in result.body


@pytest.mark.asyncio
async def test_get_rhfetch_non_200(allowed):
    request = _request()
    response_mock = MagicMock()
    response_mock.status_code = 503
    request.state.httpx_client = AsyncMock()
    request.state.httpx_client.get = AsyncMock(return_value=response_mock)
    with pytest.raises(HTTPException) as exc:
        await brave_search_api.get_rhfetch(request, "some/path")
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_get_rhfetch_http_error(allowed):
    request = _request()
    request.state.httpx_client = AsyncMock()
    request.state.httpx_client.get = AsyncMock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(HTTPException) as exc:
        await brave_search_api.get_rhfetch(request, "some/path")
    assert exc.value.status_code == 500
