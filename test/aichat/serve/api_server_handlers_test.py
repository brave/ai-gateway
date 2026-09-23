from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from aichat.serve import api_server


def _request(path="/v1/chat/completions", api_key=None):
    req = MagicMock(spec=Request)
    req.url = MagicMock()
    req.url.path = path
    req.headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    req.state = MagicMock()
    return req


@pytest.mark.asyncio
async def test_api_key_dispatch_invalid_key(monkeypatch):
    monkeypatch.setattr(api_server.server_settings, "api_key_chat_enabled", True)
    call_next = AsyncMock()
    request = _request(api_key="brv_live_not-valid")
    response = await api_server.api_key_dispatch(request, call_next)
    assert response.status_code == 401
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_key_dispatch_other_path(monkeypatch):
    monkeypatch.setattr(api_server.server_settings, "api_key_chat_enabled", True)
    sentinel = MagicMock()
    call_next = AsyncMock(return_value=sentinel)
    response = await api_server.api_key_dispatch(_request(path="/other"), call_next)
    assert response is sentinel
    call_next.assert_awaited_once()


@pytest.mark.asyncio
async def test_api_key_dispatch_disabled(monkeypatch):
    monkeypatch.setattr(api_server.server_settings, "api_key_chat_enabled", False)
    sentinel = MagicMock()
    call_next = AsyncMock(return_value=sentinel)
    response = await api_server.api_key_dispatch(
        _request(api_key="brv_live_" + "a" * 32), call_next
    )
    assert response is sentinel


@pytest.mark.asyncio
async def test_api_key_dispatch_valid_key(monkeypatch):
    monkeypatch.setattr(api_server.server_settings, "api_key_chat_enabled", True)
    sentinel = MagicMock()
    handle = AsyncMock(return_value=sentinel)
    with patch.object(api_server.api_key_chat_api, "handle_chat_completions", handle):
        response = await api_server.api_key_dispatch(
            _request(api_key="brv_live_" + "a" * 32), AsyncMock()
        )
    assert response is sentinel


@pytest.mark.asyncio
async def test_validation_exception_handler_overflow():
    request = _request()
    response = await api_server.validation_exception_handler(
        request, OverflowError("x")
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_request_validation_error_handler():
    request = _request()
    exc = MagicMock()
    exc.errors.return_value = [{"type": "missing", "msg": "field required"}]
    with patch.object(api_server.sentry_sdk, "capture_exception"):
        response = await api_server.request_validation_exception_handler(request, exc)
    assert response.status_code == 400
    assert "field required" in response.body.decode()


@pytest.mark.asyncio
async def test_exception_handler_non_production(monkeypatch):
    request = _request()
    with (
        patch.object(api_server.server_settings, "env", "dev"),
        patch.object(api_server.sentry_sdk, "capture_exception"),
    ):
        response = await api_server.exception_handler(request, RuntimeError("boom"))
    assert response.status_code == 500
    assert "boom" in response.body.decode()


@pytest.mark.asyncio
async def test_exception_handler_production_message(monkeypatch):
    request = _request()
    with (
        patch.object(api_server.server_settings, "env", api_server.PRODUCTION),
        patch.object(api_server.sentry_sdk, "capture_exception"),
    ):
        response = await api_server.exception_handler(request, RuntimeError("secret"))
    assert "Internal Server Error" in response.body.decode()
    assert "secret" not in response.body.decode()


@pytest.mark.asyncio
async def test_value_error_handler():
    request = _request()
    response = await api_server.value_error_handler(request, ValueError("bad"))
    assert response.status_code == 400
