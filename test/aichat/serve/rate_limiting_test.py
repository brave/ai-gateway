from unittest import mock
from unittest.mock import AsyncMock

import pytest

from aichat.serve.rate_limiting import (
    check_and_increment_automatic_mode_daily_count,
    check_automatic_mode_daily_limit,
    check_content_agent_rate_limits,
    check_rate_limit,
    check_route_rate_limit,
    hash_ip_with_salt,
    rate_limit_route,
)

# ---------------------------------------------------------------------------
# IP hashing
# ---------------------------------------------------------------------------


def test_hash_ip_with_salt_is_deterministic():
    h1 = hash_ip_with_salt("192.168.1.100", "test-salt")
    h2 = hash_ip_with_salt("192.168.1.100", "test-salt")
    assert h1 == h2
    assert h1 != "192.168.1.100"


def test_hash_ip_with_salt_different_salts_diverge():
    a = hash_ip_with_salt("192.168.1.100", "salt-a")
    b = hash_ip_with_salt("192.168.1.100", "salt-b")
    assert a != b


def test_hash_ip_with_salt_unknown_is_localhost():
    assert hash_ip_with_salt("UNKNOWN", "s") == hash_ip_with_salt("127.0.0.1", "s")


# ---------------------------------------------------------------------------
# Helpers / fixtures for the aichat-internal-driven path
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_request():
    """A FastAPI-shaped Request mock that carries an httpx_client on
    ``state`` and exposes a few headers/query-params we use in tests."""
    request = mock.Mock()
    request.state = mock.Mock()
    request.state.httpx_client = mock.Mock()
    request.query_params = {}
    return request


@pytest.fixture(autouse=True)
def _mock_internal_salts(monkeypatch):
    """Stub the salts call so each test gets stable hashes without
    needing to patch every call site individually."""

    async def _ok(*_args, **_kwargs):
        return {"epoch": 1, "current": "current-salt", "previous": "previous-salt"}

    monkeypatch.setattr("aichat.serve.internal_client.rate_limit_salts", _ok)
    # Reset the in-process salt cache between tests.
    from aichat.serve import rate_limiting

    rate_limiting._internal_salt_cache.clear()


@pytest.fixture(autouse=True)
def _enable_internal_api(monkeypatch):
    """Enables the internal-API path for these tests (self-host mode
    would otherwise short-circuit to allow)."""
    monkeypatch.setattr(
        "aichat.serve.internal_settings.internal_settings.internal_api_enabled",
        True,
    )


@pytest.mark.asyncio
async def test_fetch_internal_salts_caches_within_epoch(monkeypatch):
    """Repeated calls in the same 24h epoch must hit the cache and issue
    exactly one HTTP call to aichat-internal."""
    from aichat.serve import rate_limiting
    from aichat.serve.rate_limiting import _fetch_internal_salts

    calls = 0

    async def _counting(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {"epoch": 42, "current": "cur", "previous": "prev"}

    monkeypatch.setattr("aichat.serve.internal_client.rate_limit_salts", _counting)
    rate_limiting._internal_salt_cache.clear()

    client = mock.Mock()
    for _ in range(5):
        result = await _fetch_internal_salts(client)
        assert result == ("cur", "prev")
    assert calls == 1


# ---------------------------------------------------------------------------
# check_rate_limit
# ---------------------------------------------------------------------------


@mock.patch("aichat.serve.rate_limiting.model_settings")
@pytest.mark.asyncio
async def test_check_rate_limit_force_error(mock_model_settings, mock_request):
    mock_model_settings.models = {"m": {"free": True, "rate_limit": 10}}
    mock_request.query_params = {"force-error-rate-limit-user": "1"}
    assert (await check_rate_limit(mock_request, "m", False, "ip")).allowed is False


@mock.patch("aichat.serve.rate_limiting.model_settings")
@pytest.mark.asyncio
async def test_check_rate_limit_premium_host_bypasses(
    mock_model_settings, mock_request
):
    mock_model_settings.models = {"m": {"free": True}}
    assert (await check_rate_limit(mock_request, "m", True, "ip")).allowed is True


@mock.patch("aichat.serve.rate_limiting.model_settings")
@pytest.mark.asyncio
async def test_check_rate_limit_non_free_model_bypasses(
    mock_model_settings, mock_request
):
    mock_model_settings.models = {"m": {"free": False}}
    assert (await check_rate_limit(mock_request, "m", False, "ip")).allowed is True


@mock.patch("aichat.serve.rate_limiting.model_settings")
@pytest.mark.asyncio
async def test_check_rate_limit_content_agent_bypasses(
    mock_model_settings, mock_request
):
    mock_model_settings.models = {"m": {"free": True}}
    assert (
        await check_rate_limit(
            mock_request, "m", False, "ip", is_content_agent_request=True
        )
    ).allowed is True


@mock.patch("aichat.serve.internal_client.rate_limit_check", new_callable=AsyncMock)
@mock.patch("aichat.serve.rate_limiting.model_settings")
@pytest.mark.asyncio
async def test_check_rate_limit_calls_internal_when_free_model(
    mock_model_settings, mock_internal, mock_request
):
    mock_model_settings.models = {
        "m": {"free": True, "rate_limit": 10, "rate_limit_interval_seconds": 60}
    }
    mock_internal.return_value = {"allowed": True}
    assert (await check_rate_limit(mock_request, "m", False, "1.2.3.4")).allowed is True
    mock_internal.assert_awaited_once()


@mock.patch("aichat.serve.internal_client.rate_limit_check", new_callable=AsyncMock)
@mock.patch("aichat.serve.rate_limiting.model_settings")
@pytest.mark.asyncio
async def test_check_rate_limit_denies_on_internal_failure(
    mock_model_settings, mock_internal, mock_request
):
    mock_model_settings.models = {
        "m": {"free": True, "rate_limit": 10, "rate_limit_interval_seconds": 60}
    }
    mock_internal.return_value = None
    assert (
        await check_rate_limit(mock_request, "m", False, "1.2.3.4")
    ).allowed is False


@mock.patch("aichat.serve.rate_limiting.model_settings")
@pytest.mark.asyncio
async def test_check_rate_limit_denies_when_no_rate_limit_config(
    mock_model_settings, mock_request
):
    mock_model_settings.models = {"m": {"free": True}}
    assert (await check_rate_limit(mock_request, "m", False, "ip")).allowed is False


@mock.patch("aichat.serve.internal_client.rate_limit_check", new_callable=AsyncMock)
@mock.patch("aichat.serve.rate_limiting.model_settings")
@pytest.mark.asyncio
async def test_check_rate_limit_premium_credential_hits_internal(
    mock_model_settings, mock_internal, mock_request
):
    # Premium host + valid SKU: no longer a local bypass — internal is
    # called with is_premium_request=True and owns the daily cap.
    mock_model_settings.models = {"m": {"free": False}}
    mock_internal.return_value = {"allowed": True}
    result = await check_rate_limit(
        mock_request,
        "m",
        True,
        "1.2.3.4",
        has_valid_premium_credential=True,
    )
    assert result.allowed is True
    mock_internal.assert_awaited_once()
    kwargs = mock_internal.await_args.kwargs
    assert kwargs["is_premium_request"] is True
    assert kwargs["maximum_requests"] == 0
    assert kwargs["interval_in_seconds"] == 0


@mock.patch("aichat.serve.internal_client.rate_limit_check", new_callable=AsyncMock)
@mock.patch("aichat.serve.rate_limiting.model_settings")
@pytest.mark.asyncio
async def test_check_rate_limit_premium_daily_cap_denied(
    mock_model_settings, mock_internal, mock_request
):
    mock_model_settings.models = {"m": {"free": False}}
    mock_internal.return_value = {"allowed": False, "count": 400}
    result = await check_rate_limit(
        mock_request,
        "m",
        True,
        "1.2.3.4",
        has_valid_premium_credential=True,
    )
    assert result.allowed is False


@pytest.mark.asyncio
async def test_check_rate_limit_premium_host_no_credential_still_bypasses(
    mock_request,
):
    # Existing behavior preserved: premium host without a valid SKU
    # doesn't hit the premium bucket (aichat rejects the request later
    # with INVALID_SKU_CREDENTIAL).
    assert (
        await check_rate_limit(
            mock_request,
            "any-model",
            True,
            "ip",
            has_valid_premium_credential=False,
        )
    ).allowed is True


# ---------------------------------------------------------------------------
# check_content_agent_rate_limits
# ---------------------------------------------------------------------------


@mock.patch(
    "aichat.serve.internal_client.rate_limit_content_agent", new_callable=AsyncMock
)
@pytest.mark.asyncio
async def test_check_content_agent_allowed(mock_internal):
    mock_internal.return_value = {"allowed": True}
    assert await check_content_agent_rate_limits("ip", httpx_client=mock.Mock()) is True


@mock.patch(
    "aichat.serve.internal_client.rate_limit_content_agent", new_callable=AsyncMock
)
@pytest.mark.asyncio
async def test_check_content_agent_denies_on_internal_failure(mock_internal):
    mock_internal.return_value = None
    assert (
        await check_content_agent_rate_limits("ip", httpx_client=mock.Mock()) is False
    )


@pytest.mark.asyncio
async def test_check_content_agent_denies_without_httpx_client():
    assert await check_content_agent_rate_limits("ip", httpx_client=None) is False


# ---------------------------------------------------------------------------
# check_automatic_mode_daily_limit (peek)
# ---------------------------------------------------------------------------


@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@pytest.mark.asyncio
async def test_check_automatic_mode_returns_zero_when_disabled(mock_settings):
    mock_settings.rate_limiting_enabled = False
    out = await check_automatic_mode_daily_limit("ip", httpx_client=mock.Mock())
    assert out == {"count": 0, "exceeded": False, "limit": 0}


@mock.patch(
    "aichat.serve.internal_client.rate_limit_automatic_mode_peek",
    new_callable=AsyncMock,
)
@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@pytest.mark.asyncio
async def test_check_automatic_mode_plumbs_limit_through(mock_settings, mock_internal):
    mock_settings.rate_limiting_enabled = True
    mock_settings.automatic_premium_model_daily_response_limit = 5
    mock_internal.return_value = {"count": 2, "exceeded": False, "limit": 5}
    out = await check_automatic_mode_daily_limit("ip", httpx_client=mock.Mock())
    assert out == {"count": 2, "exceeded": False, "limit": 5}


@mock.patch(
    "aichat.serve.internal_client.rate_limit_automatic_mode_peek",
    new_callable=AsyncMock,
)
@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@pytest.mark.asyncio
async def test_check_automatic_mode_fails_closed(mock_settings, mock_internal):
    mock_settings.rate_limiting_enabled = True
    mock_settings.automatic_premium_model_daily_response_limit = 3
    mock_internal.return_value = None
    out = await check_automatic_mode_daily_limit("ip", httpx_client=mock.Mock())
    assert out["exceeded"] is True


# ---------------------------------------------------------------------------
# check_and_increment_automatic_mode_daily_count
# ---------------------------------------------------------------------------


@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@pytest.mark.asyncio
async def test_check_and_increment_automatic_returns_true_when_disabled(mock_settings):
    mock_settings.rate_limiting_enabled = False
    assert await check_and_increment_automatic_mode_daily_count("ip") is True


@mock.patch(
    "aichat.serve.internal_client.rate_limit_automatic_mode_check",
    new_callable=AsyncMock,
)
@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@pytest.mark.asyncio
async def test_check_and_increment_automatic_returns_internal_verdict(
    mock_settings, mock_internal
):
    mock_settings.rate_limiting_enabled = True
    mock_settings.automatic_premium_model_daily_response_limit = 3
    mock_internal.return_value = {"allowed": True, "count": 1}
    assert (
        await check_and_increment_automatic_mode_daily_count(
            "ip", httpx_client=mock.Mock()
        )
        is True
    )


@mock.patch(
    "aichat.serve.internal_client.rate_limit_automatic_mode_check",
    new_callable=AsyncMock,
)
@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@pytest.mark.asyncio
async def test_check_and_increment_automatic_fails_closed(mock_settings, mock_internal):
    mock_settings.rate_limiting_enabled = True
    mock_settings.automatic_premium_model_daily_response_limit = 3
    mock_internal.return_value = None
    assert (
        await check_and_increment_automatic_mode_daily_count(
            "ip", httpx_client=mock.Mock()
        )
        is False
    )


# ---------------------------------------------------------------------------
# check_route_rate_limit
# ---------------------------------------------------------------------------


@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@pytest.mark.asyncio
async def test_check_route_rate_limit_returns_true_when_disabled(mock_settings):
    mock_settings.rate_limiting_enabled = False
    assert await check_route_rate_limit("/r", "ip", daily_limit=10) is True


@mock.patch("aichat.serve.internal_client.rate_limit_route", new_callable=AsyncMock)
@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@pytest.mark.asyncio
async def test_check_route_rate_limit_returns_internal_verdict(
    mock_settings, mock_internal
):
    mock_settings.rate_limiting_enabled = True
    mock_internal.return_value = {"allowed": True, "count": 1}
    assert (
        await check_route_rate_limit(
            "/r", "ip", daily_limit=10, httpx_client=mock.Mock()
        )
        is True
    )


@mock.patch("aichat.serve.internal_client.rate_limit_route", new_callable=AsyncMock)
@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@pytest.mark.asyncio
async def test_check_route_rate_limit_fails_closed(mock_settings, mock_internal):
    mock_settings.rate_limiting_enabled = True
    mock_internal.return_value = None
    assert (
        await check_route_rate_limit(
            "/r", "ip", daily_limit=10, httpx_client=mock.Mock()
        )
        is False
    )


# ---------------------------------------------------------------------------
# rate_limit_route decorator
# ---------------------------------------------------------------------------


@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@mock.patch("aichat.serve.rate_limiting.check_route_rate_limit", new_callable=AsyncMock)
@mock.patch("aichat.serve.rate_limiting.get_real_ip")
@pytest.mark.asyncio
async def test_rate_limit_route_decorator_with_explicit_limit(
    mock_get_ip, mock_check_limit, mock_settings
):
    from fastapi import Request

    mock_settings.rate_limiting_enabled = True
    mock_get_ip.return_value = "192.168.1.100"
    mock_check_limit.return_value = True

    request = mock.Mock(spec=Request)
    request.headers.get.return_value = "192.168.1.100"
    request.url.path = "/test"

    @rate_limit_route(daily_limit=100)
    async def endpoint(request: Request):
        return {"ok": True}

    assert await endpoint(request) == {"ok": True}
    mock_check_limit.assert_awaited_once_with(
        "/test",
        "192.168.1.100",
        daily_limit=100,
        httpx_client=request.state.httpx_client,
        config_key=None,
    )


@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@mock.patch("aichat.serve.rate_limiting.check_route_rate_limit", new_callable=AsyncMock)
@mock.patch("aichat.serve.rate_limiting.get_real_ip")
@pytest.mark.asyncio
async def test_rate_limit_route_decorator_with_config_key(
    mock_get_ip, mock_check_limit, mock_settings
):
    from fastapi import Request

    mock_settings.rate_limiting_enabled = True
    mock_get_ip.return_value = "192.168.1.100"
    mock_check_limit.return_value = True

    request = mock.Mock(spec=Request)
    request.headers.get.return_value = "192.168.1.100"
    request.url.path = "/rhfetch"

    @rate_limit_route(config_key="rhfetch", route_path="/rhfetch")
    async def endpoint(request: Request):
        return {"ok": True}

    assert await endpoint(request) == {"ok": True}
    mock_check_limit.assert_awaited_once_with(
        "/rhfetch",
        "192.168.1.100",
        daily_limit=None,
        httpx_client=request.state.httpx_client,
        config_key="rhfetch",
    )


@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@mock.patch("aichat.serve.rate_limiting.check_route_rate_limit", new_callable=AsyncMock)
@mock.patch("aichat.serve.rate_limiting.get_real_ip")
@pytest.mark.asyncio
async def test_rate_limit_route_decorator_raises_when_limited(
    mock_get_ip, mock_check_limit, mock_settings
):
    from fastapi import HTTPException, Request

    mock_settings.rate_limiting_enabled = True
    mock_get_ip.return_value = "192.168.1.100"
    mock_check_limit.return_value = False

    request = mock.Mock(spec=Request)
    request.headers.get.return_value = "192.168.1.100"
    request.url.path = "/test"

    @rate_limit_route(daily_limit=100, error_message="exceeded")
    async def endpoint(request: Request):
        return {"ok": True}

    with pytest.raises(HTTPException) as exc:
        await endpoint(request)
    assert exc.value.status_code == 429
    assert exc.value.detail == "exceeded"


@mock.patch("aichat.serve.rate_limiting.check_route_rate_limit", new_callable=AsyncMock)
@mock.patch("aichat.serve.rate_limiting.rate_limiting_settings")
@pytest.mark.asyncio
async def test_rate_limit_route_decorator_skips_when_globally_disabled(
    mock_settings, mock_check_limit
):
    """When RATE_LIMITING_ENABLED is false the decorator must not
    touch the request — callers in unit tests may pass a MagicMock."""
    from fastapi import Request

    mock_settings.rate_limiting_enabled = False
    request = mock.Mock(spec=Request)

    @rate_limit_route(config_key="anything")
    async def endpoint(request: Request):
        return {"ok": True}

    assert await endpoint(request) == {"ok": True}
    mock_check_limit.assert_not_awaited()


@pytest.mark.asyncio
async def test_rate_limit_route_decorator_no_limit_no_config_is_noop():
    @rate_limit_route()
    async def endpoint():
        return {"ok": True}

    assert await endpoint() == {"ok": True}
