import pytest

from aichat.serve.dependencies import discard_pool


@pytest.fixture(autouse=True)
def _mock_internal_auth(monkeypatch):
    monkeypatch.setattr(
        "aichat.serve.internal_settings.internal_settings.internal_api_enabled",
        True,
    )

    async def _mock_auth_verify(*args, **kwargs):
        return {
            "service_key_allowed": True,
            "service_key_id": "test",
            "brave_key_allowed": True,
        }

    async def _mock_sku_verify(*args, **kwargs):
        return {"allowed": True}

    monkeypatch.setattr("aichat.serve.internal_client.auth_verify", _mock_auth_verify)
    monkeypatch.setattr("aichat.serve.internal_client.sku_verify", _mock_sku_verify)
    # Rate-limit helpers return permissive verdicts by default so tests
    # that don't specifically exercise rate limiting see the same
    # behavior as before. Tests that care can override per-test.
    permissive = {
        "rate_limit_salts": {
            "epoch": 0,
            "current": "current-salt",
            "previous": "previous-salt",
        },
        "rate_limit_check": {"allowed": True},
        "rate_limit_content_agent": {"allowed": True},
        "rate_limit_route": {"allowed": True, "count": 0},
        "rate_limit_automatic_mode_peek": {
            "count": 0,
            "exceeded": False,
            "limit": 3,
        },
        "rate_limit_automatic_mode_check": {"allowed": True, "count": 1},
    }
    for name, payload in permissive.items():

        async def _stub(*_args, _payload=payload, **_kwargs):
            return _payload

        monkeypatch.setattr(f"aichat.serve.internal_client.{name}", _stub)


@pytest.fixture(autouse=True)
def _reset_redis_pool():
    """Discard the async Redis connection pool after each test so that
    connections created on one event loop are not reused on a different one
    (e.g. between successive ``TestClient`` invocations)."""
    yield
    discard_pool()
