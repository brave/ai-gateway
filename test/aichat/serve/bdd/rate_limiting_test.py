import asyncio
from unittest import mock

import pytest
from fastapi import HTTPException
from pytest_bdd import given, parsers, scenarios, then, when
from starlette.requests import Request

from aichat.serve import internal_client, rate_limiting
from aichat.serve.internal_settings import internal_settings
from aichat.serve.rate_limiting import (
    check_and_increment_automatic_mode_daily_count,
    check_automatic_mode_daily_limit,
    check_content_agent_rate_limits,
    check_rate_limit,
    check_route_rate_limit,
    hash_ip_with_salt,
    rate_limit_route,
)
from aichat.serve.rate_limiting_settings import rate_limiting_settings

FEATURE = "features/rate_limiting.feature"

# TODO: remove test/aichat/serve/rate_limiting_test.py#test_hash_ip_with_salt_is_deterministic test/aichat/serve/rate_limiting_test.py:21
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_hash_ip_with_salt_different_salts_diverge test/aichat/serve/rate_limiting_test.py:28
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_hash_ip_with_salt_unknown_is_localhost test/aichat/serve/rate_limiting_test.py:34
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_fetch_internal_salts_caches_within_epoch test/aichat/serve/rate_limiting_test.py:80
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_rate_limit_force_error test/aichat/serve/rate_limiting_test.py:110
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_rate_limit_premium_host_bypasses test/aichat/serve/rate_limiting_test.py:118
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_rate_limit_non_free_model_bypasses test/aichat/serve/rate_limiting_test.py:127
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_rate_limit_content_agent_bypasses test/aichat/serve/rate_limiting_test.py:136
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_rate_limit_calls_internal_when_free_model test/aichat/serve/rate_limiting_test.py:150
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_rate_limit_denies_on_internal_failure test/aichat/serve/rate_limiting_test.py:164
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_rate_limit_denies_when_no_rate_limit_config test/aichat/serve/rate_limiting_test.py:178
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_rate_limit_premium_credential_hits_internal test/aichat/serve/rate_limiting_test.py:188
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_rate_limit_premium_daily_cap_denied test/aichat/serve/rate_limiting_test.py:213
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_rate_limit_premium_host_no_credential_still_bypasses test/aichat/serve/rate_limiting_test.py:229
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_content_agent_allowed test/aichat/serve/rate_limiting_test.py:255
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_content_agent_denies_on_internal_failure test/aichat/serve/rate_limiting_test.py:264
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_content_agent_denies_without_httpx_client test/aichat/serve/rate_limiting_test.py:272
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_automatic_mode_returns_zero_when_disabled test/aichat/serve/rate_limiting_test.py:283
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_automatic_mode_plumbs_limit_through test/aichat/serve/rate_limiting_test.py:295
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_automatic_mode_fails_closed test/aichat/serve/rate_limiting_test.py:309
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_and_increment_automatic_returns_true_when_disabled test/aichat/serve/rate_limiting_test.py:324
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_and_increment_automatic_returns_internal_verdict test/aichat/serve/rate_limiting_test.py:335
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_and_increment_automatic_fails_closed test/aichat/serve/rate_limiting_test.py:355
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_route_rate_limit_returns_true_when_disabled test/aichat/serve/rate_limiting_test.py:374
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_route_rate_limit_returns_internal_verdict test/aichat/serve/rate_limiting_test.py:382
# TODO: remove test/aichat/serve/rate_limiting_test.py#test_check_route_rate_limit_fails_closed test/aichat/serve/rate_limiting_test.py:398
scenarios(FEATURE)


@pytest.fixture
def ctx():
    return {"salt_calls": 0, "last_call": None, "error": None}


@pytest.fixture(autouse=True)
def _clear_salt_cache():
    rate_limiting._internal_salt_cache.clear()
    yield
    rate_limiting._internal_salt_cache.clear()


def make_request(query_params: dict | None = None) -> mock.Mock:
    request = mock.Mock()
    request.state = mock.Mock()
    request.state.httpx_client = mock.Mock()
    request.query_params = query_params or {}
    return request


@given("the gateway app")
def _():
    pass


@given("the internal API is enabled")
def _(monkeypatch):
    monkeypatch.setattr(internal_settings, "internal_api_enabled", True)


@given("rate limiting is enabled")
def _(monkeypatch):
    monkeypatch.setattr(rate_limiting_settings, "rate_limiting_enabled", True)


@given("rate limiting is disabled")
def _(monkeypatch):
    monkeypatch.setattr(rate_limiting_settings, "rate_limiting_enabled", False)


@given(parsers.parse("the automatic daily limit is {limit:d}"))
def _(monkeypatch, limit):
    monkeypatch.setattr(
        rate_limiting_settings,
        "automatic_premium_model_daily_response_limit",
        limit,
    )


@given("aichat-internal allows the rate limit check")
def _(monkeypatch, ctx):
    async def _allow(*args, **kwargs):
        ctx["last_call"] = kwargs
        return {"allowed": True}

    monkeypatch.setattr(internal_client, "rate_limit_check", _allow)


@given("aichat-internal denies the rate limit check")
def _(monkeypatch):
    async def _deny(*args, **kwargs):
        return {"allowed": False}

    monkeypatch.setattr(internal_client, "rate_limit_check", _deny)


@given("aichat-internal is unavailable")
def _(monkeypatch):
    async def _down(*args, **kwargs):
        return None

    for name in (
        "rate_limit_check",
        "rate_limit_content_agent",
        "rate_limit_automatic_mode_peek",
        "rate_limit_automatic_mode_check",
        "rate_limit_route",
    ):
        monkeypatch.setattr(internal_client, name, _down)


@given(parsers.parse("aichat-internal peek reports count {count:d} of {limit:d}"))
def _(monkeypatch, ctx, count, limit):
    async def _peek(*args, **kwargs):
        ctx["last_call"] = kwargs
        return {"count": count, "exceeded": False, "limit": limit}

    monkeypatch.setattr(internal_client, "rate_limit_automatic_mode_peek", _peek)


@given("aichat-internal denies the automatic mode check")
def _(monkeypatch):
    async def _deny(*args, **kwargs):
        return {"allowed": False}

    monkeypatch.setattr(internal_client, "rate_limit_automatic_mode_check", _deny)


@given("aichat-internal denies the route check")
def _(monkeypatch):
    async def _deny(*args, **kwargs):
        return {"allowed": False}

    monkeypatch.setattr(internal_client, "rate_limit_route", _deny)


@given(
    parsers.parse(
        'model "{model}" is free with limit {limit:d} per {interval:d} seconds'
    )
)
def _(monkeypatch, model, limit, interval):
    monkeypatch.setattr(
        rate_limiting,
        "model_settings",
        mock.Mock(
            models={
                model: {
                    "free": True,
                    "rate_limit": limit,
                    "rate_limit_interval_seconds": interval,
                }
            }
        ),
    )


@given(parsers.parse('model "{model}" is free without limits'))
def _(monkeypatch, model):
    monkeypatch.setattr(
        rate_limiting, "model_settings", mock.Mock(models={model: {"free": True}})
    )


@given(parsers.parse('model "{model}" is free'))
def _(monkeypatch, model):
    monkeypatch.setattr(
        rate_limiting, "model_settings", mock.Mock(models={model: {"free": True}})
    )


@given(parsers.parse('model "{model}" is not free'))
def _(monkeypatch, model):
    monkeypatch.setattr(
        rate_limiting, "model_settings", mock.Mock(models={model: {"free": False}})
    )


@when(parsers.parse('the ip "{ip}" is hashed with salt "{salt}"'))
def _(ctx, ip, salt):
    ctx.setdefault("hashes", {})[f"{ip}:{salt}"] = hash_ip_with_salt(ip, salt)


@then("the hash is stable across calls")
def _(ctx):
    h = ctx["hashes"]["192.168.1.100:s"]
    assert h == hash_ip_with_salt("192.168.1.100", "s")
    assert h != "192.168.1.100"


@then('hashing with salt "t" produces a different hash')
def _(ctx):
    assert ctx["hashes"]["192.168.1.100:s"] != ctx["hashes"]["192.168.1.100:t"]


@then('the hash equals the hash of "127.0.0.1" with salt "s"')
def _(ctx):
    assert ctx["hashes"]["UNKNOWN:s"] == ctx["hashes"]["127.0.0.1:s"]


@when("five salt lookups happen in a row")
def _(monkeypatch, ctx):
    async def _counting(*args, **kwargs):
        ctx["salt_calls"] += 1
        return {"epoch": 1, "current": "cur", "previous": "prev"}

    monkeypatch.setattr(internal_client, "rate_limit_salts", _counting)
    client = mock.Mock()
    for _ in range(5):
        assert asyncio.run(rate_limiting._fetch_internal_salts(client)) == (
            "cur",
            "prev",
        )


@then("aichat-internal was asked for salts exactly once")
def _(ctx):
    assert ctx["salt_calls"] == 1


@when(
    parsers.parse('a rate limit is checked with query "{query}"'),
    target_fixture="verdict",
)
def _(query):
    request = make_request({"force-error-rate-limit-user": "1"})
    return asyncio.run(check_rate_limit(request, "m", False, "ip"))


@when("a rate limit is checked", target_fixture="verdict")
def _(ctx):
    request = make_request()
    return asyncio.run(check_rate_limit(request, "m", False, "1.2.3.4"))


@when(
    "a rate limit is checked from a premium host without a credential",
    target_fixture="verdict",
)
def _(ctx):
    request = make_request()
    return asyncio.run(
        check_rate_limit(
            request, "any-model", True, "ip", has_valid_premium_credential=False
        )
    )


@when("a premium rate limit is checked", target_fixture="verdict")
def _(ctx):
    request = make_request()
    return asyncio.run(
        check_rate_limit(
            request, "m", True, "1.2.3.4", has_valid_premium_credential=True
        )
    )


@when("a content agent rate limit is checked", target_fixture="verdict")
def _(ctx):
    request = make_request()
    return asyncio.run(
        check_rate_limit(request, "m", False, "ip", is_content_agent_request=True)
    )


@then("the rate limit verdict allows")
def _(verdict):
    assert verdict.allowed is True


@then("the rate limit verdict denies")
def _(verdict):
    assert verdict.allowed is False


@then("the rate limit payload carried the model config")
def _(ctx):
    call = ctx["last_call"]
    assert call["model"] == "m"
    assert call["maximum_requests"] == 10
    assert call["interval_in_seconds"] == 60
    assert call["is_free_model"] is True


@then("the premium bucket was queried with no local cap")
def _(ctx):
    call = ctx["last_call"]
    assert call["is_premium_request"] is True
    assert call["maximum_requests"] == 0
    assert call["interval_in_seconds"] == 0


@when("a content agent limit is checked", target_fixture="content_agent_allowed")
def _(ctx):
    return asyncio.run(check_content_agent_rate_limits("ip", httpx_client=mock.Mock()))


@then("the content agent verdict denies")
def _(content_agent_allowed):
    assert content_agent_allowed is False


@when("the automatic mode daily limit is peeked", target_fixture="peek")
def _(ctx):
    return asyncio.run(check_automatic_mode_daily_limit("ip", httpx_client=mock.Mock()))


@then(
    parsers.parse(
        "the peek result is count {count:d} exceeded {exceeded} limit {limit:d}"
    )
)
def _(peek, count, exceeded, limit):
    assert peek == {
        "count": count,
        "exceeded": exceeded == "true",
        "limit": limit,
    }


@when("the automatic mode count is incremented", target_fixture="increment_allowed")
def _(ctx):
    return asyncio.run(
        check_and_increment_automatic_mode_daily_count("ip", httpx_client=mock.Mock())
    )


@then("the increment verdict denies")
def _(increment_allowed):
    assert increment_allowed is False


@when(
    parsers.parse('a route limit is checked for "{route}"'),
    target_fixture="route_allowed",
)
def _(route):
    return asyncio.run(check_route_rate_limit(route, "ip", httpx_client=mock.Mock()))


@then("the route limit verdict allows")
def _(route_allowed):
    assert route_allowed is True


@then("the route limit verdict denies")
def _(route_allowed):
    assert route_allowed is False


@when("a decorated handler with limit 5 is called", target_fixture="error")
def _(ctx):
    @rate_limit_route(daily_limit=5, route_path="/v1/demo")
    async def handler(request):
        return "ok"

    request = mock.Mock(spec=Request)
    request.headers = {"X-Forwarded-For": "1.2.3.4"}
    request.url.path = "/v1/demo"
    request.state.httpx_client = mock.Mock()
    try:
        asyncio.run(handler(request))
        return None
    except HTTPException as exc:
        return exc


@when("a decorated handler without configuration is called", target_fixture="result")
def _(ctx):
    @rate_limit_route()
    async def handler(request):
        return "ok"

    return asyncio.run(handler(mock.Mock()))


@then(parsers.parse('the handler raised {code:d} "{detail}"'))
def _(error, code, detail):
    assert isinstance(error, HTTPException)
    assert error.status_code == code
    assert error.detail == detail


@then(parsers.parse('the handler returned "{value}"'))
def _(result, value):
    assert result == value
