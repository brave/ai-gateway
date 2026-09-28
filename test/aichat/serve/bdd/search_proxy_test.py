# BDD coverage for aichat/serve/brave_search_api.py and
# aichat/serve/services/search.py (InlineSearchHelper.pop_completed_searches).
#
# TODO: remove test/aichat/serve/brave_search_api_test.py#test_get_cors_headers_no_origin test/aichat/serve/brave_search_api_test.py:26
# TODO: remove test/aichat/serve/brave_search_api_test.py#test_get_cors_headers_allowed_origin test/aichat/serve/brave_search_api_test.py:30
# TODO: remove test/aichat/serve/brave_search_api_test.py#test_get_cors_headers_disallowed_origin test/aichat/serve/brave_search_api_test.py:39
# TODO: remove test/aichat/serve/brave_search_api_test.py#test_options_rhfetch_returns_cors_response test/aichat/serve/brave_search_api_test.py:54
# TODO: remove test/aichat/serve/brave_search_api_test.py#test_get_rhfetch_success test/aichat/serve/brave_search_api_test.py:62
# TODO: remove test/aichat/serve/brave_search_api_test.py#test_get_rhfetch_non_200 test/aichat/serve/brave_search_api_test.py:75
# TODO: remove test/aichat/serve/brave_search_api_test.py#test_get_rhfetch_http_error test/aichat/serve/brave_search_api_test.py:87

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException, Request
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import brave_search_api
from aichat.serve.services.search import InlineSearchHelper
from aichat.serve.services.search_settings import search_settings

FEATURE = "features/search_proxy.feature"
scenarios(FEATURE)


@pytest.fixture
def ctx():
    return {}


def _mock_request(origin=None):
    request = MagicMock(spec=Request)
    headers = {"Origin": origin} if origin else {}
    request.headers = headers
    request.query_params = "q=news"
    return request


@given("the search proxy harness")
def search_proxy_harness(ctx, monkeypatch):
    async def _always_allow(*args, **kwargs):
        return True

    monkeypatch.setattr(
        "aichat.serve.rate_limiting.check_route_rate_limit", _always_allow
    )
    return ctx


@given("allowed cors origins are configured")
def allowed_origins(ctx, monkeypatch):
    monkeypatch.setattr(
        search_settings, "search_api_allowed_cors_origins", '["https://a.com"]'
    )
    monkeypatch.setattr(search_settings, "brave_search_api_url", "https://bsa.ai")
    monkeypatch.setattr(search_settings, "brave_search_rh_api_key", "rh-key")


@when(parsers.parse("the cors headers are computed for origin {origin}"))
def compute_cors(ctx, origin):
    request = _mock_request(None if origin == "none" else origin)
    ctx["cors"] = brave_search_api.get_cors_headers(request)


@then(parsers.parse("the cors headers are {expected}"))
def cors_assert(ctx, expected):
    if expected == "present":
        assert ctx["cors"] is not None
        assert ctx["cors"]["Access-Control-Allow-Origin"] == "https://a.com"
    elif expected == "absent":
        assert ctx["cors"] is None
    else:
        raise AssertionError(f"unknown cors expectation: {expected!r}")


@when(parsers.parse('the rhfetch preflight is requested from "{origin}"'))
def rhfetch_preflight(ctx, origin):
    ctx["response"] = asyncio.run(
        brave_search_api.options_rhfetch(_mock_request(origin))
    )


@then("the preflight carries the allowlisted cors headers")
def preflight_cors_assert(ctx):
    resp = ctx["response"]
    assert resp.headers["Access-Control-Allow-Origin"] == "https://a.com"
    assert resp.headers["Access-Control-Allow-Methods"] == "GET, POST, OPTIONS"


@then("the preflight carries no cors headers")
def preflight_no_cors_assert(ctx):
    assert ctx["response"].headers.get("Access-Control-Allow-Origin") is None


def _fake_httpx_response(status_code=200, payload=None):
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=payload or {})
    return response


@given(parsers.parse("the upstream returns 200 with {payload}"))
def upstream_200(ctx, payload):
    client = MagicMock()
    client.get = AsyncMock(return_value=_fake_httpx_response(200, json.loads(payload)))
    ctx["client"] = client


@given(parsers.parse("the upstream returns {status_code:d}"))
def upstream_non_200(ctx, status_code):
    client = MagicMock()
    client.get = AsyncMock(return_value=_fake_httpx_response(status_code))
    ctx["client"] = client


@given("the upstream transport drops")
def upstream_transport_drops(ctx):
    client = MagicMock()
    client.get = AsyncMock(side_effect=httpx.ConnectError("dropped"))
    ctx["client"] = client


def _capture_http_exception(fn):
    try:
        return {"result": asyncio.run(fn()), "error": None}
    except HTTPException as e:
        return {"result": None, "error": e}


@when(parsers.parse('the rich fetch for "{path}" is requested from "{origin}"'))
def rich_fetch(ctx, path, origin):
    request = _mock_request(origin)
    request.state.httpx_client = ctx["client"]
    ctx["result"] = _capture_http_exception(
        lambda: brave_search_api.get_rhfetch(request, path)
    )


@then(parsers.parse("the proxied body is {payload} and the token header was sent"))
def proxied_body_assert(ctx, payload):
    resp = ctx["result"]["result"]
    assert json.loads(resp.body) == json.loads(payload)
    expected_url = "https://bsa.ai/res/v1/web/rich/fetch/topic/news?q=news"
    ctx["client"].get.assert_awaited_once_with(
        expected_url, headers={"x-subscription-token": "rh-key"}
    )


@then(parsers.parse("the rich fetch raises the upstream failure with {code:d}"))
def rich_fetch_failure_assert(ctx, code):
    error = ctx["result"]["error"]
    assert isinstance(error, HTTPException)
    # 503 propagates the upstream status; transport errors map to 500.
    assert error.status_code == code


# ---------- InlineSearchHelper.pop_completed_searches ----------


@given(parsers.parse("the inline search helper with {kind} and one pending task"))
def inline_helper(ctx, kind):
    ctx["kind"] = kind


async def _finish_with(value):
    await asyncio.sleep(0)
    return value


async def _finish_failed():
    await asyncio.sleep(0)
    raise RuntimeError("search failed")


@when("the completed searches are popped")
def pop_completed(ctx):
    import contextlib

    async def _run():
        helper = InlineSearchHelper(MagicMock())
        loop = asyncio.get_running_loop()
        if ctx["kind"] == "one finished":
            task = loop.create_task(_finish_with("inline-result"))
            await task
        elif ctx["kind"] == "one failed":
            task = loop.create_task(_finish_failed())
            await asyncio.wait([task])
        else:
            raise AssertionError(f"unknown inline-search kind: {ctx['kind']}")
        pending = loop.create_task(asyncio.sleep(1))
        helper.inline_search_promises = [task, pending]
        popped = helper.pop_completed_searches()
        remaining = len(helper.inline_search_promises)
        pending.cancel()
        with contextlib.suppress(BaseException):
            await pending
        return popped, remaining

    ctx["popped"], ctx["remaining"] = asyncio.run(_run())


@then("the finished result is returned and the pending task remains")
def popped_finished_assert(ctx):
    assert ctx["popped"] == ["inline-result"]
    assert ctx["remaining"] == 1


@then("no results are returned and the pending task remains")
def popped_failed_assert(ctx):
    assert ctx["popped"] == []
    assert ctx["remaining"] == 1
