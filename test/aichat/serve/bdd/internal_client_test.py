import asyncio
import json

import httpx
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import internal_client
from aichat.serve.internal_settings import internal_settings

# Capture real wrappers before test/aichat/serve/conftest.py autouse fixtures
# stub the module attributes; these scenarios exercise the actual HTTP client.
_REAL_AUTH_VERIFY = internal_client.auth_verify
_REAL_SKU_VERIFY = internal_client.sku_verify
_REAL_SALTS = internal_client.rate_limit_salts
_REAL_RATE_LIMIT_CHECK = internal_client.rate_limit_check

FEATURE = "features/internal_client.feature"

scenarios(FEATURE)


@pytest.fixture
def ctx():
    return {
        "requests": [],
        "attempts": 0,
        "status": 200,
        "break_first": False,
        "break_all": False,
    }


_CLIENTS: list[httpx.AsyncClient] = []


@pytest.fixture(autouse=True)
def _close_mock_clients():
    yield
    while _CLIENTS:
        asyncio.run(_CLIENTS.pop().aclose())


def make_client(ctx) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        ctx["attempts"] += 1
        if ctx["break_all"] or (ctx["break_first"] and ctx["attempts"] == 1):
            raise httpx.ReadError("broken pipe", request=request)
        ctx["requests"].append(
            {
                "method": request.method,
                "path": request.url.path,
                "body": json.loads(request.content) if request.content else None,
            }
        )
        return httpx.Response(ctx["status"], json={"ok": True, "current": "c"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _CLIENTS.append(client)
    return client


@given(parsers.parse('aichat-internal base url "{url}"'))
def _(url, monkeypatch):
    monkeypatch.setattr(internal_settings, "internal_base_url", url)


@given("aichat-internal answers 503")
def _(ctx):
    ctx["status"] = 503


@given("aichat-internal breaks the first response")
def _(ctx):
    ctx["break_first"] = True


@given("aichat-internal breaks every response")
def _(ctx):
    ctx["break_all"] = True


@when("auth verify is sent", target_fixture="result")
def _(ctx):
    return asyncio.run(
        _REAL_AUTH_VERIFY(
            make_client(ctx),
            authorization="Bearer t",
            digest="d",
            body_sha256_b64="h",
            x_forwarded_host=None,
            x_brave_key=None,
            metadata={"model": "m"},
        )
    )


@when(
    parsers.parse('sku verify is sent with credential "{credential}"'),
    target_fixture="result",
)
def _(ctx, credential):
    return asyncio.run(
        _REAL_SKU_VERIFY(
            make_client(ctx),
            sku_credential=credential,
            service_key_id="sk-1",
            idempotency_key="idem-1",
        )
    )


@when("rate limit salts are requested", target_fixture="result")
def _(ctx):
    return asyncio.run(_REAL_SALTS(make_client(ctx)))


@when(
    parsers.parse('a rate limit check is sent for model "{model}"'),
    target_fixture="result",
)
def _(ctx, model):
    return asyncio.run(
        _REAL_RATE_LIMIT_CHECK(
            make_client(ctx),
            model=model,
            hashed_ip_current="cur-hash",
            hashed_ip_previous="prev-hash",
            is_premium_host=False,
            is_content_agent_request=False,
            is_free_model=True,
            maximum_requests=10,
            interval_in_seconds=60,
            force_error_rate_limit=False,
        )
    )


@then(parsers.parse('aichat-internal received a {method} to "{path}"'))
def _(ctx, method, path, result):
    assert ctx[
        "requests"
    ], f"no request reached aichat-internal (attempts={ctx['attempts']})"
    recorded = ctx["requests"][-1]
    assert recorded["method"] == method
    assert recorded["path"] == path


@then("the auth verify payload was forwarded")
def _(ctx):
    assert ctx["requests"][-1]["body"] == {
        "authorization": "Bearer t",
        "digest": "d",
        "body_sha256_b64": "h",
        "x_forwarded_host": None,
        "x_brave_key": None,
        "metadata": {"model": "m"},
    }


@then("the sku verify payload carried the credential and idempotency key")
def _(ctx):
    assert ctx["requests"][-1]["body"] == {
        "sku_credential": "sku-1",
        "service_key_id": "sk-1",
        "idempotency_key": "idem-1",
    }


@then("the rate limit payload carried the hashed identity")
def _(ctx):
    body = ctx["requests"][-1]["body"]
    assert body["hashed_ip_current"] == "cur-hash"
    assert body["hashed_ip_previous"] == "prev-hash"
    assert body["maximum_requests"] == 10
    assert body["interval_in_seconds"] == 60


@then("the verdict request returns nothing")
def _(ctx, result):
    # 5xx responses are not retried: a single attempt, no verdict.
    assert ctx["attempts"] == 1
    assert result is None


@then("the salts request succeeded on the second attempt")
def _(ctx, result):
    assert ctx["attempts"] == 2
    assert result == {"ok": True, "current": "c"}


@then("the auth verdict succeeded on the second attempt")
def _(ctx, result):
    assert ctx["attempts"] == 2
    assert result == {"ok": True, "current": "c"}


@then("the rate limit check returns nothing after one attempt")
def _(ctx, result):
    assert ctx["attempts"] == 1
    assert result is None


@then("the salts request returns nothing after two attempts")
def _(ctx, result):
    assert ctx["attempts"] == 2
    assert result is None
