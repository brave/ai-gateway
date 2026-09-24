import asyncio
import base64
import json
from hashlib import sha256
from unittest import mock

import blake3
import pytest
from fastapi import HTTPException
from pytest_bdd import given, parsers, scenarios, then, when
from starlette.requests import Request

from aichat.serve import internal_client
from aichat.serve.auth import (
    _request_auth_verdict,
    check_brave_services_key_v2,
    check_premium_host,
    check_sku_credential,
    check_x_brave_key,
    create_idempotency_key,
)
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.internal_settings import internal_settings
from aichat.serve.server_settings import server_settings

FEATURE = "features/auth.feature"

CHAT_PAYLOAD = json.dumps(
    {"model": "llama-2-13b-chat", "messages": [{"role": "user", "content": "hi"}]}
).encode("utf-8")


# TODO: remove test/aichat/serve/auth_test.py#test_create_idempotency_key_known_payload test/aichat/serve/auth_test.py:15
# TODO: remove test/aichat/serve/auth_test.py#test_create_idempotency_key_is_deterministic test/aichat/serve/auth_test.py:25
# TODO: remove test/aichat/serve/auth_test.py#test_create_idempotency_key_changes_with_inputs test/aichat/serve/auth_test.py:32
# TODO: remove test/aichat/serve/auth_test.py#test_create_idempotency_key_none_inputs_random test/aichat/serve/auth_test.py:40
scenarios(FEATURE)


@pytest.fixture
def ctx():
    return {"verdict": None, "auth_calls": [], "sku_calls": [], "request": None}


def make_request(body: bytes = b"") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
        "query_string": b"",
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)
    request.state.httpx_client = object()
    return request


@given("the gateway app")
def _():
    pass


@given("the internal API is enabled")
def _(monkeypatch):
    # test/aichat/serve/conftest.py already enables it; keep explicit for
    # scenario readability.
    monkeypatch.setattr(internal_settings, "internal_api_enabled", True)


@given("the internal API is disabled")
def _(monkeypatch):
    monkeypatch.setattr(internal_settings, "internal_api_enabled", False)


@given("aichat-internal answers auth requests with service_key_allowed true")
def _(monkeypatch, ctx):
    async def _verify(*args, **kwargs):
        ctx["auth_calls"].append(kwargs)
        return {
            "service_key_allowed": True,
            "service_key_id": "sk-1",
            "brave_key_allowed": True,
        }

    monkeypatch.setattr(internal_client, "auth_verify", _verify)


@when("an auth verdict is requested for a chat payload", target_fixture="verdict")
def _(ctx):
    ctx["request"] = make_request(CHAT_PAYLOAD)
    ctx["verdict"] = asyncio.run(
        _request_auth_verdict(ctx["request"], authorization="Bearer t")
    )
    return ctx["verdict"]


@then("no verdict is produced")
def _(verdict):
    assert verdict is None


@then("the verdict is returned")
def _(verdict):
    assert verdict is not None and verdict["service_key_allowed"] is True


@then("the auth request carried the body digest")
def _(ctx):
    expected = base64.b64encode(sha256(CHAT_PAYLOAD).digest()).decode("utf-8")
    assert ctx["auth_calls"][0]["body_sha256_b64"] == expected


@then(parsers.parse('the auth request metadata excluded "{field}"'))
def _(ctx, field):
    assert field not in ctx["auth_calls"][0]["metadata"]


@then(parsers.parse('the auth request metadata carries the chat model "{model}"'))
def _(ctx, model):
    assert ctx["auth_calls"][0]["metadata"]["model"] == model


@given(parsers.parse('messages for {user} and model "{model}"'))
def _(ctx, user, model):
    ctx["messages"] = [{"role": "user", "content": f"hello {user}"}]
    ctx["model"] = model


@when("the idempotency key is created twice", target_fixture="keys")
def _(ctx):
    k1 = create_idempotency_key(ctx["messages"], ctx["model"])
    k2 = create_idempotency_key(ctx["messages"], ctx["model"])
    return {"first": k1, "second": k2}


@when('the idempotency key is created for model "mixtral-8x7b-instruct"')
def _(ctx):
    ctx["other_model_key"] = create_idempotency_key(
        ctx["messages"], "mixtral-8x7b-instruct"
    )


@when("the idempotency key is created for different messages")
def _(ctx):
    ctx["other_messages_key"] = create_idempotency_key(
        [{"role": "user", "content": "different content"}], ctx["model"]
    )


@then("both idempotency keys match")
def _(keys):
    assert keys["first"] == keys["second"]


@then("the keys for different payloads differ")
def _(keys, ctx):
    assert keys["first"] != ctx["other_model_key"]
    assert keys["first"] != ctx["other_messages_key"]


@when("the idempotency key is created without messages or model", target_fixture="key")
def _(ctx):
    first = create_idempotency_key(None, None)
    ctx["random_key"] = create_idempotency_key(None, None)
    return first


@then("two random idempotency keys differ")
def _(key, ctx):
    assert key != ctx["random_key"]


@then(parsers.parse('the idempotency key starts with "{prefix}"'))
def _(key, prefix):
    assert key.startswith(prefix)


@then("the key matches the canonical payload digest")
def _(keys, ctx):
    # create_idempotency_key digests the canonical JSON of messages+model;
    # pinning the exact digest guards the wire contract with aichat-internal.
    canonical = json.dumps(
        {"messages": ctx["messages"], "model": ctx["model"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert keys["first"] == blake3.blake3(canonical.encode("utf-8")).hexdigest()


@given(parsers.parse('the premium host header "{host}"'))
def _(ctx, host):
    ctx["host"] = host


@given(parsers.parse('the forwarded host header "{host}"'))
def _(ctx, host):
    ctx["forwarded_host"] = host


@given(parsers.parse('the configured premium host is "{host}"'))
def _(host, monkeypatch):
    monkeypatch.setattr(external_service_settings, "ai_chat_premium_host", host)


@when("the premium host is checked", target_fixture="is_premium")
def _(ctx):
    return check_premium_host(
        host=ctx.get("host"), x_forwarded_host=ctx.get("forwarded_host")
    )


@then("the request is recognized as premium")
def _(is_premium):
    assert is_premium is True


@then("the request is not recognized as premium")
def _(is_premium):
    assert is_premium is False


@given(parsers.parse("the environment is {env}"))
def _(env, monkeypatch):
    monkeypatch.setattr(server_settings, "env", env)


@when(
    parsers.parse('the premium credential is checked with cookie "{cookie}"'),
    target_fixture="sku_allowed",
)
def _(ctx, cookie):
    try:
        return asyncio.run(
            check_sku_credential(
                make_request(CHAT_PAYLOAD), sku_credential=cookie, auth_verdict={}
            )
        )
    except HTTPException as exc:
        ctx["error"] = exc
        return None


@when(
    "the premium credential is checked without a cookie", target_fixture="sku_allowed"
)
def _(ctx):
    try:
        return asyncio.run(
            check_sku_credential(
                make_request(CHAT_PAYLOAD), sku_credential=None, auth_verdict={}
            )
        )
    except HTTPException as exc:
        ctx["error"] = exc
        return None


@then("the premium credential is accepted")
def _(sku_allowed):
    assert sku_allowed is True


@then("the premium credential is denied")
def _(sku_allowed):
    assert sku_allowed is False


@then(parsers.parse("the request is rejected with {code:d}"))
def _(ctx, code):
    error = ctx.get("error")
    assert isinstance(error, HTTPException) and error.status_code == code
    if code == 503 and "sku_stub" in ctx:
        # Prove the 503 comes from the unavailable sku backend, not an
        # unrelated branch: the stub must have been called.
        ctx["sku_stub"].assert_called_once()


@given("aichat-internal cannot answer sku requests")
def _(monkeypatch, ctx):
    async def _down(*args, **kwargs):
        return None

    stub = mock.Mock(side_effect=_down)
    ctx["sku_stub"] = stub
    monkeypatch.setattr(internal_client, "sku_verify", stub)


@when("the brave key is checked", target_fixture="brave_ok")
def _(ctx):
    try:
        return asyncio.run(check_x_brave_key(ctx.get("verdict")))
    except HTTPException as exc:
        ctx["error"] = exc
        return None


@then("the brave key is accepted")
def _(brave_ok):
    assert brave_ok is True


@then("the brave key is rejected")
def _(brave_ok):
    assert brave_ok is False


@given("no auth verdict exists")
def _(ctx):
    ctx["verdict"] = None


@given("the auth verdict says brave_key_allowed false")
def _(ctx):
    ctx["verdict"] = {"brave_key_allowed": False}


@when("the service key is checked", target_fixture="service_ok")
def _(ctx):
    ctx["request"] = make_request(CHAT_PAYLOAD)
    try:
        return asyncio.run(
            check_brave_services_key_v2(ctx["request"], verdict=ctx.get("verdict"))
        )
    except HTTPException as exc:
        ctx["error"] = exc
        return None


@then("the service key is accepted")
def _(service_ok):
    assert service_ok is True


@given("the auth verdict carries the service key state")
def _(ctx):
    ctx["verdict"] = {
        "service_key_allowed": True,
        "service_key_id": "sk-1",
        "model_override": True,
        "fallback_model": "fallback-model",
        "premium_fallback_model": "premium-fallback",
        "request_allowed": False,
    }


@then(parsers.parse('the request state carries "{field}" "{value}"'))
def _(ctx, field, value):
    assert getattr(ctx["request"].state, field) == value


@then(
    parsers.parse(
        'the request state carries model override {override} with fallback "{fallback}"'
    )
)
def _(ctx, override, fallback):
    assert ctx["request"].state.model_override is (override == "true")
    assert ctx["request"].state.model_override_fallback == fallback


@then(parsers.parse('the request state carries premium fallback "{fallback}"'))
def _(ctx, fallback):
    assert ctx["request"].state.model_override_premium_fallback == fallback


@then(parsers.parse("the request state carries request_allowed {allowed}"))
def _(ctx, allowed):
    assert ctx["request"].state.request_allowed is (allowed == "true")


@given("an auth verdict without a request_allowed field")
def _(ctx):
    ctx["verdict"] = {"service_key_allowed": True, "service_key_id": "sk-9"}
