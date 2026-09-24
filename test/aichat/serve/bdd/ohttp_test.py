# BDD coverage for aichat/serve/ohttp.py.
#
# TODO: remove test/aichat/serve/ohttp_test.py#test_returns_config_for_valid_model test/aichat/serve/ohttp_test.py:65
# TODO: remove test/aichat/serve/ohttp_test.py#test_returns_404_for_unknown_model test/aichat/serve/ohttp_test.py:82
# TODO: remove test/aichat/serve/ohttp_test.py#test_returns_404_for_model_without_e2ee test/aichat/serve/ohttp_test.py:91
# TODO: remove test/aichat/serve/ohttp_test.py#test_returns_500_when_near_verify_fails test/aichat/serve/ohttp_test.py:100
# TODO: remove test/aichat/serve/ohttp_test.py#test_returns_401_for_invalid_auth test/aichat/serve/ohttp_test.py:116
# TODO: remove test/aichat/serve/ohttp_test.py#test_proxies_request_and_streams_response test/aichat/serve/ohttp_test.py:142
# TODO: remove test/aichat/serve/ohttp_test.py#TestRelayModelOhttp#test_returns_404_for_unknown_model test/aichat/serve/ohttp_test.py:184
# TODO: remove test/aichat/serve/ohttp_test.py#TestRelayModelOhttp#test_returns_404_for_model_without_e2ee test/aichat/serve/ohttp_test.py:199
# TODO: remove test/aichat/serve/ohttp_test.py#test_returns_500_when_near_verify_fails test/aichat/serve/ohttp_test.py:214
# TODO: remove test/aichat/serve/ohttp_test.py#test_returns_500_when_near_api_key_missing test/aichat/serve/ohttp_test.py:234
# TODO: remove test/aichat/serve/ohttp_test.py#test_returns_502_on_upstream_http_error test/aichat/serve/ohttp_test.py:256

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import ohttp as ohttp_mod

FEATURE = "features/ohttp.feature"
scenarios(FEATURE)

_OHTTP_CONFIG = {
    "key_config": "a2V5X2NvbmZpZw==",
    "signing_key": "c2lnbmluZ19rZXk=",
    "signature": "c2lnbmF0dXJl",
    "endpoint_url": "https://example.near.ai/v1/chat/completions",
}

_E2EE_MODEL = {"address": "https://example.near.ai/v1/", "e2ee_support": True}


@pytest.fixture
def ctx():
    return {}


@given("the ohttp harness")
def ohttp_harness(ctx, monkeypatch):
    monkeypatch.setattr(ohttp_mod.external_service_settings, "near_api_key", "key")
    monkeypatch.setattr(
        ohttp_mod, "check_requests_common", AsyncMock(return_value=None)
    )
    return ctx


@given(parsers.parse('a model "{model_name}" with e2ee support'))
def model_with_e2ee(ctx, model_name, monkeypatch):
    ms = MagicMock()
    ms.models = {model_name: dict(_E2EE_MODEL)}
    monkeypatch.setattr(ohttp_mod, "model_settings", ms)


@given(parsers.parse('a model "{model_name}" without e2ee support'))
def model_without_e2ee(ctx, model_name, monkeypatch):
    ms = MagicMock()
    ms.models = {model_name: {"address": "https://example.ai/v1/"}}
    monkeypatch.setattr(ohttp_mod, "model_settings", ms)


@given("no models are configured")
def no_models(ctx, monkeypatch):
    ms = MagicMock()
    ms.models = {}
    monkeypatch.setattr(ohttp_mod, "model_settings", ms)


@given("near verification returns the ohttp config")
def near_verify_ok(ctx, monkeypatch):
    verify = AsyncMock(return_value=_OHTTP_CONFIG)
    monkeypatch.setattr(ohttp_mod.near, "verify_and_get_ohttp_config", verify)
    ctx["verify"] = verify


@given("near verification fails")
def near_verify_fails(ctx, monkeypatch):
    verify = AsyncMock(return_value=None)
    monkeypatch.setattr(ohttp_mod.near, "verify_and_get_ohttp_config", verify)


@given("the near api key is not configured")
def near_api_key_missing(ctx, monkeypatch):
    monkeypatch.setattr(ohttp_mod.external_service_settings, "near_api_key", None)


def _fake_upstream(
    content=b"ohttp-response-bytes", status=200, ctype="message/ohttp-res"
):
    upstream = MagicMock()
    upstream.status_code = status
    upstream.headers = {"content-type": ctype}

    async def _aiter():
        yield content

    async def _aclose():
        pass

    upstream.aiter_bytes = _aiter
    upstream.aclose = _aclose
    return upstream


@given(parsers.parse('the upstream relay returns "{content}"'))
def upstream_returns(ctx, content, monkeypatch):
    client = MagicMock()
    client.build_request.return_value = MagicMock()
    client.send = AsyncMock(return_value=_fake_upstream(content=content.encode()))
    client.aclose = AsyncMock()
    monkeypatch.setattr(ohttp_mod.httpx, "AsyncClient", lambda timeout=None: client)
    ctx["client"] = client


@given("the upstream relay connection is refused")
def upstream_refused(ctx, monkeypatch):
    import httpx

    client = MagicMock()
    client.build_request.return_value = MagicMock()
    client.send = AsyncMock(side_effect=httpx.ConnectError("refused"))
    client.aclose = AsyncMock()
    monkeypatch.setattr(ohttp_mod.httpx, "AsyncClient", lambda timeout=None: client)


@given("the common params check fails")
def common_params_fail(ctx, monkeypatch):
    check = AsyncMock(return_value={"error": "common-failure"})
    monkeypatch.setattr(ohttp_mod, "check_requests_common", check)
    ctx["common_error"] = {"error": "common-failure"}


def _raw_request(body: bytes, content_type="message/ohttp-req"):
    req = MagicMock()
    req.headers = {"content-type": content_type}

    async def _body():
        return body

    req.body = _body
    return req


@when(parsers.parse('the ohttp config for "{model_name}" is fetched'))
def fetch_ohttp_config(ctx, model_name):
    ctx["response"] = asyncio.run(
        ohttp_mod.get_model_ohttp_config(
            model_name, MagicMock(), is_valid_x_brave_key=True
        )
    )


@when(
    parsers.parse(
        'the ohttp config for "{model_name}" is fetched with an invalid services key'
    )
)
def fetch_ohttp_config_bad_key(ctx, model_name):
    req = MagicMock()
    ctx["response"] = asyncio.run(
        ohttp_mod.get_model_ohttp_config(model_name, req, is_valid_x_brave_key=False)
    )


@when(parsers.parse('the relay for "{model_name}" is posted with "{body}"'))
def post_relay(ctx, model_name, body):
    common = {"model": model_name}
    ctx["response"] = asyncio.run(
        ohttp_mod.relay_model_ohttp(
            model_name, _raw_request(body.encode()), common=common
        )
    )


def _error_type_of(response):
    if isinstance(response, dict):
        return response
    if hasattr(response, "body"):
        return json.loads(response.body)
    return response


@then("the ohttp config response is the near config")
def ohttp_config_assert(ctx):
    resp = ctx["response"]
    assert resp == _OHTTP_CONFIG
    ctx["verify"].assert_awaited_once_with("near-model")


@then("the ohttp config response is a model not found error")
@then("the relay response is a model not found error")
def model_not_found_assert(ctx):
    resp = ctx["response"]
    assert resp.status_code == 404
    assert _error_type_of(resp)["error"]["type"] == "40401"


@then("the ohttp config response is an internal error")
@then("the relay response is an internal error")
def internal_error_assert(ctx):
    resp = ctx["response"]
    assert resp.status_code == 500
    assert _error_type_of(resp)["error"]["type"] == "50001"


@then("the ohttp config response is an invalid auth key error")
def invalid_auth_assert(ctx):
    resp = ctx["response"]
    assert resp.status_code == 401
    assert _error_type_of(resp)["error"]["type"] == "40101"


@then(parsers.parse('the relay response is "{content}" with media type "{media_type}"'))
def relay_stream_assert(ctx, content, media_type):
    resp = ctx["response"]
    assert resp.status_code == 200
    assert resp.media_type == media_type

    async def _drain():
        return b"".join([chunk async for chunk in resp.body_iterator])

    assert asyncio.run(_drain()) == content.encode()


@then(parsers.parse('the upstream request hit "{gateway_url}" with the bearer token'))
def upstream_request_assert(ctx, gateway_url):
    client = ctx["client"]
    client.build_request.assert_called_once_with(
        "POST",
        gateway_url,
        headers={
            "Authorization": "Bearer key",
            "content-type": "message/ohttp-req",
        },
        content=b"ohttp-request-bytes",
    )


@then("the relay response is a bad gateway error")
def bad_gateway_assert(ctx):
    resp = ctx["response"]
    assert resp.status_code == 502
    assert _error_type_of(resp)["error"]["type"] == "50201"


@then("the relay response is the common params error")
def common_params_error_assert(ctx):
    assert ctx["response"] == ctx["common_error"]
