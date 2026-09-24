import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.protocol.open_ai_protocol import Capability, ErrorCode
from aichat.serve import common_api
from aichat.serve.rate_limiting_settings import rate_limiting_settings
from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.security_settings import security_settings

FEATURE = Path(__file__).parent / "features" / "common_params.feature"
scenarios(str(FEATURE))


@pytest.fixture
def ctx():
    return {}


@pytest.fixture(autouse=True)
def _no_rate_limiting(monkeypatch):
    """Belt: these request-check scenarios never test rate limiting unless a
    Given explicitly enables it, so default it off regardless of env."""
    monkeypatch.setattr(rate_limiting_settings, "rate_limiting_enabled", False)


@given(
    parsers.parse(
        'a model catalog with free "{free_model}" and premium "{premium_model}"'
    )
)
def given_catalog(free_model, premium_model, monkeypatch, ctx):
    catalog = {
        free_model: {"type": "llm", "free": True},
        premium_model: {"type": "llm", "free": False},
    }
    ctx["catalog"] = catalog
    monkeypatch.setattr(model_settings, "models", catalog)


@given(parsers.parse('the request asks for model "{requested}"'))
def given_requested_model(requested, ctx):
    ctx["resolve_request"] = SimpleNamespace(model=requested)


@given(parsers.parse("the host is premium {is_premium}"))
def given_host_premium(is_premium, ctx):
    ctx["is_premium_host"] = is_premium == "True"


# TODO: remove common_api_test.py#test_resolve_model_* test/aichat/serve/common_api_test.py:168-229
@when("the model is resolved")
def when_resolve_model(ctx):
    ctx["resolved"] = common_api.resolve_model(
        ctx["resolve_request"], is_premium_host=ctx["is_premium_host"]
    )


@then(parsers.parse('the resolved model is "{resolved}"'))
def then_resolved(resolved, ctx):
    assert ctx["resolved"] == resolved


# --- apply_model_override ---------------------------------------------------------------


@given(
    parsers.parse(
        "an override state {override} with premium fallback {premium_fallback}"
        " and fallback {fallback}"
    )
)
def given_override_state(override, premium_fallback, fallback, ctx):
    raw = MagicMock(spec=Request)
    raw.state.model_override = override == "on"
    raw.state.model_override_premium_fallback = (
        None if premium_fallback == "none" else premium_fallback
    )
    raw.state.model_override_fallback = "automatic" if fallback == "none" else fallback
    raw.state.service_key_id = "key-1"
    ctx["raw_request"] = raw


@when(parsers.parse('the model override is applied to "{model}"'))
def when_apply_override(model, ctx):
    ctx["override_result"] = common_api.apply_model_override(
        ctx["raw_request"], model, ctx["is_premium_host"]
    )


@then(parsers.parse('the resulting model is "{result}"'))
def then_override_result(result, ctx):
    assert ctx["override_result"] == result


# --- create_completion_common_params -----------------------------------------------------


@given('a request for model "automatic" with events and messages')
def given_ccp_request(ctx):
    ctx["ccp_request"] = SimpleNamespace(
        model="automatic",
        events=["e1"],
        messages=[{"role": "user", "content": "hi"}],
    )


@when("the completion common params are built without an override")
def when_build_common_params(ctx):
    raw = MagicMock(spec=Request)
    raw.url = SimpleNamespace(path="/v1/chat/completions")
    raw.state.model_override = False
    raw.state.service_key_id = "unknown"
    ctx["ccp_raw"] = raw
    base = {
        "is_premium_host": False,
        "has_valid_premium_credential": False,
        "x_forwarded_host": None,
        "x_forwarded_for": None,
        "is_valid_brave_services_key": True,
        "is_valid_brave_services_key_v2": True,
        "request_allowed": True,
        "api_version": 2,
    }
    ctx["common_result"] = asyncio.run(
        common_api.create_completion_common_params(
            raw_request=raw,
            request=ctx["ccp_request"],
            model="automatic",
            common=base,
        )
    )


@then("the raw request state records the model, events and messages")
def then_state_recorded(ctx):
    raw = ctx["ccp_raw"]
    assert raw.state.model == "automatic"
    assert raw.state.events == ["e1"]
    assert raw.state.messages == [{"role": "user", "content": "hi"}]


@then("the common params mark the automatic request")
def then_automatic_marked(ctx):
    assert ctx["common_result"]["is_automatic_model_request"] is True


# --- check_requests_common ----------------------------------------------------------------


@given(parsers.parse("the common check state {state}"))
def given_common_state(state, ctx):
    ctx["common_state"] = state


@given("a premium host without a valid SKU credential")
def given_premium_no_sku(ctx):
    ctx["common"] = {
        "model": "premium-x",
        "is_premium_host": True,
        "has_valid_premium_credential": False,
        "is_valid_brave_services_key": True,
        "is_valid_brave_services_key_v2": True,
        "request_allowed": True,
        "x_forwarded_for": None,
        "is_automatic_model_request": False,
        "api_version": 2,
    }


@given("rate limiting is enabled and the verdict denies the request")
def given_rate_denied(monkeypatch):
    monkeypatch.setattr(rate_limiting_settings, "rate_limiting_enabled", True)
    monkeypatch.setattr(
        common_api,
        "check_rate_limit",
        AsyncMock(return_value=SimpleNamespace(allowed=False, fallback_model=None)),
    )


@given(parsers.parse('rate limiting is enabled and the verdict proposes "{model}"'))
def given_rate_fallback(model, monkeypatch):
    monkeypatch.setattr(rate_limiting_settings, "rate_limiting_enabled", True)
    monkeypatch.setattr(
        common_api,
        "check_rate_limit",
        AsyncMock(return_value=SimpleNamespace(allowed=True, fallback_model=model)),
    )


@given("premium-x supports the content agent")
def given_content_agent_support(monkeypatch):
    monkeypatch.setattr(
        common_api,
        "get_model_config",
        MagicMock(return_value=SimpleNamespace(content_agent_support=True)),
    )


@given("the request advertises the content agent capability")
def given_content_agent_capability(ctx):
    ctx["capability"] = Capability.content_agent


@given(
    parsers.parse("the automatic daily limit peek reports {count:d} of {limit:d} used")
)
def given_daily_peek(count, limit, monkeypatch):
    monkeypatch.setattr(
        common_api,
        "check_automatic_mode_daily_limit",
        AsyncMock(return_value={"exceeded": False, "count": count, "limit": limit}),
    )


@given("rate limiting is enabled")
def given_rate_enabled(monkeypatch):
    monkeypatch.setattr(rate_limiting_settings, "rate_limiting_enabled", True)
    monkeypatch.setattr(
        common_api,
        "check_rate_limit",
        AsyncMock(return_value=SimpleNamespace(allowed=True, fallback_model=None)),
    )


@given("the content agent rate limit is exhausted")
def given_content_agent_exhausted(monkeypatch):
    monkeypatch.setattr(
        common_api,
        "check_content_agent_rate_limits",
        AsyncMock(return_value=False),
    )


# TODO: remove common_api_test.py#test_check_requests_common_* test/aichat/serve/common_api_test.py:15-165
@when(parsers.parse('the common request checks run for model "{model}"'))
def when_common_checks(model, ctx):
    state = ctx.get("common_state")
    common = ctx.get("common") or {
        "model": model,
        "is_premium_host": False,
        "has_valid_premium_credential": False,
        "is_valid_brave_services_key": True,
        "is_valid_brave_services_key_v2": True,
        "request_allowed": True,
        "x_forwarded_for": None,
        "x_forwarded_host": None,
        "is_automatic_model_request": model == "automatic",
        "api_version": 2,
    }
    common["model"] = model
    if state == "invalid_services_key":
        common["is_valid_brave_services_key_v2"] = False
    elif state == "request_not_allowed":
        common["request_allowed"] = False
    elif state == "unknown_model":
        # Relies on the catalog given: "mystery-m" is intentionally absent.
        assert model not in ctx["catalog"]
    elif state == "premium_model_free_user":
        # Free user (no premium credential) requesting a premium-only model.
        assert model == "premium-x"
        assert common["has_valid_premium_credential"] is False
    elif state is not None:
        raise AssertionError(f"unknown common check state in feature: {state!r}")
    ctx["common"] = common

    raw = MagicMock()
    raw.state.capability = ctx.get("capability")
    raw.state.service_key_id = "svc-1"
    raw.state.request_allowed = True
    raw.state.httpx_client = None
    ctx["common_result"] = asyncio.run(common_api.check_requests_common(raw, common))


@then(parsers.parse("a JSON error response with status {status:d} is returned"))
def then_common_error(status, ctx):
    assert ctx["common_result"].status_code == status


@then("the common request checks pass")
def then_checks_pass(ctx):
    assert ctx["common_result"] is None


@then(parsers.parse('the common model becomes "{model}"'))
def then_common_model(model, ctx):
    assert ctx["common"]["model"] == model


# --- peek ----------------------------------------------------------------------------------


@given(
    parsers.parse(
        "a rate limit peek reporting exceeded {exceeded} count {count:d} of limit {limit:d}"
    )
)
def given_peek(exceeded, count, limit, ctx):
    ctx["peek"] = {
        "exceeded": exceeded == "True",
        "count": int(count),
        "limit": int(limit),
    }


@when("the peek is evaluated")
def when_peek(ctx):
    ctx["peek_verdict"] = common_api._peek_under_limit(ctx["peek"])


@then(parsers.parse("the peek verdict is {verdict}"))
def then_peek(verdict, ctx):
    assert ctx["peek_verdict"] == (verdict == "True")


# --- bearer tokens ------------------------------------------------------------------------


@given(parsers.parse('an authorization header "{header}"'))
def given_auth_header(header, ctx):
    ctx["auth_header"] = None if header == "<empty>" else header


@when("the bearer token is extracted")
def when_extract_bearer(ctx):
    ctx["bearer"] = common_api.extract_bearer_token(ctx["auth_header"])


@then(parsers.parse("the extracted token is {token}"))
def then_bearer(token, ctx):
    expected = None if token == "none" else token
    assert ctx["bearer"] == expected


# --- internal models api key ---------------------------------------------------------------


@given(parsers.parse("the internal models API key is {configured}"))
def given_internal_key(configured, monkeypatch):
    monkeypatch.setattr(
        security_settings,
        "internal_models_api_key",
        None if configured == "unset" else "secret123",
    )


@when("the internal models API key is validated")
def when_validate_internal_key(ctx):
    ctx["key_valid"] = common_api.is_valid_internal_models_api_key(ctx["auth_header"])


@then(parsers.parse("the validation result is {result}"))
def then_key_valid(result, ctx):
    assert ctx["key_valid"] == (result == "True")


@when("the internal models API key is required")
def when_key_required(ctx):
    ctx["required"] = common_api.require_internal_models_api_key(ctx.get("auth_header"))


@then("no error response is produced")
def then_no_error(ctx):
    assert ctx["required"] is None


# --- create_error_response -----------------------------------------------------------------


@when(
    'an error response is created for the invalid SKU credential code and message "no SKU"'
)
def when_error_response(ctx):
    ctx["error_response"] = common_api.create_error_response(
        ErrorCode.INVALID_SKU_CREDENTIAL, "no SKU"
    )
    ctx["error_code"] = "40104"
    ctx["error_message"] = "no SKU"


@then("the JSON error status is 401")
def then_error_status(ctx):
    assert ctx["error_response"].status_code == 401


@then('the error payload type is "error"')
def then_error_type(ctx):
    payload = json.loads(ctx["error_response"].body)
    assert payload["type"] == "error"


@then(parsers.parse('the error message is "{message}"'))
def then_error_message(message, ctx):
    payload = json.loads(ctx["error_response"].body)
    assert payload["error"]["message"] == message


@then(parsers.parse('the error type is "{code_str}"'))
def then_error_type_str(code_str, ctx):
    payload = json.loads(ctx["error_response"].body)
    assert payload["error"]["type"] == code_str
