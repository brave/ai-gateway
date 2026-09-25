# TODO: dedupe with test/aichat/serve/models_api_test.py (accept-language rows, endpoint shape rows)
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import models_api
from aichat.serve.api_server import app
from aichat.serve.services import models as services_models
from aichat.serve.services.model_settings import ModelSettings

FEATURE = Path(__file__).parent / "features" / "models_catalog.feature"
scenarios(str(FEATURE))


@pytest.fixture
def ctx():
    return {}


@pytest.fixture(autouse=True)
def _clear_caches():
    services_models.get_model_config.cache_clear()
    services_models._get_upstream_to_model_id_mapping.cache_clear()
    services_models._get_model_id_to_bedrock_profile.cache_clear()
    models_api.get_browser_models.cache_clear()
    yield
    services_models.get_model_config.cache_clear()
    services_models._get_upstream_to_model_id_mapping.cache_clear()
    services_models._get_model_id_to_bedrock_profile.cache_clear()
    models_api.get_browser_models.cache_clear()


def _install(monkeypatch, ctx):
    mock = MagicMock()
    mock.models = ctx.get("models", {})
    mock.model_triaging = ctx.get("triaging", {})
    monkeypatch.setattr(services_models, "model_settings", mock)
    return mock


# --- model config resolution ---------------------------------------------------


@given(
    parsers.parse(
        'a model "cap-model" with capabilities "{caps}", type "{model_type}" and free {free}'
    )
)
def given_cap_model(caps, model_type, free, monkeypatch, ctx):
    ctx["model_id"] = "cap-model"
    ctx["models"] = {
        "cap-model": {
            "type": model_type,
            "capabilities": [c.strip() for c in caps.split(",")],
            "free": free == "True",
            "fallback_models": [],
            "backend": "vllm",
        }
    }
    _install(monkeypatch, ctx)


@given(parsers.parse('a model "{model_id}" with type "{model_type}"'))
def given_model_type(model_id, model_type, monkeypatch, ctx):
    ctx["model_id"] = model_id
    ctx["models"] = {model_id: {"type": model_type, "capabilities": []}}
    _install(monkeypatch, ctx)


@given('a model "simple" with no free flag')
def given_no_free_flag(monkeypatch, ctx):
    ctx["model_id"] = "simple"
    ctx["models"] = {"simple": {"type": "llm", "capabilities": []}}
    _install(monkeypatch, ctx)


@given(parsers.parse('a model "{model_id}" falling back to "{fallback}"'))
def given_fallback_model(model_id, fallback, monkeypatch, ctx):
    ctx["model_id"] = model_id
    ctx["models"] = {
        model_id: {
            "type": "llm",
            "capabilities": [],
            "fallback_models": [fallback],
            "backend": "vllm",
        }
    }
    _install(monkeypatch, ctx)


@given(parsers.re(r'a model "(?P<model_id>[^"]+)" with backend "(?P<backend>[^"]+)"$'))
def given_backend(model_id, backend, monkeypatch, ctx):
    ctx["model_id"] = model_id
    ctx["models"] = {model_id: {"type": "llm", "capabilities": [], "backend": backend}}
    _install(monkeypatch, ctx)


@given(
    parsers.re(
        r'fallback model "(?P<fallback>[^"]+)" with backend "(?P<backend>[^"]+)"$'
    )
)
def given_fallback_backend(fallback, backend, monkeypatch, ctx):
    ctx["models"][fallback] = {"type": "llm", "capabilities": [], "backend": backend}
    ctx["models"][ctx["model_id"]]["fallback_models"] = [fallback]
    _install(monkeypatch, ctx)


@given(
    parsers.re(
        r'a model "(?P<model_id>[^"]+)" with upstream model "(?P<upstream>[^"]+)"$'
    )
)
def given_upstream(model_id, upstream, monkeypatch, ctx):
    ctx["model_id"] = model_id
    ctx["models"] = {
        model_id: {
            "type": "llm",
            "capabilities": [],
            "backend": "vllm",
            "upstream_model": upstream,
        }
    }
    _install(monkeypatch, ctx)


@given(
    parsers.parse(
        'a model "{model_id}" with backend "{backend}" and inference profile env "{env_name}"'
    )
)
def given_bedrock_profile(model_id, backend, env_name, monkeypatch, ctx):
    ctx["model_id"] = model_id
    ctx["models"] = {
        model_id: {
            "type": "llm",
            "capabilities": [],
            "backend": backend,
            "inference_profile": env_name,
        }
    }
    _install(monkeypatch, ctx)


@given(
    parsers.re(
        r'a model "(?P<model_id>[^"]+)" with upstream model "(?P<upstream>[^"]+)"'
        r' on backend "(?P<backend>[^"]+)"$'
    )
)
def given_mantle_model(model_id, upstream, backend, monkeypatch, ctx):
    ctx["model_id"] = model_id
    ctx["models"] = {
        model_id: {
            "type": "llm",
            "capabilities": [],
            "backend": backend,
            "upstream_model": upstream,
        }
    }
    _install(monkeypatch, ctx)


@given(parsers.parse('inference profile env "{env_name}" set to "{value}"'))
def given_profile_env(env_name, value, monkeypatch):
    monkeypatch.setenv(env_name, value)


@given(parsers.parse('model triaging with only a "{category}" category'))
def given_triaging_category(category, monkeypatch, ctx):
    ctx["triaging"] = {category: {"premium": "p1", "non-premium": "f1"}}
    ctx["models"] = {}
    _install(monkeypatch, ctx)


@given(parsers.parse('model triaging with a "{category}" category'))
def given_triaging_default(category, monkeypatch, ctx):
    ctx["triaging"] = {category: {"premium": "p1", "non-premium": "f1"}}
    ctx["models"] = {}
    _install(monkeypatch, ctx)


@when("the model config is resolved")
def when_resolve_config(ctx):
    ctx["config"] = services_models.get_model_config(ctx["model_id"])


@when("the model config is resolved twice")
def when_resolve_config_twice(ctx):
    ctx["config"] = services_models.get_model_config(ctx["model_id"])
    ctx["config_again"] = services_models.get_model_config(ctx["model_id"])


@when("the bedrock fallback is checked")
def when_bedrock_fallback(ctx):
    config = ctx.get("config")
    if config is None:
        config = services_models.get_model_config(ctx["model_id"])
        ctx["config"] = config
    ctx["verdict"] = config.has_bedrock_fallback()


@when(parsers.parse('the litellm name "{litellm_name}" is normalized'))
def when_normalize(litellm_name, ctx):
    if litellm_name == "none":
        ctx["normalized"] = services_models.normalize_model_name(None)
    else:
        ctx["normalized"] = services_models.normalize_model_name(litellm_name)


@when(
    parsers.parse(
        'the litellm name "{litellm_name}" is normalized with requested model "{requested}"'
    )
)
def when_normalize_requested(litellm_name, requested, ctx):
    ctx["normalized"] = services_models.normalize_model_name(litellm_name, requested)


@given(parsers.parse('bedrock models "{model_ids}" sharing profile env "{env_name}"'))
def given_shared_profiles(model_ids, env_name, monkeypatch, ctx):
    ctx["models"] = {
        mid.strip(): {
            "type": "llm",
            "capabilities": [],
            "backend": "bedrock",
            "inference_profile": env_name,
        }
        for mid in model_ids.split(",")
    }
    ctx["model_id"] = model_ids.split(",")[0].strip()
    _install(monkeypatch, ctx)


@when("the model settings are constructed")
def when_construct_settings(caplog, monkeypatch, ctx):
    import logging

    monkeypatch.delenv("MODEL_TRIAGING", raising=False)
    monkeypatch.delenv("MODELS", raising=False)
    with caplog.at_level(
        logging.WARNING, logger="aichat.serve.services.model_settings"
    ):
        ctx["warning_count_before"] = len(caplog.records)
        ModelSettings(
            models=ctx.get("models", {}), model_triaging=ctx.get("triaging", {})
        )
        ctx["warning_count_after"] = len(caplog.records)


@then(parsers.parse("the config tool support is {tool_support}"))
def then_tool_support(tool_support, ctx):
    assert ctx["config"].tool_support == (tool_support == "True")


@then(parsers.parse("the config free flag is {free}"))
def then_free_flag(free, ctx):
    assert ctx["config"].free == (free == "True")


@then("there is no model config")
def then_no_config(ctx):
    assert ctx["config"] is None


@then(parsers.parse('the config dict model id is "{model_id}"'))
def then_config_dict(model_id, ctx):
    assert ctx["config"].to_dict()["model_id"] == model_id


@then(parsers.parse('the config lookup for "{key}" is {expected}'))
def then_config_lookup(key, expected, ctx):
    assert ctx["config"].get(key) == (expected == "True")


@then("both resolutions return the same config")
def then_cached_config(ctx):
    assert ctx["config"] is ctx["config_again"]
    assert ctx["config"] is not None


@then(parsers.parse('the config fallback models are "{fallbacks}"'))
def then_fallback_models(fallbacks, ctx):
    assert ctx["config"].fallback_models == [fallbacks]


@then(parsers.parse("the fallback verdict is {verdict}"))
def then_fallback_verdict(verdict, ctx):
    assert ctx["verdict"] == (verdict == "True")


@then(parsers.parse('the normalized model is "{expected}"'))
def then_normalized(expected, ctx):
    expected_value = None if expected == "none" else expected
    assert ctx["normalized"] == expected_value


@then("a misconfiguration warning is logged")
def then_warning_logged(caplog, ctx):
    assert any(
        'MODEL_TRIAGING is configured but missing a "default" entry' in r.getMessage()
        and r.name == "aichat.serve.services.model_settings"
        for r in caplog.records[ctx["warning_count_before"] :]
    )


@then("no misconfiguration warning is logged")
def then_no_warning(caplog, ctx):
    assert not any(
        'MODEL_TRIAGING is configured but missing a "default" entry' in r.getMessage()
        for r in caplog.records[ctx["warning_count_before"] :]
    )


# --- models_api ----------------------------------------------------------------


@given(
    parsers.parse(
        'a browser client requesting models with language "{accept_language}"'
    )
)
def given_client_language(accept_language, monkeypatch, ctx):
    ctx["accept_language"] = None if accept_language == "nothing" else accept_language
    ctx["models"] = {
        "basic": {
            "backend": "vllm",
            "friendly_name": "Basic Model",
            "maker": "Acme",
            "free": True,
            "capabilities": ["files"],
            "category": "chat",
            "description": {"en": "A test model.", "de": "Ein Testmodell."},
        }
    }
    _install_api(monkeypatch, ctx)


@given('a browser client with model "basic" defined')
def given_client_basic(monkeypatch, ctx):
    ctx["models"] = {
        "basic": {
            "backend": "vllm",
            "friendly_name": "Basic Model",
            "maker": "Acme",
            "free": True,
            "capabilities": ["files"],
            "category": "chat",
            "description": {"en": "A test model."},
        }
    }
    _install_api(monkeypatch, ctx)


def _install_api(monkeypatch, ctx):
    mock = MagicMock()
    mock.models = ctx.get("models", {})
    monkeypatch.setattr(models_api, "model_settings", mock)
    external = MagicMock()
    external.near_api_key = None
    monkeypatch.setattr(models_api, "external_service_settings", external)
    mcp = MagicMock()
    mcp.deep_research_enabled = False
    monkeypatch.setattr(models_api, "mcp_settings", mcp)


@when("the models are listed")
def when_models_listed(ctx):
    _list_models(ctx, accept_language=ctx.get("accept_language"))


@when(parsers.parse('the models are listed with language "{language}"'))
def when_models_listed_lang(language, ctx):
    _list_models(ctx, accept_language=language)


def _list_models(ctx, accept_language):

    with TestClient(app) as client:
        headers = {"Accept-Language": accept_language} if accept_language else {}
        ctx["response"] = client.get("/v1/models", headers=headers)


@then(parsers.parse('the "{model_id}" model display description is "{description}"'))
def then_display_description(model_id, description, ctx):
    # Look up by key, not position, so a synthetic "automatic" entry added
    # by prod cannot break this assertion.
    entry = _catalog_entry(ctx, model_id)
    assert entry["options"]["description"] == description


@then(parsers.parse('the catalog contains model "{model_id}"'))
def then_catalog_contains(model_id, ctx):
    keys = [entry["key"] for entry in ctx["response"].json()]
    assert model_id in keys


def _catalog_entry(ctx, model_id):
    entries = [e for e in ctx["response"].json() if e["key"] == model_id]
    assert entries, [e["key"] for e in ctx["response"].json()]
    return entries[0]


@then(parsers.parse('the catalog entry options name is "{model_id}"'))
def then_options_name(model_id, ctx):
    # Look up by model key; positional [0] would break if entries are reordered.
    entry = _catalog_entry(ctx, model_id)
    assert entry["options"]["name"] == model_id


@then(parsers.parse('the "{model_id}" catalog entry access is "{access}"'))
def then_access(model_id, access, ctx):
    # Look up by model key; positional [0] would break if entries are reordered.
    entry = _catalog_entry(ctx, model_id)
    assert entry["options"]["access"] == access
