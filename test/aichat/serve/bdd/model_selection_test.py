# TODO: dedupe with test/aichat/serve/services/model_selection_test.py where noted
# TODO: remove model_selection_test.py#test_content_agent_capability_supported_model_passes_through test/aichat/serve/services/model_selection_test.py:78
# TODO: remove model_selection_test.py#test_content_agent_capability_unsupported_model_falls_back test/aichat/serve/services/model_selection_test.py:52
# TODO: remove model_selection_test.py#test_non_automatic_model test/aichat/serve/services/model_selection_test.py:145
# TODO: remove model_selection_test.py#test_premium_user_gets_premium_model test/aichat/serve/services/model_selection_test.py:162
# TODO: remove model_selection_test.py#test_non_premium_user_gets_free_model test/aichat/serve/services/model_selection_test.py:189
# TODO: remove model_selection_test.py#test_tab_focus_routes_to_tab_focus_model test/aichat/serve/services/model_selection_test.py:270
# TODO: remove model_selection_test.py#test_long_context_detection test/aichat/serve/services/model_selection_test.py:291
# TODO: remove model_selection_test.py#test_androcles_classification test/aichat/serve/services/model_selection_test.py:314
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.protocol.open_ai_protocol import (
    Capability,
    TextContentPart,
)
from aichat.serve.services import model_selection
from aichat.serve.services import models as services_models
from aichat.serve.services.androcles_prefetch import AndroclesPrefetch
from aichat.serve.services.models import get_model_config

FEATURE = Path(__file__).parent / "features" / "model_selection.feature"
scenarios(str(FEATURE))


@pytest.fixture
def ctx():
    return {}


@pytest.fixture(autouse=True)
def _clear_model_config_cache():
    get_model_config.cache_clear()
    services_models._get_upstream_to_model_id_mapping.cache_clear()
    services_models._get_model_id_to_bedrock_profile.cache_clear()
    yield
    get_model_config.cache_clear()
    services_models._get_upstream_to_model_id_mapping.cache_clear()
    services_models._get_model_id_to_bedrock_profile.cache_clear()


# --- model catalog givens ----------------------------------------------------


@given(parsers.parse('model "{model_id}" with content agent support'))
def given_cap_model(model_id, monkeypatch, ctx):
    ctx["models"] = {model_id: {"type": "llm", "capabilities": ["content_agent"]}}
    ctx["model_id"] = model_id
    _install_settings(monkeypatch, ctx)


@given(parsers.parse('model "{model_id}" without content agent support'))
def given_non_cap_model(model_id, monkeypatch, ctx):
    ctx["models"] = {model_id: {"type": "llm", "capabilities": []}}
    ctx["model_id"] = model_id
    _install_settings(monkeypatch, ctx)


@given(parsers.parse('a content agent default model "{default_model}"'))
def given_agent_default(default_model, monkeypatch, ctx):
    ctx["content_agent_default_model"] = default_model
    _install_settings(monkeypatch, ctx)


@given(parsers.parse('model "{model_id}" exists'))
def given_model_exists(model_id, monkeypatch, ctx):
    ctx.setdefault("models", {})[model_id] = {"type": "llm", "capabilities": []}
    _install_settings(monkeypatch, ctx)


@given(parsers.parse('model "{model_id}" with weighted members "{members}"'))
def given_weighted_model(model_id, members, monkeypatch, ctx):
    members_list = []
    for entry in members.split(","):
        name, weight = entry.strip().rsplit(":", 1)
        members_list.append({"model": name, "weight": float(weight)})
    ctx.setdefault("models", {})[model_id] = {
        "type": "llm",
        "capabilities": [],
        "models": members_list,
    }
    _install_settings(monkeypatch, ctx)


@given("no models configured beyond triaging defaults")
def given_no_models(monkeypatch, ctx):
    ctx["models"] = {}
    _install_settings(monkeypatch, ctx)


@given("no triaging configuration")
def given_no_triaging(monkeypatch, ctx):
    ctx["triaging"] = {}
    _install_settings(monkeypatch, ctx)


@given("no models and no triaging configuration")
def given_nothing_configured(monkeypatch, ctx):
    ctx["models"] = {}
    ctx["triaging"] = {}
    _install_settings(monkeypatch, ctx)


@given(
    parsers.parse(
        'triaging default premium "{premium}" and non-premium "{non_premium}"'
    )
)
def given_triage_default(premium, non_premium, monkeypatch, ctx):
    ctx.setdefault("triaging", {})["default"] = {
        "premium": premium,
        "non-premium": non_premium,
    }
    _install_settings(monkeypatch, ctx)


@given(
    parsers.re(
        r'triaging (?!default )(?P<category>\S+) premium "(?P<premium>[^"]+)"'
        r' and non-premium "(?P<non_premium>[^"]+)"'
    )
)
def given_triage_category(category, premium, non_premium, monkeypatch, ctx):
    ctx.setdefault("triaging", {})[category] = {
        "premium": premium,
        "non-premium": non_premium,
    }
    _install_settings(monkeypatch, ctx)


def _install_settings(monkeypatch, ctx):
    mock = MagicMock()
    mock.models = ctx.get("models", {})
    mock.model_triaging = ctx.get("triaging", {})
    mock.content_agent_default_model = ctx.get(
        "content_agent_default_model", "agent-model"
    )
    monkeypatch.setattr(model_selection, "model_settings", mock)
    from aichat.serve.services import models as services_models

    monkeypatch.setattr(services_models, "model_settings", mock)


@given(parsers.parse("the automatic daily count check {verdict} the request"))
def given_daily_count(verdict, monkeypatch, ctx):
    counter = AsyncMock(return_value=verdict == "allows")
    monkeypatch.setattr(
        model_selection, "check_and_increment_automatic_mode_daily_count", counter
    )
    ctx["daily_counter"] = counter


@given(parsers.parse("the random source favors the {position} option"))
def given_random_favors(position, monkeypatch, ctx):
    # First/second select an explicit member; the heavier-member scenario
    # asserts the weights separately via then_picker_weights.
    pick = {"first": 0, "second": 1}[position]

    def _choices(population, weights=None, k=1):
        return [population[min(pick, len(population) - 1)]]

    # Patch random.choices only. model_selection.random IS the stdlib module
    # (Python module objects are shared), so the swap is process-wide and is
    # reverted at test teardown (not per step) — only the choices attribute
    # is replaced, so other random.* uses still behave normally.
    picker = MagicMock(side_effect=_choices)
    monkeypatch.setattr(model_selection.random, "choices", picker)
    ctx["picker"] = picker


@given("brave summary generation is enabled")
def given_summary_enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_BRAVE_SUMMARY", "true")


@given("brave summary generation is disabled")
def given_summary_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_BRAVE_SUMMARY", "")


@given(parsers.parse('a user message carrying a "{part_type}" part'))
def given_summary_message(part_type, ctx):
    if part_type == "brave-request-summary":
        part = SimpleNamespace(type=part_type, text="Summarize this page")
    elif part_type == "brave-filter-tabs":
        part = SimpleNamespace(type=part_type, text="Tabs")
    else:
        part = TextContentPart(type="text", text="plain")
    ctx["messages"] = [
        SimpleNamespace(
            role="user",
            content=[part, TextContentPart(type="text", text="hi")],
        )
    ]


@given("a long conversation over the token threshold")
def given_long_conversation(ctx, monkeypatch):
    from aichat.serve.services import model_selection as ms

    long_text = "x" * 30000
    ctx["messages"] = [SimpleNamespace(role="user", content=long_text)]
    monkeypatch.setattr(
        ms.conversation_settings, "conversation_token_limit_for_long_context", 100
    )


@given(parsers.parse('a prefetch with task type "{task_type}"'))
def given_prefetch_task(task_type, ctx):
    ctx["prefetch"] = AndroclesPrefetch(task_type=task_type)


@given("a prefetch without a task type")
def given_prefetch_empty(ctx):
    ctx["prefetch"] = AndroclesPrefetch(task_type=None)


@given(parsers.parse('androcles classifies the request as "{task}"'))
def given_androcles_classifies(task, monkeypatch, ctx):
    # Dominant probability at the production index imported from the module,
    # so the stub tracks the real label layout instead of a copied one.
    from aichat.serve.androcles import ANDROCLES_TRIAGE_INDICES

    probabilities = [0.1] * (max(ANDROCLES_TRIAGE_INDICES.values()) + 1)
    probabilities[ANDROCLES_TRIAGE_INDICES[task]] = 0.95
    stub = AsyncMock(return_value=probabilities)
    monkeypatch.setattr(model_selection, "androcles_inference", stub)
    ctx["androcles_stub"] = stub


@given("androcles classification returns nothing")
def given_androcles_none(monkeypatch, ctx):
    stub = AsyncMock(return_value=None)
    monkeypatch.setattr(model_selection, "androcles_inference", stub)
    ctx["androcles_stub"] = stub


@then("androcles classified the last user message")
def then_androcles_classified(ctx):
    stub = ctx["androcles_stub"]
    stub.assert_awaited_once()
    assert stub.await_args.args[0] == ctx["last_user_message"]


@given(parsers.parse('a last user message "{content}"'))
def given_last_user_message(content, ctx):
    ctx["last_user_message"] = content


# --- when steps -----------------------------------------------------------------


@when("a model is selected with content agent capability")
def when_select_content_agent(ctx):
    ctx["selected"] = asyncio.run(
        model_selection.select_model_for_request(
            model=ctx["model_id"],
            messages=[],
            brave_capability=[Capability.content_agent],
        )
    )


@when(parsers.parse('a model is selected for "{model}" with content agent capability'))
def when_select_auto_content_agent(model, ctx):
    ctx["selected"] = asyncio.run(
        model_selection.select_model_for_request(
            model=model, messages=[], brave_capability=[Capability.content_agent]
        )
    )


@when(parsers.re(r'a model is selected for "(?P<model>[^"]+)"$'))
def when_select_plain(model, ctx):
    # Anchored: the bare variant must NOT match the suffixed variants below
    # (a plain {model} parse would greedily span their quotes).
    ctx["selected"] = asyncio.run(
        model_selection.select_model_for_request(
            model=model,
            messages=ctx.get("messages", []),
            last_user_message_content=ctx.get("last_user_message"),
            media_type=ctx.get("media_type"),
            androcles_prefetch=ctx.get("prefetch"),
        )
    )


@when(parsers.parse('a model is selected for "{model}" with premium access'))
def when_select_premium(model, ctx):
    ctx["selected"] = asyncio.run(
        model_selection.select_model_for_request(
            model=model,
            messages=ctx.get("messages", []),
            is_premium=True,
            last_user_message_content=ctx.get("last_user_message"),
            media_type=ctx.get("media_type"),
            androcles_prefetch=ctx.get("prefetch"),
        )
    )


@when(parsers.parse('a model is selected for "{model}" without premium access'))
def when_select_free(model, ctx):
    ctx["selected"] = asyncio.run(
        model_selection.select_model_for_request(
            model=model, messages=ctx.get("messages", [])
        )
    )


@when(
    parsers.parse(
        'a model is selected for "{model}" without premium access and rate key "{rate_key}"'
    )
)
def when_select_free_rate_key(model, rate_key, ctx):
    ctx["selected"] = asyncio.run(
        model_selection.select_model_for_request(
            model=model,
            messages=ctx.get("messages", []),
            rate_key=rate_key,
            httpx_client=None,
        )
    )


@when(parsers.parse('a model is selected for "{model}" with media type "{media_type}"'))
def when_select_media(model, media_type, ctx):
    ctx["selected"] = asyncio.run(
        model_selection.select_model_for_request(
            model=model, messages=[], media_type=media_type
        )
    )


@when(parsers.parse('the triage models for "{category}" are resolved'))
def when_triage_models(category, ctx):
    ctx["triage_error"] = None
    try:
        ctx["triage_mapping"] = model_selection._get_triage_models(category)
    except RuntimeError as exc:
        ctx["triage_error"] = exc


@given(parsers.parse("a message history with {description}"))
def given_message_history(description, ctx):
    if description == "a tab focus part":
        ctx["messages"] = [
            SimpleNamespace(
                role="user",
                content=[
                    SimpleNamespace(type="brave-filter-tabs", text="Tabs"),
                    TextContentPart(type="text", text="hi"),
                ],
            )
        ]
    elif description == "a summary part":
        ctx["messages"] = [
            SimpleNamespace(
                role="user",
                content=[
                    SimpleNamespace(type="brave-request-summary", text="Sum it"),
                    TextContentPart(type="text", text="hi"),
                ],
            )
        ]
    elif description == "plain string content":
        ctx["messages"] = [SimpleNamespace(role="user", content="just text")]
    elif description == "an assistant message after user":
        ctx["messages"] = [
            SimpleNamespace(
                role="user",
                content=[SimpleNamespace(type="brave-filter-tabs", text="Tabs")],
            ),
            SimpleNamespace(role="assistant", content="answer"),
        ]
    elif description == "a user message with unknown parts":
        ctx["messages"] = [
            SimpleNamespace(
                role="user",
                content=[TextContentPart(type="text", text="hi")],
            )
        ]
    elif description == "missing content parts":
        ctx["messages"] = [SimpleNamespace(role="user", content=None)]
    else:  # no messages
        ctx["messages"] = []


@when(parsers.parse('the last user message is checked for type "{part_type}"'))
def when_check_content_type(part_type, ctx):
    ctx["detected"] = model_selection._last_user_message_has_content_type(
        ctx.get("messages", []), {part_type}
    )


# --- then steps -----------------------------------------------------------------


@then(parsers.parse('the selected model is "{expected}"'))
def then_selected(expected, ctx):
    assert ctx["selected"] == expected


@then(parsers.parse("the picker weighed the options as {weights}"))
def then_picker_weights(ctx, weights):
    # random.choices(model_ids, model_weights, k=1) — the weights must be
    # the ensemble's configured weights, not a discarded default.
    ctx["picker"].assert_called_once()
    call = ctx["picker"].call_args
    weights_arg = call.kwargs.get(
        "weights", call.args[1] if len(call.args) > 1 else None
    )
    assert list(weights_arg) == [float(w) for w in weights.split(",")]


@then("the daily count was incremented")
def then_daily_incremented(ctx):
    ctx["daily_counter"].assert_awaited_once()


@then(
    parsers.parse(
        'the resolved mapping is premium "{premium}" and non-premium "{non_premium}"'
    )
)
def then_triage_mapping(premium, non_premium, ctx):
    mapping = ctx["triage_mapping"]
    assert mapping["premium"] == premium
    assert mapping["non-premium"] == non_premium


@then("a triage configuration error is raised")
def then_triage_error(ctx):
    assert isinstance(ctx.get("triage_error"), RuntimeError)


@then(parsers.parse("the detection result is {result}"))
def then_detection(result, ctx):
    assert ctx["detected"] == (result == "True")
