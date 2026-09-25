# TODO: remove androcles_test.py#test_task_from_single_label_coding_index test/aichat/serve/androcles_test.py:17
# TODO: remove androcles_test.py#test_task_from_single_label_language_index test/aichat/serve/androcles_test.py:23
# TODO: remove androcles_test.py#test_task_from_single_label_vision_index test/aichat/serve/androcles_test.py:29
# TODO: remove androcles_test.py#test_androcles_inference_returns_none_on_timeout test/aichat/serve/androcles_test.py:40
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve.androcles import (
    androcles_inference,
    task_type_from_androcles_probabilities,
)
from aichat.serve.services.dynamic_leo.androcles_labels import ANDROCLES_NUM_LABELS

FEATURE = Path(__file__).parent / "features" / "androcles.feature"
scenarios(str(FEATURE))


@pytest.fixture
def ctx():
    return {}


def _router_returning(values):
    response = MagicMock()
    response.json.return_value = {"outputs": [{"data": values}]}
    router = MagicMock()
    router.allm_passthrough_route = AsyncMock(return_value=response)
    return router


@given(parsers.parse('an androcles input list with text parts "{text}"'))
def given_list_input(text, ctx):
    from aichat.protocol.open_ai_protocol import TextContentPart

    ctx["input"] = [
        TextContentPart(type="text", text=text),
        TextContentPart(type="text", text=text),
    ]


def _install_no_contact_router(ctx, monkeypatch):
    """If the early-return regresses, the router must not be contacted."""
    router = MagicMock()
    ctx["router"] = router
    monkeypatch.setattr("aichat.serve.androcles.get_global_router", lambda: router)


@given("an empty androcles input")
def given_empty_input(ctx, monkeypatch):
    ctx["input"] = "   "
    _install_no_contact_router(ctx, monkeypatch)


@given("an androcles input of 42")
def given_nonstring_input(ctx, monkeypatch):
    ctx["input"] = 42
    _install_no_contact_router(ctx, monkeypatch)


@given(parsers.parse('a router that returns output data "{text}"'))
def given_router_output(text, monkeypatch, ctx):
    values = [float(v) for v in text.split(",")]
    router = _router_returning(values)
    monkeypatch.setattr("aichat.serve.androcles.get_global_router", lambda: router)
    ctx["router"] = router


@given(parsers.parse('a probability vector favoring "{task}"'))
def given_favoring_task(task, ctx):
    # Dominant probability at the production index imported from the module,
    # so the scenario tracks the real label layout instead of a copy.
    from aichat.serve.androcles import ANDROCLES_TRIAGE_INDICES

    values = [0.1] * (max(ANDROCLES_TRIAGE_INDICES.values()) + 1)
    values[ANDROCLES_TRIAGE_INDICES[task]] = 0.95
    ctx["probabilities"] = values


@given("a flat probability vector")
def given_flat_vector(ctx):
    # Full label width with equal low probabilities: no index can dominate.
    from aichat.serve.services.dynamic_leo.androcles_labels import ANDROCLES_NUM_LABELS

    ctx["probabilities"] = [0.1] * ANDROCLES_NUM_LABELS


@given("a short probability vector")
def given_short_vector(ctx):
    # Shorter than any index: the derivation must skip out-of-range entries.
    ctx["probabilities"] = [0.1] * 3


@given("no probabilities at all")
def given_no_probabilities(ctx):
    ctx["probabilities"] = None


@given("a router that times out")
def given_router_timeout(monkeypatch, ctx):
    async def slow_call(*args, **kwargs):
        # Cancelled by wait_for well before this completes. Returns a VALID
        # payload so a regression that drops wait_for surfaces as a non-None
        # probability result instead of silently returning None too.
        await asyncio.sleep(1)
        return SimpleNamespace(
            json=lambda: {"outputs": [{"data": [0.1] * ANDROCLES_NUM_LABELS}]}
        )

    router = MagicMock()
    router.allm_passthrough_route = AsyncMock(side_effect=slow_call)
    monkeypatch.setattr("aichat.serve.androcles.get_global_router", lambda: router)
    ctx["called_router"] = router


@given("a router that raises an error")
def given_router_error(monkeypatch, ctx):
    router = MagicMock()
    # AsyncMock: the failure must surface on await, like the real Triton call.
    router.allm_passthrough_route = AsyncMock(side_effect=RuntimeError("triton down"))
    monkeypatch.setattr("aichat.serve.androcles.get_global_router", lambda: router)
    ctx["called_router"] = router


@given(parsers.parse("a router returning {count:d} probabilities"))
def given_inference_router(count, monkeypatch, ctx):
    values = [0.1] * count
    router = _router_returning(values)
    monkeypatch.setattr("aichat.serve.androcles.get_global_router", lambda: router)
    ctx["router"] = router


@when("the androcles inference runs")
def when_inference(ctx):
    ctx["probabilities"] = asyncio.run(androcles_inference(ctx.get("input")))


@when("the androcles inference runs without a timeout")
def when_inference_no_timeout(ctx):
    # timeout_seconds=0 intentionally disables the wait_for wrapper
    # (androcles.py only wraps when timeout_seconds > 0).
    ctx.setdefault("input", "classify me")
    ctx["probabilities"] = asyncio.run(
        androcles_inference(ctx["input"], timeout_seconds=0)
    )


@when("the androcles inference runs with a timeout")
def when_inference_timeout(ctx):
    ctx.setdefault("input", "classify me")
    ctx["probabilities"] = asyncio.run(
        androcles_inference(ctx["input"], timeout_seconds=0.05)
    )


@when("the task type is derived")
def when_task_type(ctx):
    ctx["task"] = task_type_from_androcles_probabilities(ctx.get("probabilities"))


@then(parsers.parse('the request payload carried the joined text "{text}"'))
def then_joined_text(text, ctx):
    call_args = ctx["router"].allm_passthrough_route.await_args
    request_data = call_args.kwargs
    payload = request_data["json"]["inputs"][0]["data"]
    assert payload == [[f"{text} {text}"]]


@then("no probabilities are returned")
def then_no_probabilities(ctx):
    assert ctx["probabilities"] is None
    router = ctx.get("router")
    if router is not None:
        # Early-return path: the router was never contacted.
        router.allm_passthrough_route.assert_not_called()
    called_router = ctx.get("called_router")
    if called_router is not None:
        # Timeout/error paths: the failure came FROM the awaited call.
        called_router.allm_passthrough_route.assert_awaited_once()


@then(parsers.parse('the probabilities are "{expected}"'))
def then_probabilities(expected, ctx):
    values = [float(v) for v in expected.split(",")]
    assert ctx["probabilities"] == values


@then(parsers.parse("the derived task is {task}"))
def then_task(task, ctx):
    assert ctx["task"] == (None if task == "none" else task)
