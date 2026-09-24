# TODO: remove androcles_test.py#test_task_from_single_label_coding_index test/aichat/serve/androcles_test.py:17
# TODO: remove androcles_test.py#test_task_from_single_label_language_index test/aichat/serve/androcles_test.py:23
# TODO: remove androcles_test.py#test_task_from_single_label_vision_index test/aichat/serve/androcles_test.py:29
# TODO: remove androcles_test.py#test_androcles_inference_returns_none_on_timeout test/aichat/serve/androcles_test.py:40
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve.androcles import (
    androcles_inference,
    task_type_from_androcles_probabilities,
)

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
    ctx["input"] = [MagicMock(text=text), MagicMock(text=text)]


@given("an empty androcles input")
def given_empty_input(ctx):
    ctx["input"] = "   "


@given("an androcles input of 42")
def given_nonstring_input(ctx):
    ctx["input"] = 42


@given(parsers.parse('a router that returns output data "{text}"'))
def given_router_output(text, monkeypatch, ctx):
    values = [float(v) for v in text.split(",")]
    router = _router_returning(values)
    monkeypatch.setattr("aichat.serve.androcles.get_global_router", lambda: router)
    ctx["router"] = router


@given(parsers.parse("probabilities {probabilities}"))
def given_probabilities(probabilities, ctx):
    ctx["probabilities"] = [float(v) for v in probabilities.split(",")]


@given("no probabilities at all")
def given_no_probabilities(ctx):
    ctx["probabilities"] = None


@given("a router that times out")
def given_router_timeout(monkeypatch):
    async def slow_call(**kwargs):
        await asyncio.sleep(10)

    router = MagicMock()
    router.allm_passthrough_route = AsyncMock(side_effect=slow_call)
    monkeypatch.setattr("aichat.serve.androcles.get_global_router", lambda: router)


@given("a router that raises an error")
def given_router_error(monkeypatch):
    router = MagicMock()
    router.allm_passthrough_route = MagicMock(side_effect=RuntimeError("triton down"))
    monkeypatch.setattr("aichat.serve.androcles.get_global_router", lambda: router)


@when(
    parsers.parse(
        "the androcles inference runs with a router returning {count:d} probabilities"
    )
)
def when_inference_router(count, monkeypatch, ctx):
    values = [0.1] * count
    router = _router_returning(values)
    monkeypatch.setattr("aichat.serve.androcles.get_global_router", lambda: router)
    ctx["router"] = router
    ctx["probabilities"] = asyncio.run(androcles_inference(ctx.get("input")))


@when("the androcles inference runs")
def when_inference(ctx):
    ctx["probabilities"] = asyncio.run(androcles_inference(ctx.get("input")))


@when("the androcles inference runs without a timeout")
def when_inference_no_timeout(monkeypatch, ctx):
    ctx.setdefault("input", "classify me")
    ctx["probabilities"] = asyncio.run(
        androcles_inference(ctx["input"], timeout_seconds=0)
    )


@when("the androcles inference runs with a timeout")
def when_inference_timeout(monkeypatch, ctx):
    ctx.setdefault("input", "classify me")
    ctx["probabilities"] = asyncio.run(
        androcles_inference(ctx["input"], timeout_seconds=5)
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


@then(parsers.parse('the probabilities are "{expected}"'))
def then_probabilities(expected, ctx):
    values = [float(v) for v in expected.split(",")]
    assert ctx["probabilities"] == values


@then(parsers.parse("the derived task is {task}"))
def then_task(task, ctx):
    assert ctx["task"] == (None if task == "none" else task)
