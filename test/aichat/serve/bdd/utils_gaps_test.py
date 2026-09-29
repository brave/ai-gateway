"""BDD steps for remaining aichat/serve/utils.py branches (non-llm mapping skip,
normalize last-segment fallback, get_malloc_trim success, periodic loop, dict estimate).

New coverage — no unit-test duplication for these branches (utils unit tests cover
normalize/parse_last_user_input happy paths; sse_utils BDD binding already links
the normalize dups)."""

import asyncio
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import utils

FEATURE = "features/utils_gaps.feature"
scenarios(FEATURE)


@pytest.fixture
def ctx() -> dict:
    return {}


@pytest.fixture(autouse=True)
def clear_mapping_cache():
    utils._get_upstream_to_model_id_mapping.cache_clear()
    yield
    utils._get_upstream_to_model_id_mapping.cache_clear()


@given("the utils gaps harness")
def given_harness(ctx: dict) -> None:
    ctx["value"] = None
    ctx["trim_calls"] = []


@given(parsers.parse('model settings contain a non-llm config "{model_id}"'))
def given_non_llm(ctx: dict, model_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        utils.model_settings,
        "models",
        {model_id: {"type": "embedding", "upstream_model": "embed-x"}},
    )


@given(parsers.parse('model settings contain upstream "{upstream}" for "{model_id}"'))
def given_upstream(
    ctx: dict, upstream: str, model_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        utils.model_settings,
        "models",
        {model_id: {"backend": "vllm", "upstream_model": upstream}},
    )


@given('find_library returns "c" and CDLL exposes malloc_trim')
def given_libc_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_trim(_size: int) -> int:
        return 1

    fake_trim.argtypes = None
    fake_trim.restype = None
    fake_libc = SimpleNamespace(malloc_trim=fake_trim)
    monkeypatch.setattr(utils.ctypes.util, "find_library", lambda _n: "c")
    monkeypatch.setattr(utils.ctypes, "CDLL", lambda _n, use_errno=False: fake_libc)


@given("malloc trim is stubbed as a counting function")
def given_trim_stub(ctx: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_trim(_size: int) -> int:
        ctx["trim_calls"].append(_size)
        return 1

    monkeypatch.setattr(utils, "_malloc_trim", fake_trim)


@when("the upstream to model id mapping is refreshed")
def when_mapping_refreshed(ctx: dict) -> None:
    utils._get_upstream_to_model_id_mapping.cache_clear()
    ctx["mapping"] = utils._get_upstream_to_model_id_mapping()


@when(parsers.parse('normalize model name is called with "{litellm_model}"'))
def when_normalize(ctx: dict, litellm_model: str) -> None:
    ctx["value"] = utils.normalize_model_name(litellm_model)


@when("get malloc trim runs")
def when_get_malloc_trim(ctx: dict) -> None:
    ctx["value"] = utils.get_malloc_trim()


@when("the periodic malloc trim loop runs briefly")
def when_periodic(ctx: dict) -> None:
    async def main() -> None:
        task = asyncio.ensure_future(utils.periodic_malloc_trim(0))
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(main())


@when(parsers.parse('token estimate is computed for a dict content with type "{kind}"'))
def when_dict_estimate(ctx: dict, kind: str) -> None:
    ctx["value"] = utils.get_token_count_estimate({"type": kind})


@then(parsers.parse('"{model_id}" is not in the mapping keys'))
def then_not_in_mapping(ctx: dict, model_id: str) -> None:
    assert model_id not in ctx["mapping"]


@then(parsers.parse('the normalized model is "{expected}"'))
def then_normalized(ctx: dict, expected: str) -> None:
    assert ctx["value"] == expected


@then("a callable is returned")
def then_callable(ctx: dict) -> None:
    assert callable(ctx["value"])


@then("the trim function was called at least once")
def then_trim_called(ctx: dict) -> None:
    assert len(ctx["trim_calls"]) >= 1


@then("the token estimate is 0")
def then_estimate_zero(ctx: dict) -> None:
    assert ctx["value"] == 0
