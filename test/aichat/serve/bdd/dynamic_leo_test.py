# BDD: dynamic leo signals + embedding gemma.
# TODO: remove test_run_dynamic_leo_logs_keyword_matches \
#   test/aichat/serve/services/dynamic_leo/signals_test.py:10
# TODO: remove test_run_dynamic_leo_continues_when_embedding_tasks_fail \
#   test/aichat/serve/services/dynamic_leo/signals_test.py:44
# TODO: remove test_generate_embeddings_with_timeout_returns_empty_on_timeout \
#   test/aichat/serve/services/dynamic_leo/test_embedding_gemma.py:20

import asyncio

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve.services.dynamic_leo import (
    androcles_labels,
    embedding_gemma,
    signals,
    user_text,
)
from aichat.serve.services.dynamic_leo import (
    config as dleo_config,
)
from aichat.serve.services.dynamic_leo.settings import dynamic_leo_settings
from aichat.serve.services.model_settings import model_settings

FEATURE = "features/dynamic_leo.feature"
scenarios(FEATURE)

from test.aichat.serve.bdd.helpers import new_patch as _new_patch


@pytest.fixture
def ctx():
    return {"undo": []}


@pytest.fixture(autouse=True)
def _dleo_cleanup(ctx):
    # Reset caches before AND after, then restore the pre-scenario values:
    # nulling alone would leave later tests reading empty caches as a
    # "populated miss" instead of the real prior state.
    saved = {
        "emb_sig": embedding_gemma._embedding_cache_sig,
        "emb_vec": dict(embedding_gemma._embedding_vectors),
        "cat_sig": signals._categories_sig,
        # Keep the original None-ness: restoring None as {} would read as a
        # populated cache miss.
        "cat_cache": (
            None
            if signals._categories_cache is None
            else dict(signals._categories_cache)
        ),
    }

    def _reset():
        embedding_gemma._embedding_cache_sig = None
        embedding_gemma._embedding_vectors = {}
        signals._categories_sig = None
        signals._categories_cache = None

    def _restore():
        embedding_gemma._embedding_cache_sig = saved["emb_sig"]
        embedding_gemma._embedding_vectors = saved["emb_vec"]
        signals._categories_sig = saved["cat_sig"]
        signals._categories_cache = saved["cat_cache"]

    _reset()
    yield
    _restore()
    for mp in reversed(ctx.get("undo", [])):
        mp.undo()


def _category(similar=None, keywords=None, labels=None):
    return dleo_config.DynamicLeoCategoryConfig(
        keywords=keywords or [], similar=similar or [], labels=labels or {}
    )


@given("the dynamic leo harness")
def given_harness(ctx):
    pass


# --- embedding_gemma ---


@when("the embedding input for a blank phrase is formatted")
def when_format_blank(ctx):
    ctx["formatted"] = embedding_gemma.format_embeddinggemma_input("   \n")


@then("the formatted input is empty")
def then_formatted_empty(ctx):
    assert ctx["formatted"] == ""


SIM_CASES = {
    "[1, 0]": [1.0, 0.0],
    "[0, 1]": [0.0, 1.0],
    "[]": [],
    "[1]": [1.0],
    "[0, 0]": [0.0, 0.0],
}


@when(parsers.parse("the cosine similarity of {left} and {right} is computed"))
def when_cosine(ctx, left, right):
    ctx["similarity"] = embedding_gemma.cosine_similarity(
        SIM_CASES[left], SIM_CASES[right]
    )


@then(parsers.parse("the similarity result is {result}"))
def then_similarity(ctx, result):
    assert ctx["similarity"] == float(result)


@given("the embedding timeout is disabled")
def given_no_timeout(ctx):
    mp = _new_patch(ctx)
    mp.setattr(dynamic_leo_settings, "dynamic_leo_embedding_timeout_seconds", 0.0)
    mp.setattr(
        embedding_gemma,
        "generate_embeddings",
        _fake_embeddings([{"embedding": [1.0, 0.1]}] * 2),
    )


def _fake_embeddings(rows):
    async def fake_generate(model_id, input_data):
        return {"data": rows}

    return fake_generate


@when("phrase vectors are fetched for two phrases")
def when_fetch_vectors(ctx):
    ctx["vectors"] = asyncio.run(
        embedding_gemma.fetch_vectors_for_unique_phrases("embed-x", ["alpha", "beta"])
    )


@then("the vectors map each phrase to an embedding")
def then_vectors(ctx):
    assert set(ctx["vectors"]) == {"alpha", "beta"}


@when("phrase vectors are fetched but the backend returns fewer rows")
def when_fetch_short(ctx):
    mp = _new_patch(ctx)
    mp.setattr(
        embedding_gemma,
        "generate_embeddings",
        _fake_embeddings([{"embedding": [1.0]}]),
    )
    ctx["vectors"] = asyncio.run(
        embedding_gemma.fetch_vectors_for_unique_phrases("embed-x", ["alpha", "beta"])
    )


@then("no vectors are recorded")
def then_no_vectors(ctx):
    assert ctx["vectors"] == {}


@when("phrase vectors are fetched and the backend raises")
def when_fetch_raises(ctx):
    mp = _new_patch(ctx)

    async def boom(model_id, input_data):
        raise RuntimeError("embedding service down")

    mp.setattr(embedding_gemma, "generate_embeddings", boom)
    ctx["vectors"] = asyncio.run(
        embedding_gemma.fetch_vectors_for_unique_phrases("embed-x", ["alpha"])
    )


@when("phrase vectors are fetched with an embedding-free row")
def when_fetch_partial(ctx):
    mp = _new_patch(ctx)
    mp.setattr(
        embedding_gemma,
        "generate_embeddings",
        _fake_embeddings([{"embedding": [1.0, 0.0]}, {"embedding": []}]),
    )
    ctx["vectors"] = asyncio.run(
        embedding_gemma.fetch_vectors_for_unique_phrases("embed-x", ["alpha", "beta"])
    )


@then("only the usable phrase has a vector")
def then_partial_vectors(ctx):
    assert set(ctx["vectors"]) == {"alpha"}


@when("phrase vectors are fetched twice for the same phrases")
def when_fetch_twice(ctx):
    calls = []

    async def fake_generate(model_id, input_data):
        calls.append(input_data)
        return {"data": [{"embedding": [1.0, 0.5]}]}

    mp = _new_patch(ctx)
    mp.setattr(embedding_gemma, "generate_embeddings", fake_generate)
    ctx["calls"] = calls
    first = asyncio.run(
        embedding_gemma.phrase_vectors_for_request("embed-x", ["alpha"])
    )
    second = asyncio.run(
        embedding_gemma.phrase_vectors_for_request("embed-x", ["alpha"])
    )
    ctx["same_object"] = first is second


@then("the backend is called only once")
def then_called_once(ctx):
    assert len(ctx["calls"]) == 1
    assert ctx["same_object"] is True


@when("phrase vectors are fetched then refetched for other phrases")
def when_refetch(ctx):
    calls = []

    async def fake_generate(model_id, input_data):
        calls.append(input_data)
        return {"data": [{"embedding": [1.0, 0.5]}]}

    mp = _new_patch(ctx)
    mp.setattr(embedding_gemma, "generate_embeddings", fake_generate)
    ctx["calls"] = calls
    asyncio.run(embedding_gemma.phrase_vectors_for_request("embed-x", ["alpha"]))
    asyncio.run(embedding_gemma.phrase_vectors_for_request("embed-x", ["beta"]))


@then("the backend is called twice")
def then_called_twice(ctx):
    assert len(ctx["calls"]) == 2


@given("category phrases with stored vectors")
def given_phrase_vectors(ctx):
    ctx["categories"] = {"tracking": _category(similar=["track my package"])}
    ctx["phrase_vectors"] = {"track my package": [1.0, 0.0]}


@when(parsers.parse("embedding reasons are computed for a {query} query"))
def when_query_reasons(ctx, query):
    vec = [1.0, 0.0] if query == "matching" else [0.0, 1.0]
    ctx["reasons"] = embedding_gemma.embedding_reasons_for_query_vec(
        vec, ctx["categories"], ctx["phrase_vectors"], 0.72, "embedding_last"
    )


@then(parsers.parse("the matched categories are {matched}"))
def then_matched(ctx, matched):
    expected = {"tracking"} if matched == "tracking" else set()
    assert set(ctx["reasons"]) == expected


@when("embedding match reasons are computed for blank text")
def when_match_blank(ctx):
    ctx["reasons"] = asyncio.run(
        embedding_gemma.embedding_match_reasons_for_text(
            "   ",
            "embed-x",
            {"c": _category(similar=["x"])},
            {"x": [1.0]},
            0.72,
            reason_tag="embedding_last",
        )
    )


@then("no embedding reasons are returned")
def then_no_reasons(ctx):
    assert ctx["reasons"] == {}


@when("embedding match reasons are computed and the backend raises")
def when_match_raises(ctx):
    mp = _new_patch(ctx)

    async def boom(model_id, input_data):
        raise RuntimeError("embedding down")

    mp.setattr(embedding_gemma, "generate_embeddings", boom)
    ctx["reasons"] = asyncio.run(
        embedding_gemma.embedding_match_reasons_for_text(
            "find my parcel",
            "embed-x",
            {"c": _category(similar=["x"])},
            {"x": [1.0]},
            0.72,
            reason_tag="embedding_last",
        )
    )


# --- signals ---


@when("the dynamic leo config is parsed twice")
def when_config_twice(ctx):
    raw = {"tracking": {"similar": ["track my package"]}}
    first = signals._categories_from_config(raw)
    second = signals._categories_from_config(raw)
    ctx["same_object"] = first is second


@then("the cached categories object is reused")
def then_cache_reused(ctx):
    assert ctx["same_object"] is True


@when(parsers.parse("the probabilities {left} and {right} are merged"))
def when_merge_probs(ctx, left, right):
    ctx["merged"] = signals._merge_probs_for_triage(
        _json_loads(left), _json_loads(right)
    )


def _json_loads(text):
    import json

    return json.loads(text)


@then(parsers.parse("the merged probabilities are {merged}"))
def then_merged(ctx, merged):
    assert ctx["merged"] == _json_loads(merged)


@given(parsers.parse("the explicit embedding model is {explicit}"))
def given_explicit_model(ctx, explicit):
    ctx["explicit"] = None if explicit == "none" else explicit.strip()


@given(parsers.parse("the configured models are {spec}"))
def given_models_spec(ctx, spec):
    models = {}
    for entry in spec.split(" and "):
        name, kind = entry.strip().split(" is ")
        models[name.strip()] = {"type": kind.strip()}
    ctx["configured"] = models
    ctx["resolved"] = None


@when("the embedding model id is resolved")
def when_resolve_model(ctx):
    mp = _new_patch(ctx)
    mp.setattr(
        dynamic_leo_settings, "dynamic_leo_embedding_model", ctx["explicit"] or ""
    )
    mp.setattr(model_settings, "models", ctx["configured"])
    ctx["resolved"] = signals._embedding_model_id()


@then(parsers.parse("the resolved embedding model is {resolved}"))
def then_resolved(ctx, resolved):
    # chat-model -> embed-x is intentional: an explicit model that is not
    # type=embedding warns and falls back to auto-discovery (signals.py).
    assert ctx["resolved"] == (None if resolved == "none" else resolved.strip())


@when("dynamic leo runs while disabled")
def when_dleo_disabled(ctx):
    mp = _new_patch(ctx)
    mp.setattr(dynamic_leo_settings, "enable_dynamic_leo", False)
    ctx["prefetch"] = asyncio.run(signals.run_dynamic_leo([]))


@then("no prefetch is returned")
def then_no_prefetch(ctx):
    assert ctx["prefetch"] is None


@when("dynamic leo runs without user text")
def when_dleo_no_text(ctx):
    mp = _new_patch(ctx)
    mp.setattr(dynamic_leo_settings, "enable_dynamic_leo", True)
    from aichat.protocol.open_ai_protocol import AssistantMessage

    ctx["prefetch"] = asyncio.run(
        signals.run_dynamic_leo([AssistantMessage(content="hi")])
    )


@given("dynamic leo with keywords and similar phrases")
def given_dleo_categories(ctx):
    mp = _new_patch(ctx)
    raw_config = {
        "tracking": {
            "keywords": ["track my package"],
            "similar": ["where is my parcel"],
            "labels": {"Coding": 0.9},
        }
    }
    mp.setattr(dynamic_leo_settings, "enable_dynamic_leo", True)
    mp.setattr(dynamic_leo_settings, "dynamic_leo_config", raw_config)
    mp.setattr(model_settings, "models", {"embed-x": {"type": "embedding"}})


@when("phrase prefetch raises while dynamic leo runs")
def when_prefetch_raises(ctx):
    mp = _new_patch(ctx)
    # model_settings.models already patched by the given above.
    mp.setattr(signals, "androcles_inference", _async_return(None))

    async def boom(model_id, phrases):
        raise RuntimeError("prefetch down")

    mp.setattr(signals, "phrase_vectors_for_request", boom)

    async def _no_embeddings(*args, **kwargs):
        raise AssertionError("embedding path must not run when prefetch fails")

    mp.setattr(signals, "embedding_match_reasons_for_text", _no_embeddings)

    ctx["prefetch"] = asyncio.run(
        signals.run_dynamic_leo([_user_message("track my package please")])
    )


def _user_message(text):
    from aichat.protocol.open_ai_protocol import UserMessage

    return UserMessage(role="user", content=text)


@then("the prefetch completes with keyword categories only")
def then_keyword_only(ctx):
    # androcles_inference is stubbed to None, so no task type may leak in;
    # the classification must come from keyword categories alone.
    assert ctx["prefetch"].task_type is None
    assert ctx["prefetch"].matched_categories == frozenset({"tracking"})


@given(parsers.parse("a category with the label {label} at threshold {threshold}"))
def given_label_category(ctx, label, threshold):
    ctx["label_categories"] = {"coding": _category(labels={label: float(threshold)})}


@when("label reasons are computed for probabilities hitting the label")
def when_label_reasons(ctx):
    probs = [0.0] * androcles_labels.ANDROCLES_NUM_LABELS
    coding_index = androcles_labels.androcles_label_index("Coding")
    assert coding_index is not None
    probs[coding_index] = 0.95
    ctx["reasons"] = androcles_labels.label_reasons(
        ctx["label_categories"], probs, None
    )


@then("the matched category is reported with a labels reason")
def then_label_reason(ctx):
    assert ctx["reasons"] == {"coding": frozenset({"labels"})}


@when("the dynamic leo config is parsed from a list")
def when_config_list(ctx):
    ctx["parsed"] = signals._categories_from_config([1, 2, 3])


@then("the parsed categories are none")
def then_parsed_none(ctx):
    assert ctx["parsed"] is None


@when("the last user turns are extracted with a blank message")
def when_blank_turns(ctx):
    ctx["turns"] = user_text.extract_last_n_user_plain_text_turns(
        [_user_message("   ")], 3
    )


@then("no turns are returned")
def then_no_turns(ctx):
    assert ctx["turns"] == []


@when(parsers.parse("the androcles label index for {name} is looked up"))
def when_label_index(ctx, name):
    ctx["index"] = androcles_labels.androcles_label_index(name)


@then(parsers.parse("the label index is {index}"))
def then_label_index(ctx, index):
    # The literal numbers pin the ANDROCLES_LABELS order on purpose: the
    # Triton probability vector layout is a fixed contract with the model.
    assert ctx["index"] == (None if index == "none" else int(index))


@given("dynamic leo with keywords in both turns")
def given_keywords_both_turns(ctx):
    mp = _new_patch(ctx)
    mp.setattr(dynamic_leo_settings, "enable_dynamic_leo", True)
    mp.setattr(
        dynamic_leo_settings,
        "dynamic_leo_config",
        {"tracking": {"keywords": ["track my package"], "labels": {"Coding": 0.9}}},
    )
    from aichat.protocol.open_ai_protocol import UserMessage

    ctx["messages"] = [
        UserMessage(content="track my package now"),
        UserMessage(content="track my package again"),
    ]


@when("dynamic leo runs with keyword hits in the last and prior turns")
def when_dleo_keywords(ctx):
    mp = _new_patch(ctx)
    mp.setattr(
        signals,
        "androcles_inference",
        _async_return(None),
    )
    # Keyword-only config must never reach the embedding path; make any
    # accidental embedding call fail loudly.
    for fn in ("phrase_vectors_for_request", "embedding_match_reasons_for_text"):

        async def _boom(*args, **kwargs):
            raise AssertionError("embedding path must not run for keyword configs")

        mp.setattr(signals, fn, _boom)
    ctx["prefetch"] = asyncio.run(signals.run_dynamic_leo(ctx["messages"]))


def _async_return(value):
    async def fake(*args, **kwargs):
        return value

    return fake


@then(parsers.parse("the matched categories merge both keyword reasons"))
def then_merged_keyword(ctx):
    # Both turns hit the same keyword, so their reasons must merge under the
    # single "tracking" category — an exact frozenset also rules out
    # duplicates or spurious categories from either turn. The individual
    # reason strings are not observable: AndroclesPrefetch only carries the
    # merged category frozenset and the task type.
    assert ctx["prefetch"].matched_categories == frozenset({"tracking"})
