import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import utils
from aichat.serve.conversation_settings import conversation_settings
from aichat.serve.services.model_settings import model_settings

FEATURE = Path(__file__).parent / "features" / "sse_utils.feature"
scenarios(str(FEATURE))


@pytest.fixture
def ctx():
    return {}


# --- normalize_model_name --------------------------------------------------------


@given(parsers.parse('a model catalog where "{model_id}" maps upstream "{upstream}"'))
def given_catalog(model_id, upstream, monkeypatch, ctx):
    ctx["catalog"] = {
        model_id: {"type": "llm", "upstream_model": upstream},
        "titan": {
            "type": "llm",
            "backend": "bedrock",
            "inference_profile": "TITAN_PROFILE",
        },
    }


@given(
    parsers.parse(
        'the bedrock model "titan" uses inference profile env "TITAN_PROFILE"'
    )
)
def given_bedrock_profile(ctx):
    pass


@given(parsers.parse('the environment variable "TITAN_PROFILE" is set to "titan-v2"'))
def given_profile_env(monkeypatch):
    monkeypatch.setenv("TITAN_PROFILE", "titan-v2")


@when(parsers.parse('the litellm model "{litellm}" is normalized'))
def when_normalize(litellm, monkeypatch, ctx):
    model_settings.models = ctx["catalog"]
    try:
        utils._get_upstream_to_model_id_mapping.cache_clear()
        ctx["normalized"] = utils.normalize_model_name(
            None if litellm == "<empty>" else litellm
        )
    finally:
        utils._get_upstream_to_model_id_mapping.cache_clear()


@then(parsers.parse("the normalized model is {normalized}"))
def then_normalized(normalized, ctx):
    expected = None if normalized == "none" else normalized
    assert ctx["normalized"] == expected


# --- parse_last_user_input -------------------------------------------------------


@given(
    parsers.parse(
        'a prompt transcript ending with the response seed and user question "{question}"'
    )
)
def given_transcript(question, ctx):
    ctx["transcript"] = f"[INST] <</SYS>>\n {question} [/INST] Here is your response:"


@given("a prompt transcript without the response seed")
def given_transcript_no_seed(ctx):
    ctx["transcript"] = "[INST] some random prompt text without seed"


@when("the last user input is parsed")
def when_parse_last_input(ctx):
    ctx["parsed"] = utils.parse_last_user_input(ctx["transcript"])


@then(parsers.parse("the parsed input is {expected}"))
def then_parsed(expected, ctx):
    expected = expected.strip('"')
    assert ctx["parsed"] == ("" if expected == "nothing" else expected)


# --- static content generators ----------------------------------------------------


@given(parsers.parse('static content "{content}"'))
def given_static_content(content, ctx):
    ctx["static_content"] = content


@when("the static content is streamed with no delay")
def when_stream_static(ctx):
    async def run():
        return [
            chunk
            async for chunk in utils.static_content_generator(ctx["static_content"], 0)
        ]

    ctx["chunks"] = asyncio.run(run())


@then(parsers.parse('the chunks reassemble to "{content}"'))
def then_reassemble(content, ctx):
    assert "".join(ctx["chunks"]) == content


@then("the last chunk is not longer than 5 characters")
def then_chunk_size(ctx):
    assert all(len(c) <= 5 for c in ctx["chunks"])


@when(parsers.parse('the static content is streamed as SSE for model "{model}"'))
def when_stream_sse(model, ctx):
    async def run():
        return [
            event
            async for event in utils.generate_static_content_generator(
                model, ctx["static_content"], 0
            )
        ]

    ctx["sse_events"] = asyncio.run(run())


@then("every SSE event except the last is a completion with cumulative content")
def then_completion_events(ctx):
    events = ctx["sse_events"]
    cumulative = ""
    for raw in events[:-1]:
        assert raw.startswith("data: {")
        payload = json.loads(raw[len("data: ") :])
        cumulative += payload["completion"][len(cumulative) :]
    assert cumulative == ctx["static_content"]


@then("the final SSE event is the DONE sentinel")
def then_final_done(ctx):
    assert ctx["sse_events"][-1].strip() == "data: [DONE]"


# --- parse_free_model_names --------------------------------------------------------


@given(
    parsers.parse(
        'a model catalog where "{free_model}" is free, "{premium_model}" is not'
        ' and "{embedding_model}" is a non-llm free model'
    )
)
def given_free_catalog(free_model, premium_model, embedding_model, monkeypatch, ctx):
    ctx["free_catalog"] = {
        free_model: {"type": "llm", "free": True},
        premium_model: {"type": "llm", "free": False},
        embedding_model: {"type": "embedding", "free": True},
    }
    monkeypatch.setattr(
        utils, "model_settings", SimpleNamespace(models=ctx["free_catalog"])
    )


@when("the free model names are parsed")
def when_parse_free(ctx):
    ctx["free_models"] = utils.parse_free_model_names(ctx["free_catalog"])


@then(parsers.parse('the free models are "{expected}"'))
def then_free_models(expected, ctx):
    assert ctx["free_models"] == [expected]


# --- get_rate_limiting_key ----------------------------------------------------------


@given(
    parsers.parse('rate key "{rate_key}" for model "{model}" and interval {interval:d}')
)
def given_rate_key_inputs(rate_key, model, interval, ctx):
    ctx["rate_inputs"] = (model, rate_key, interval)


@when("the rate limiting key is built twice")
def when_build_rate_key(ctx):
    model, rate_key, interval = ctx["rate_inputs"]
    ctx["rate_key_one"] = utils.get_rate_limiting_key(model, rate_key, interval)
    ctx["rate_key_two"] = utils.get_rate_limiting_key(model, rate_key, interval)


@then("the two keys are equal")
def then_keys_equal(ctx):
    assert ctx["rate_key_one"] == ctx["rate_key_two"]


@then("the key contains the model and the rate key")
def then_key_parts(ctx):
    model, rate_key, _ = ctx["rate_inputs"]
    assert model in ctx["rate_key_one"]
    assert rate_key in ctx["rate_key_one"]


# --- SSEResponse ----------------------------------------------------------------------


@given(parsers.parse('a raw generator emitting "{first}" and "{second}"'))
def given_raw_generator(first, second, ctx):
    async def gen():
        try:
            yield first
            yield second
        finally:
            ctx["generator_closed"] = True

    ctx["generator"] = gen


@given(parsers.parse('a raw generator emitting "{first}"'))
def given_raw_generator_one(first, ctx):
    async def gen():
        try:
            yield first
        finally:
            ctx["generator_closed"] = True

    ctx["generator"] = gen


@when("the SSE response is consumed")
def when_consume_sse(ctx):
    response = utils.SSEResponse(ctx["generator"]())

    async def run():
        return [event async for event in response.body_iterator]

    ctx["sse_framed"] = asyncio.run(run())


@then('the events are "data: one", "data: two" and the DONE sentinel')
def then_framed_events(ctx):
    expected = ["data: one\n\n", "data: two\n\n", "data: [DONE]\n\n"]
    assert ctx["sse_framed"] == expected


@then("the generator was closed")
def then_generator_closed(ctx):
    assert ctx["generator_closed"] is True


@given(parsers.parse('a custom header "{name}" set to "{value}"'))
def given_custom_header(name, value, ctx):
    ctx["custom_headers"] = {name: value}


@when("the SSE response headers are inspected")
def when_inspect_headers(ctx):
    response = utils.SSEResponse(ctx["generator"](), ctx["custom_headers"])
    ctx["sse_headers"] = response.headers


@then(parsers.parse('the header "{name}" is "{value}"'))
def then_header(name, value, ctx):
    assert ctx["sse_headers"][name] == value


# --- token counting ---------------------------------------------------------------------


@given(parsers.parse("token counting input {description}"))
def given_counting_input(description, ctx, monkeypatch):
    ctx["counting_kind"] = description.replace(" ", "_").replace("-", "_")
    if description == "a_broken_tokenizer" or description.startswith("calculate"):
        ctx["counting_messages"] = [{"role": "user", "content": "hello there friend"}]


@when("the tokens are counted")
def when_count_tokens(ctx, monkeypatch):
    kind = ctx["counting_kind"]
    messages = ctx.get("counting_messages") or [
        {"role": "user", "content": "hello there friend"}
    ]
    if kind == "plain_text_messages":
        ctx["token_count"] = utils.count_tokens(messages)
    elif kind == "multimodal_text_parts":
        ctx["token_count"] = utils.count_tokens(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}, {"type": "other"}],
                }
            ]
        )
    elif kind == "a_broken_tokenizer":
        with patch.object(utils.tiktoken, "get_encoding", side_effect=RuntimeError):
            ctx["token_count"] = utils.count_tokens(messages)
    elif kind == "calculate_plain_content":
        ctx["token_count"] = utils.calculate_message_tokens(messages)
    elif kind == "calculate_non_text_part":
        ctx["token_count"] = utils.calculate_message_tokens(
            [{"role": "user", "content": [{"type": "image_url"}]}]
        )
    elif kind == "calculate_tool_calls":
        ctx["token_count"] = utils.calculate_message_tokens(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}],
                }
            ]
        )


@then(parsers.parse("the token count is greater than zero {validity}"))
def then_token_count(validity, ctx):
    assert ctx["token_count"] > 0


# --- get_token_count_estimate --------------------------------------------------------------


@given("a custom tokenizer counting 3 tokens per encode")
def given_custom_tokenizer(ctx):
    tokenizer = MagicMock()

    def encode(content, add_special_tokens=False):
        assert add_special_tokens is False
        return [1, 2, 3]

    tokenizer.encode = encode
    ctx["custom_tokenizer"] = tokenizer


@given(parsers.parse("token estimate content {description}"))
def given_estimate_content(description, ctx):
    ctx["estimate_kind"] = description.replace(" ", "_")
    if ctx["estimate_kind"] == "a_plain_string":
        ctx["estimate_content"] = "hello world"
    elif ctx["estimate_kind"] == "a_nested_list":
        ctx["estimate_content"] = ["hello", "world"]
    elif ctx["estimate_kind"] == "a_text_dict":
        ctx["estimate_content"] = {"type": "text", "text": "hello world"}
    elif ctx["estimate_kind"] == "an_image_dict":
        ctx["estimate_content"] = {"type": "image_url"}
    elif ctx["estimate_kind"] == "an_unknown_dict":
        ctx["estimate_content"] = {"mystery": 1}
    else:  # an_unrecognized_object
        ctx["estimate_content"] = 42


@when("the token count estimate is computed with the custom tokenizer")
def when_estimate_custom(ctx):
    content = ctx["estimate_content"]
    ctx["estimate"] = utils.get_token_count_estimate(
        content, tokenizer=ctx["custom_tokenizer"]
    )


@when(
    parsers.parse(
        'the token count estimate is computed for "{content}" with the custom tokenizer'
    )
)
def when_estimate_string(content, ctx):
    ctx["estimate"] = utils.get_token_count_estimate(
        content, tokenizer=ctx["custom_tokenizer"]
    )


@then(parsers.parse("the estimate is {estimate}"))
def then_estimate(estimate, ctx):
    if estimate == "image_estimate":
        assert ctx["estimate"] == conversation_settings.image_token_estimate
    else:
        assert ctx["estimate"] == int(estimate)


# --- get_malloc_trim ---------------------------------------------------------------------


@given(parsers.parse("libc lookup {libc_lookup}"))
def given_libc_lookup(libc_lookup, monkeypatch, ctx):
    ctx["libc_lookup"] = libc_lookup


@when("malloc trim support is loaded")
def when_load_malloc_trim(ctx, monkeypatch):
    import ctypes
    import ctypes.util

    lookup = ctx["libc_lookup"]
    if lookup == "missing":
        monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    elif lookup == "load failure":
        monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libc")
        monkeypatch.setattr(ctypes, "CDLL", MagicMock(side_effect=OSError("no libc")))
    else:  # no trim symbol
        monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libc")
        monkeypatch.setattr(ctypes, "CDLL", MagicMock(return_value=SimpleNamespace()))
    ctx["trim_fn"] = utils.get_malloc_trim()


@then(parsers.parse("the trim function is {trim_result}"))
def then_trim_fn(trim_result, ctx):
    assert ctx["trim_fn"] is None


# --- periodic_malloc_trim -----------------------------------------------------------------


@given("a no-op trim function")
def given_noop_trim(monkeypatch):
    monkeypatch.setattr(utils, "_malloc_trim", None)


@when("the periodic trim loop runs for one tick")
def when_periodic_trim(ctx):
    async def run():
        task = asyncio.create_task(utils.periodic_malloc_trim(0))
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"

    ctx["trim_loop"] = asyncio.run(run())


@then("the loop exits cleanly without trimming")
def then_trim_loop(ctx):
    assert ctx["trim_loop"] == "cancelled"
