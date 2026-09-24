"""BDD bindings for media service generators (STT/TTS/embeddings/image/systemone).

Duplicated-coverage TODO map (unit tests already cover some scenarios):
# TODO: remove embeddings_generator_test.py#test_transform_openai_to_triton_single_string test/aichat/serve/services/embeddings_generator_test.py:8
# TODO: remove embeddings_generator_test.py#test_transform_openai_to_triton_list test/aichat/serve/services/embeddings_generator_test.py:16
# TODO: remove embeddings_generator_test.py#test_transform_triton_to_openai test/aichat/serve/services/embeddings_generator_test.py:22
# TODO: remove embeddings_generator_test.py#test_transform_triton_to_openai_no_outputs test/aichat/serve/services/embeddings_generator_test.py:42
# TODO: remove system_one_generator_test.py#test_build_triton_request_serializes_state_and_questions test/aichat/serve/services/system_one_generator_test.py:13
# TODO: remove system_one_generator_test.py#test_build_triton_request_passes_model_override test/aichat/serve/services/system_one_generator_test.py:26
# TODO: remove system_one_generator_test.py#test_format_system_one_response_strips_routing_by_default test/aichat/serve/services/system_one_generator_test.py:39
# TODO: remove system_one_generator_test.py#test_decode_result_json_from_triton_bytes_output test/aichat/serve/services/system_one_generator_test.py:56
# TODO: remove system_one_generator_test.py#test_run_system_one_calls_passthrough test/aichat/serve/services/system_one_generator_test.py:70
"""

import ast
import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve.services.embeddings import generator as emb_gen
from aichat.serve.services.image_gen import generator as img_gen
from aichat.serve.services.stt import generator as stt_gen
from aichat.serve.services.system_one import generator as so_gen
from aichat.serve.services.tts import generator as tts_gen

FEATURE = "features/media_generation.feature"

scenarios(FEATURE)


@pytest.fixture
def ctx():
    return {}


@given("the media generator harness")
def _(ctx):
    return ctx


# --- STT ---


def _pool(mock_ctx, result):
    pool = MagicMock()
    pool.call = AsyncMock(return_value=result)
    return pool


@when("a WAV upload is decoded through the sandbox returning pcm for 16000 samples")
def _(ctx, monkeypatch):
    pcm = np.zeros(16000, dtype=np.float32).tobytes()
    result = {
        "pcm_b64": base64.b64encode(pcm).decode("ascii"),
        "n_samples": 16000,
        "sample_rate": 16000,
        "duration_ms": 250.0,
    }
    monkeypatch.setattr(stt_gen, "get_pool", lambda: _pool(ctx, result))
    ctx["audio"], ctx["duration_ms"] = asyncio.run(
        stt_gen.decode_wav_pcm_to_mono_16k_float32(b"wav")
    )


@when("the sandbox returns a sample count that does not match the buffer")
def _(ctx, monkeypatch):
    pcm = np.zeros(10, dtype=np.float32).tobytes()
    result = {
        "pcm_b64": base64.b64encode(pcm).decode(),
        "n_samples": 5,
        "duration_ms": 1.0,
    }
    monkeypatch.setattr(stt_gen, "get_pool", lambda: _pool(ctx, result))
    try:
        asyncio.run(stt_gen.decode_wav_pcm_to_mono_16k_float32(b"wav"))
    except ValueError as e:
        ctx["error"] = e


@then(
    parsers.parse(
        "the decoded audio has {count:d} samples and duration {duration} milliseconds"
    )
)
def _(ctx, count, duration):
    assert ctx["audio"].size == count
    assert ctx["duration_ms"] == float(duration)


@then(parsers.parse('decoding raises ValueError "{text}"'))
def _(ctx, text):
    assert isinstance(ctx["error"], ValueError)
    assert str(ctx["error"]) == text


@when(parsers.parse('a triton response of kind "{kind}" is normalized'))
def _(ctx, kind):
    if kind == "dict":
        payload = {"outputs": []}
        ctx["result"] = stt_gen._triton_response_to_dict(payload)
        ctx["expected"] = payload
    elif kind == "json_object":
        payload = {"outputs": [{"name": "x"}]}
        ctx["result"] = stt_gen._triton_response_to_dict(
            SimpleNamespace(json=lambda: payload)
        )
        ctx["expected"] = payload
    else:
        try:
            stt_gen._triton_response_to_dict(object())
            ctx["error"] = None
        except ValueError as e:
            ctx["error"] = e


@then(parsers.parse("the conversion {outcome}"))
def _(ctx, outcome):
    if outcome == "raises ValueError":
        assert isinstance(ctx["error"], ValueError)
    else:
        assert ctx["result"] == ctx["expected"]


@given(parsers.parse("a triton transcription payload {payload}"))
def _(ctx, payload):
    ctx["triton_data"] = {
        "named_output_str": {
            "outputs": [{"name": "transcription", "data": ["hello there"]}]
        },
        "first_output_str": {"outputs": [{"name": "other", "data": ["fallback"]}]},
        "bytes_list": {"outputs": [{"name": "transcription", "data": [b"bytes text"]}]},
        "no_outputs": {},
        "missing_data": {"outputs": [{"name": "transcription"}]},
        "unexpected_type": {"outputs": [{"name": "transcription", "data": 123}]},
    }[payload]


@given(parsers.parse("a malformed transcription payload {defect}"))
def _(ctx, defect):
    ctx["triton_data"] = {
        "no_outputs": {},
        "missing_data": {"outputs": [{"name": "transcription"}]},
        "unexpected_type": {"outputs": [{"name": "transcription", "data": 123}]},
    }[defect]


@when("the transcription is extracted")
def _(ctx):
    try:
        ctx["text"] = stt_gen._extract_transcription(ctx["triton_data"])
        ctx["error"] = None
    except ValueError as e:
        ctx["error"] = e


@then(parsers.parse('the extracted text is "{text}"'))
def _(ctx, text):
    assert ctx["text"] == text


@then("extraction raises ValueError")
def _(ctx):
    assert isinstance(ctx["error"], ValueError)


@when("audio samples are transcribed through the router")
def _(ctx, monkeypatch):
    router = MagicMock()
    router.allm_passthrough_route = AsyncMock(
        return_value={"outputs": [{"name": "transcription", "data": ["hi there"]}]}
    )
    monkeypatch.setattr(stt_gen, "get_global_router", lambda: router)
    ctx["router"] = router
    audio = np.zeros(4, dtype=np.float32)
    ctx["transcript"] = asyncio.run(stt_gen.transcribe_audio("parakeet", audio))


@then(
    parsers.parse(
        "the triton request has an audio_signal tensor of {count:d} float32 samples"
    )
)
def _(ctx, count):
    inp = ctx["router"].allm_passthrough_route.await_args.kwargs["json"]["inputs"][0]
    assert inp["name"] == "audio_signal"
    assert inp["shape"] == [1, count]
    assert inp["datatype"] == "FP32"


@then(parsers.parse('the transcript "{text}" is returned'))
def _(ctx, text):
    assert ctx["transcript"] == text


def _wire_upload(ctx, monkeypatch, n_samples, text):
    pcm = np.zeros(n_samples, dtype=np.float32).tobytes()
    monkeypatch.setattr(
        stt_gen,
        "get_pool",
        lambda: _pool(
            ctx,
            {
                "pcm_b64": base64.b64encode(pcm).decode(),
                "n_samples": n_samples,
                "duration_ms": 0.0,
            },
        ),
    )
    router = MagicMock()
    router.allm_passthrough_route = AsyncMock(
        return_value={"outputs": [{"name": "transcription", "data": [text]}]}
    )
    monkeypatch.setattr(stt_gen, "get_global_router", lambda: router)


@when("an upload with 32000 samples is transcribed")
def _(ctx, monkeypatch):
    _wire_upload(ctx, monkeypatch, 32000, "hi")
    ctx["result"] = asyncio.run(
        stt_gen.transcribe_upload("parakeet", b"wav", "clip.wav")
    )


@then(
    parsers.parse('the transcript is "{text}" and duration is {duration} milliseconds')
)
def _(ctx, text, duration):
    transcript, duration_ms = ctx["result"]
    assert transcript == text
    assert duration_ms == float(duration)


@when("the triton backend returns blank text")
def _(ctx, monkeypatch):
    _wire_upload(ctx, monkeypatch, 16000, "  ")
    try:
        asyncio.run(stt_gen.transcribe_upload("parakeet", b"wav", "clip.wav"))
        ctx["error"] = None
    except RuntimeError as e:
        ctx["error"] = e


@then(parsers.parse('transcription raises RuntimeError "{text}"'))
def _(ctx, text):
    assert isinstance(ctx["error"], RuntimeError)
    assert str(ctx["error"]) == text


# --- TTS ---


@when(parsers.parse('speech is generated with voice "{voice}"'))
def _(ctx, voice, monkeypatch):
    router = MagicMock()
    router.aspeech = AsyncMock(return_value="speech-result")
    monkeypatch.setattr(tts_gen, "get_global_router", lambda: router)
    ctx["router"] = router
    ctx["result"] = asyncio.run(tts_gen.generate_speech(model="tts-m", voice=voice))


@then("the router aspeech call receives the same arguments")
def _(ctx):
    kwargs = ctx["router"].aspeech.await_args.kwargs
    assert kwargs == {"model": "tts-m", "voice": "af_heart"}


class _FakeUpstream:
    def __init__(self, status_code=200, chunks=(b"a", b"b"), body=b"oops"):
        self.status_code = status_code
        self._chunks = list(chunks)
        self._body = body
        self.aclose = AsyncMock()
        self.aread = AsyncMock(return_value=self._body)

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


def _fake_tts_http(monkeypatch, upstream):
    client = MagicMock()
    built = MagicMock()
    client.build_request = MagicMock(return_value=built)
    client.send = AsyncMock(return_value=upstream)
    client.aclose = AsyncMock()
    fake = MagicMock()
    fake.AsyncClient = MagicMock(return_value=client)
    monkeypatch.setattr(tts_gen, "httpx", fake)
    return client


@given(
    parsers.parse('a tts model config with address "{address}" and api key "{api_key}"')
)
def _(ctx, address, api_key):
    cfg = {"address": address}
    ctx["auth"] = "empty" if api_key == "none" else api_key
    if api_key != "none":
        cfg["api_key"] = api_key
    ctx["model_cfg"] = cfg
    ctx["url"] = address.rstrip("/") + "/audio/speech"


@when("speech streaming is requested")
def _(ctx, monkeypatch):
    monkeypatch.setattr(
        tts_gen, "model_settings", SimpleNamespace(models={"tts-m": ctx["model_cfg"]})
    )
    ctx["client"] = _fake_tts_http(monkeypatch, _FakeUpstream())
    stream = asyncio.run(tts_gen.stream_speech("tts-m", {"input": "hello"}))

    async def _collect():
        return [c async for c in stream]

    ctx["chunks"] = asyncio.run(_collect())


@then(
    parsers.parse(
        'the upstream request posts to "{url}" with Authorization "Bearer {auth}"'
    )
)
def _(ctx, url, auth):
    call = ctx["client"].build_request.call_args
    assert call.args[1] == url
    assert call.kwargs["headers"]["Authorization"] == f"Bearer {auth}"
    assert call.kwargs["headers"]["Content-Type"] == "application/json"


@then("the stream yields the upstream audio chunks")
def _(ctx):
    assert ctx["chunks"] == [b"a", b"b"]


@when("the upstream speech endpoint returns status 500")
def _(ctx, monkeypatch):
    monkeypatch.setattr(
        tts_gen, "model_settings", SimpleNamespace(models={"tts-m": ctx["model_cfg"]})
    )
    _fake_tts_http(monkeypatch, _FakeUpstream(status_code=500))
    try:
        asyncio.run(tts_gen.stream_speech("tts-m", {"input": "hello"}))
        ctx["error"] = None
    except RuntimeError as e:
        ctx["error"] = e


@then(parsers.parse('streaming raises RuntimeError "{text}"'))
def _(ctx, text):
    assert isinstance(ctx["error"], RuntimeError)
    assert text in str(ctx["error"])


# --- Embeddings ---


@when(parsers.parse("an embeddings request carries input {input_desc}"))
def _(ctx, input_desc):
    input_data = "hello" if input_desc == "a single string" else ["a", "b"]
    ctx["payload"] = emb_gen._transform_openai_to_triton("emb-model", input_data)


@then(parsers.parse("the triton payload has shape [{count:d}, 1] and data rows {rows}"))
def _(ctx, count, rows):
    inp = ctx["payload"]["json"]["inputs"][0]
    assert inp["shape"] == [count, 1]
    assert inp["data"] == ast.literal_eval(rows)


@given(
    parsers.parse(
        "a triton embeddings response with {input_count:d} inputs and a 4-float data array"
    )
)
def _(ctx, input_count):
    ctx["triton_response"] = {"outputs": [{"data": list(range(8))}]}
    ctx["input_count"] = input_count
    ctx["input_texts"] = ["a b c d e f g", "one two"] if input_count == 2 else ["a b c"]


@when("the response is transformed")
def _(ctx):
    ctx["result"] = emb_gen._transform_triton_to_openai(
        "emb-model", ctx["triton_response"], ctx["input_count"], ctx["input_texts"]
    )


@then(
    parsers.parse(
        "the OpenAI response has {input_count:d} vectors of dimension {dim:d} and usage tokens {tokens:d}"
    )
)
def _(ctx, input_count, dim, tokens):
    assert len(ctx["result"]["data"]) == input_count
    for item in ctx["result"]["data"]:
        assert len(item["embedding"]) == dim
    assert ctx["result"]["usage"]["total_tokens"] == tokens


@when("the triton embeddings response has no outputs")
def _(ctx):
    try:
        emb_gen._transform_triton_to_openai("emb-model", {"outputs": []}, 1, ["x"])
        ctx["error"] = None
    except ValueError as e:
        ctx["error"] = e


@then(parsers.parse('transforming raises ValueError "{text}"'))
def _(ctx, text):
    assert isinstance(ctx["error"], ValueError)
    assert str(ctx["error"]) == text


class _ObjWithGet:
    def __init__(self, data):
        self.json = "not-callable"
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


@when("the triton passthrough returns an object whose json attribute is not callable")
def _(ctx, monkeypatch):
    router = MagicMock()
    router.allm_passthrough_route = AsyncMock(
        return_value=_ObjWithGet({"outputs": [{"data": [1, 2, 3, 4]}]})
    )
    monkeypatch.setattr(emb_gen, "get_global_router", lambda: router)
    ctx["result"] = asyncio.run(
        emb_gen.generate_embeddings(model="emb-model", input="a b")
    )


@then("the embeddings response keeps the object data")
def _(ctx):
    assert ctx["result"]["data"][0]["embedding"] == [1, 2, 3, 4]


# --- Image generation ---


@when(parsers.parse('an image is generated with prompt "{prompt}"'))
def _(ctx, prompt, monkeypatch):
    router = MagicMock()
    router.aimage_generation = AsyncMock(return_value="image-result")
    monkeypatch.setattr(img_gen, "get_global_router", lambda: router)
    ctx["router"] = router
    ctx["result"] = asyncio.run(img_gen.generate_image(model="img-m", prompt=prompt))


@then("the router aimage_generation call receives the same arguments")
def _(ctx):
    kwargs = ctx["router"].aimage_generation.await_args.kwargs
    assert kwargs == {"model": "img-m", "prompt": "a cat"}


# --- System One generator ---


@given(parsers.parse("a systemone model config {config_desc}"))
def _(ctx, config_desc):
    if config_desc == "with defaults":
        cfg = {}
    else:
        cfg = {
            "method": "GET",
            "endpoint": "http://ep",
            "address": "http://addr",
            "api_key": "sk",
        }
    ctx["cfg"] = cfg


@when(parsers.parse('passthrough kwargs are built for model "{model}"'))
def _(ctx, monkeypatch):
    monkeypatch.setattr(
        so_gen, "model_settings", SimpleNamespace(models={"so-model": ctx["cfg"]})
    )
    ctx["kwargs"] = so_gen._allm_passthrough_kwargs("so-model", json="json-payload")


@then(parsers.parse("the kwargs are {expected}"))
def _(ctx, expected):
    assert ctx["kwargs"] == ast.literal_eval(expected)


@when(parsers.parse("the field value {value_desc} is serialized"))
def _(ctx, value_desc):
    value = "state-1" if value_desc == "a plain string" else {"a": 1}
    ctx["serialized"] = so_gen._serialize_json_field(value)


@then(parsers.parse("the serialized value is {expected}"))
def _(ctx, expected):
    assert ctx["serialized"] == ast.literal_eval(expected)


@given(parsers.parse("a triton result payload {payload}"))
def _(ctx, payload):
    good = '{"answers": %d}'
    ctx["triton_data"] = {
        "named_string": {"outputs": [{"name": "result_json", "data": [good % 1]}]},
        "first_output_string": {"outputs": [{"name": "other", "data": [good % 2]}]},
        "bytes_value": {
            "outputs": [{"name": "result_json", "data": [[bytes(good % 3, "utf-8")]]}]
        },
        "nested_list": {"outputs": [{"name": "result_json", "data": [[good % 4]]}]},
        "non_string_number": {"outputs": [{"name": "result_json", "data": [[5]]}]},
    }[payload]


@when("the result json is decoded")
def _(ctx):
    ctx["decoded"] = so_gen._decode_result_json(ctx["triton_data"])


@then(parsers.parse("the decoded payload is {decoded}"))
def _(ctx, decoded):
    assert ctx["decoded"] == ast.literal_eval(decoded)


@when(
    parsers.parse(
        'the backend payload {payload} is formatted with route_only "{route_only}" and include_routing "{include_routing}"'
    )
)
def _(ctx, payload, route_only, include_routing):
    ctx["formatted"] = so_gen.format_system_one_response(
        ast.literal_eval(payload),
        response_model="so-external",
        include_routing=include_routing == "true",
        route_only=route_only == "true",
    )


@then(parsers.parse("the formatted response is {expected}"))
def _(ctx, expected):
    assert ctx["formatted"] == ast.literal_eval(expected)


@when(
    parsers.parse(
        'the backend payload {payload} is formatted with route_only "{route_only}"'
    )
)
def _(ctx, payload, route_only):
    payload_desc = {"none_routing": "{}", "none_answers": "{}"}[payload]
    try:
        so_gen.format_system_one_response(
            ast.literal_eval(payload_desc),
            response_model="so-external",
            include_routing=False,
            route_only=route_only == "true",
        )
        ctx["error"] = None
    except ValueError as e:
        ctx["error"] = e


@then(parsers.parse('formatting raises ValueError "{message}"'))
def _(ctx, message):
    assert isinstance(ctx["error"], ValueError)
    assert str(ctx["error"]) == message


@when(
    parsers.parse(
        'a triton request is built with state "{state}" questions "{questions}" model_override "{override}" route_only {route_only}'
    )
)
def _(ctx, state, questions, override, route_only):
    ctx["request"] = so_gen.build_triton_request(
        state=state,
        questions=questions,
        model_override=override,
        route_only=route_only == "True",
    )


@then(
    "the request has tensors for state_json questions_json model_override and route_only"
)
def _(ctx):
    names = [inp["name"] for inp in ctx["request"]["inputs"]]
    assert names == ["state_json", "questions_json", "model_override", "route_only"]
    assert ctx["request"]["inputs"][2]["data"] == [["ov"]]
    assert ctx["request"]["inputs"][3]["data"] == [["true"]]


@then("outputs request the result_json tensor")
def _(ctx):
    assert ctx["request"]["outputs"] == [{"name": "result_json"}]


@when(parsers.parse('run_system_one executes for model "{model}" with state {"k": 1}'))
def _(ctx, model, monkeypatch):
    router = MagicMock()
    router.allm_passthrough_route = AsyncMock(
        return_value={
            "outputs": [{"name": "result_json", "data": ['{"answers": ["a"]}']}]
        }
    )
    monkeypatch.setattr(so_gen, "get_global_router", lambda: router)
    monkeypatch.setattr(
        so_gen,
        "model_settings",
        SimpleNamespace(models={model: {"type": "system_one"}}),
    )
    ctx["router"] = router
    ctx["response"] = asyncio.run(
        so_gen.run_system_one(
            model_id=model,
            state={"k": 1},
            questions=["q"],
            response_model="so-external",
        )
    )


@then(parsers.parse('the passthrough call model is "{model}"'))
def _(ctx, model):
    assert ctx["router"].allm_passthrough_route.await_args.kwargs["model"] == model


@then(parsers.parse("the decoded response has answers {answers}"))
def _(ctx, answers):
    assert ctx["response"]["answers"] == ast.literal_eval(answers)
