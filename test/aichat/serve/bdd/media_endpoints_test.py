"""BDD bindings for media API endpoints (STT/TTS/embeddings/images/systemone).

Duplicated-coverage TODO map (unit tests already cover some scenarios):
# TODO: remove stt_api_test.py#test_success_json test/aichat/serve/stt_api_test.py:96
# TODO: remove tts_api_test.py#test_successful_speech_generation test/aichat/serve/tts_api_test.py:28
# TODO: remove tts_api_test.py#test_speech_generation_with_format test/aichat/serve/tts_api_test.py:53
# TODO: remove tts_api_test.py#test_streaming_speech_returns_streaming_response test/aichat/serve/tts_api_test.py:195
# TODO: remove tts_api_test.py#test_different_audio_formats test/aichat/serve/tts_api_test.py:223
# TODO: remove embeddings_api_test.py#test_missing_model_parameter test/aichat/serve/embeddings_api_test.py:59
# TODO: remove embeddings_api_test.py#test_missing_input_parameter test/aichat/serve/embeddings_api_test.py:73
# TODO: remove embeddings_api_test.py#test_model_not_found test/aichat/serve/embeddings_api_test.py:87
# TODO: remove embeddings_api_test.py#test_invalid_json test/aichat/serve/embeddings_api_test.py:105
# TODO: remove embeddings_api_test.py#test_embeddings_generation_error test/aichat/serve/embeddings_api_test.py:117
# TODO: remove image_generation_api_test.py#test_missing_model_parameter test/aichat/serve/image_generation_api_test.py:54
# TODO: remove image_generation_api_test.py#test_model_not_found test/aichat/serve/image_generation_api_test.py:68
# TODO: remove image_generation_api_test.py#test_invalid_json test/aichat/serve/image_generation_api_test.py:86
# TODO: remove image_generation_api_test.py#test_image_generation_error test/aichat/serve/image_generation_api_test.py:98
# TODO: remove system_one_api_test.py#test_successful_systemone test/aichat/serve/system_one_api_test.py:23
# TODO: remove system_one_api_test.py#test_missing_state test/aichat/serve/system_one_api_test.py:54
# TODO: remove system_one_api_test.py#test_wrong_model_type test/aichat/serve/system_one_api_test.py:68
"""

import asyncio
import io
import json
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import (
    embeddings_api,
    image_generation_api,
    stt_api,
    system_one_api,
    tts_api,
)
from aichat.serve.services.security_settings import security_settings

FEATURE = "features/media_endpoints.feature"

scenarios(FEATURE)


@pytest.fixture
def ctx():
    return {}


@pytest.fixture(autouse=True)
def _unset_internal_models_key(monkeypatch):
    """Belt: the 401 scenarios set this key themselves; every other scenario
    must not inherit a key from the surrounding CI environment."""
    monkeypatch.setattr(security_settings, "internal_models_api_key", "")


@given("the media test harness")
def _(ctx):
    return ctx


def _tiny_wav_16k_mono_int16() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes((0).to_bytes(2, byteorder="little", signed=True) * 80)
    return buf.getvalue()


def _wav_file(mock_ctx, content=None):
    f = MagicMock()
    f.filename = "clip.wav"
    f.read = AsyncMock(side_effect=[content or _tiny_wav_16k_mono_int16(), b""])
    mock_ctx["file"] = f
    return f


def _req(mock_ctx, json_body=None, headers=None, json_error=None):
    request = MagicMock(spec=Request)
    request.headers = headers if headers is not None else {}
    if json_error is not None:
        request.json = AsyncMock(side_effect=json_error)
    else:
        request.json = AsyncMock(return_value=json_body)
    mock_ctx["request"] = request
    return request


def _stt_request(mock_ctx, model=None, response_format=None, headers=None):
    _req(mock_ctx, headers=headers)
    _wav_file(mock_ctx)
    kwargs = {
        "file": mock_ctx["file"],
        "model": model,
        "language": None,
        "prompt": None,
        "temperature": 0.0,
        "response_format": response_format,
    }
    mock_ctx["kwargs"] = kwargs


@then(parsers.parse('the response is a {status:d} error with "{message}"'))
def _(ctx, status, message):
    resp = ctx["response"]
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == status
    body = json.loads(resp.body.decode())
    actual = body["error"]["message"]
    assert message in actual, actual


@then("the response is a 401 error")
def _(ctx):
    assert isinstance(ctx["response"], JSONResponse)
    assert ctx["response"].status_code == 401


# --- STT ---


@when("a transcription request arrives without credentials")
def _(ctx):
    _stt_request(ctx, model="parakeet", headers={})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            security_settings, "internal_models_api_key", "secret", raising=False
        )
        ctx["response"] = asyncio.run(
            stt_api.v1_audio_transcriptions(ctx["request"], **ctx["kwargs"])
        )


@when("a transcription request has no model")
def _(ctx):
    _stt_request(ctx, model=None)
    ctx["response"] = asyncio.run(
        stt_api.v1_audio_transcriptions(ctx["request"], **ctx["kwargs"])
    )


@when(
    parsers.parse('a transcription request names the model "{model}" of type "{type}"')
)
def _(ctx, model, type, monkeypatch):
    _stt_request(ctx, model=model)
    if model == "parakeet":
        models = {model: {"type": type}}
    else:
        models = {"parakeet": {"type": "speech_to_text"}}
    monkeypatch.setattr(stt_api, "model_settings", SimpleNamespace(models=models))
    ctx["response"] = asyncio.run(
        stt_api.v1_audio_transcriptions(ctx["request"], **ctx["kwargs"])
    )


@when("reading the transcription upload raises a generic error")
def _(ctx, monkeypatch):
    request = MagicMock(spec=Request)
    request.headers = {}
    ctx["request"] = request
    f = MagicMock()
    f.filename = "clip.wav"
    f.read = AsyncMock(side_effect=RuntimeError("disk on fire"))
    ctx["file"] = f
    ctx["kwargs"] = {
        "file": f,
        "model": "parakeet",
        "language": None,
        "prompt": None,
        "response_format": None,
        "temperature": None,
    }
    monkeypatch.setattr(
        stt_api,
        "model_settings",
        SimpleNamespace(models={"parakeet": {"type": "speech_to_text"}}),
    )
    ctx["response"] = asyncio.run(
        stt_api.v1_audio_transcriptions(request, **ctx["kwargs"])
    )


@when(parsers.parse('a transcription request asks for response_format "{fmt}"'))
def _(ctx, fmt, monkeypatch):
    _stt_request(ctx, model="parakeet", response_format=fmt)
    monkeypatch.setattr(
        stt_api,
        "model_settings",
        SimpleNamespace(models={"parakeet": {"type": "speech_to_text"}}),
    )
    ctx["response"] = asyncio.run(
        stt_api.v1_audio_transcriptions(ctx["request"], **ctx["kwargs"])
    )


def _raise_of(exc: str, text: str) -> Exception:
    """Exception factory that fails loudly on a typo'd feature value."""
    types = {"RuntimeError": RuntimeError, "ValueError": ValueError}
    if exc not in types:
        raise AssertionError(f"unknown exception type in feature: {exc}")
    return types[exc](text)


@when(parsers.parse('the transcription backend raises {exc} "{text}"'))
def _(ctx, exc, text, monkeypatch):
    _stt_request(ctx, model="parakeet")
    monkeypatch.setattr(
        stt_api,
        "model_settings",
        SimpleNamespace(models={"parakeet": {"type": "speech_to_text"}}),
    )
    tx = AsyncMock(side_effect=_raise_of(exc, text))
    monkeypatch.setattr(stt_api, "transcribe_upload", tx)
    ctx["response"] = asyncio.run(
        stt_api.v1_audio_transcriptions(ctx["request"], **ctx["kwargs"])
    )


@when(parsers.parse('a transcription succeeds and asks for response_format "{fmt}"'))
def _(ctx, fmt, monkeypatch):
    _stt_request(ctx, model="parakeet", response_format=fmt)
    monkeypatch.setattr(
        stt_api,
        "model_settings",
        SimpleNamespace(models={"parakeet": {"type": "speech_to_text"}}),
    )
    monkeypatch.setattr(
        stt_api, "transcribe_upload", AsyncMock(return_value=("hello world", 4000.0))
    )
    ctx["response"] = asyncio.run(
        stt_api.v1_audio_transcriptions(ctx["request"], **ctx["kwargs"])
    )
    ctx["fmt"] = fmt


@then(
    parsers.parse(
        'the transcription response has media type "{media_type}" and matches the "{fmt}" rendering'
    )
)
def _(ctx, media_type, fmt):
    resp = ctx["response"]
    if fmt == "verbose_json":
        assert isinstance(resp, JSONResponse)
        assert resp.media_type == media_type
        assert json.loads(resp.body.decode()) == {
            "task": "transcribe",
            "language": "en",
            "duration": 4.0,
            "text": "hello world",
            "words": [],
        }
    else:
        assert isinstance(resp, Response)
        assert resp.media_type == media_type
        body = resp.body.decode()
        if fmt == "text":
            assert body == "hello world"
        elif fmt == "srt":
            assert body == "1\n00:00:00,000 --> 00:00:04,000\nhello world\n"
        elif fmt == "vtt":
            assert body == "WEBVTT\n\n00:00:00.000 --> 00:00:04.000\nhello world\n"
        else:
            raise AssertionError(f"unknown transcription format: {fmt}")


@then(parsers.parse('the transcription JSON body has text "{text}"'))
def _(ctx, text):
    resp = ctx["response"]
    assert isinstance(resp, JSONResponse)
    assert json.loads(resp.body.decode()) == {"text": text}


# --- TTS ---


@then(parsers.parse('the speech audio is returned as "{content_type}"'))
def _(ctx, content_type):
    resp = ctx["response"]
    # PROD CANDIDATE BUG: wav is not in CONTENT_TYPE_MAP, so a wav request
    # falls back to the mp3 default. Current behavior pinned deliberately;
    # fix belongs in aichat/serve/tts_api.py, not here.
    assert isinstance(resp, Response)
    assert resp.media_type == content_type


def _tts_body(ctx, fmt=None, stream=None):
    body = {"model": "test-tts-model", "input": "Hello there"}
    if fmt and fmt != "none":
        body["response_format"] = fmt
    if stream is not None:
        body["stream"] = stream
    ctx["body"] = body


@when(parsers.parse('a speech request asks for response_format "{fmt}"'))
def _(ctx, fmt, monkeypatch):
    request = MagicMock(spec=Request)
    request.headers = {}
    _tts_body(ctx, fmt=fmt)
    request.json = AsyncMock(return_value=ctx["body"])
    monkeypatch.setattr(
        tts_api,
        "model_settings",
        SimpleNamespace(models={"test-tts-model": {"type": "tts"}}),
    )
    monkeypatch.setattr(
        tts_api,
        "generate_speech",
        AsyncMock(return_value=MagicMock(content=b"audio-bytes")),
    )
    ctx["response"] = asyncio.run(tts_api.v1_audio_speech(request))


@when("the speech generator returns an object without audio content")
def _(ctx, monkeypatch):
    request = MagicMock(spec=Request)
    request.headers = {}
    _tts_body(ctx)
    request.json = AsyncMock(return_value=ctx["body"])
    monkeypatch.setattr(
        tts_api,
        "model_settings",
        SimpleNamespace(models={"test-tts-model": {"type": "tts"}}),
    )
    monkeypatch.setattr(
        tts_api, "generate_speech", AsyncMock(return_value=SimpleNamespace())
    )
    ctx["response"] = asyncio.run(tts_api.v1_audio_speech(request))


@when("a speech request sets stream to true")
def _(ctx, monkeypatch):
    request = MagicMock(spec=Request)
    request.headers = {}
    _tts_body(ctx, stream=True)
    request.json = AsyncMock(return_value=ctx["body"])
    monkeypatch.setattr(
        tts_api,
        "model_settings",
        SimpleNamespace(models={"test-tts-model": {"type": "tts"}}),
    )

    async def _gen():
        yield b"a"
        yield b"b"

    monkeypatch.setattr(tts_api, "stream_speech", AsyncMock(return_value=_gen()))
    ctx["response"] = asyncio.run(tts_api.v1_audio_speech(request))


@then(
    parsers.parse('the response is a streaming audio response of type "{content_type}"')
)
def _(ctx, content_type):
    resp = ctx["response"]
    assert type(resp).__name__ == "StreamingResponse"
    assert resp.media_type == content_type


# --- Embeddings / images shared steps ---


def _json_api_request(ctx, body, json_error=None):
    request = MagicMock(spec=Request)
    request.headers = {}
    if json_error is not None:
        request.json = AsyncMock(side_effect=json_error)
    else:
        request.json = AsyncMock(return_value=body)
    ctx["request"] = request


@when(parsers.parse("an embeddings request {defect} is submitted"))
def _(ctx, defect, monkeypatch):
    body = {"model": "text-embedding", "input": "hello"}
    json_error = None
    model_type = "embedding"
    if defect == "with no model":
        body = {"input": "hello"}
    elif defect == "with no input":
        body = {"model": "text-embedding"}
    elif defect == "naming an unknown model":
        body = {"model": "nope", "input": "hello"}
    elif defect == "naming a non-embedding model":
        model_type = "llm"
    elif defect == "with invalid JSON":
        json_error = ValueError("bad json")
    else:
        raise AssertionError(f"unknown embeddings defect in feature: {defect}")
    monkeypatch.setattr(
        embeddings_api,
        "model_settings",
        SimpleNamespace(models={"text-embedding": {"type": model_type}}),
    )
    ctx["gen"] = AsyncMock(
        return_value={
            "object": "list",
            "data": [],
            "model": "text-embedding",
            "usage": {},
        }
    )
    monkeypatch.setattr(embeddings_api, "generate_embeddings", ctx["gen"])
    _json_api_request(ctx, body, json_error=json_error)
    ctx["response"] = asyncio.run(embeddings_api.v1_embeddings(ctx["request"]))


@when(parsers.parse('the embeddings generator raises {exc} "{text}"'))
def _(ctx, exc, text, monkeypatch):
    monkeypatch.setattr(
        embeddings_api,
        "model_settings",
        SimpleNamespace(models={"text-embedding": {"type": "embedding"}}),
    )
    side_effect = _raise_of(exc, text)
    monkeypatch.setattr(
        embeddings_api, "generate_embeddings", AsyncMock(side_effect=side_effect)
    )
    _json_api_request(ctx, {"model": "text-embedding", "input": "hello"})
    ctx["response"] = asyncio.run(embeddings_api.v1_embeddings(ctx["request"]))


@when(parsers.parse("an image generation request {defect} is submitted"))
def _(ctx, defect, monkeypatch):
    body = {"model": "image-model", "prompt": "a cat"}
    json_error = None
    if defect == "with no model":
        body = {"prompt": "a cat"}
    elif defect == "naming an unknown model":
        body = {"model": "nope", "prompt": "a cat"}
    elif defect == "with invalid JSON":
        json_error = ValueError("bad json")
    else:
        raise AssertionError(f"unknown image defect in feature: {defect}")
    monkeypatch.setattr(
        image_generation_api,
        "model_settings",
        SimpleNamespace(models={"image-model": {"type": "image_gen"}}),
    )
    monkeypatch.setattr(
        image_generation_api,
        "generate_image",
        AsyncMock(return_value=MagicMock(model_dump=lambda: {"data": []})),
    )
    _json_api_request(ctx, body, json_error=json_error)
    ctx["response"] = asyncio.run(
        image_generation_api.v1_images_generations(ctx["request"])
    )


# --- Image generator ---


@when(parsers.parse('the image generator raises {exc} "{text}"'))
def _(ctx, exc, text, monkeypatch):
    monkeypatch.setattr(
        image_generation_api,
        "model_settings",
        SimpleNamespace(models={"image-model": {"type": "image_gen"}}),
    )
    side_effect = _raise_of(exc, text)
    monkeypatch.setattr(
        image_generation_api,
        "generate_image",
        AsyncMock(side_effect=side_effect),
    )
    _json_api_request(ctx, {"model": "image-model", "prompt": "a cat"})
    ctx["response"] = asyncio.run(
        image_generation_api.v1_images_generations(ctx["request"])
    )


# --- SystemOne ---


def _so_models(so_type="system_one"):
    return {"so-model": {"type": so_type, "response_model": "so-external"}}


@when("a systemone request arrives without credentials")
def _(ctx, monkeypatch):
    monkeypatch.setattr(
        system_one_api, "model_settings", SimpleNamespace(models=_so_models())
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            security_settings, "internal_models_api_key", "secret", raising=False
        )
        request = MagicMock(spec=Request)
        request.headers = {}
        request.json = AsyncMock(
            return_value={"model": "so-model", "state": {}, "questions": []}
        )
        ctx["request"] = request
        ctx["response"] = asyncio.run(system_one_api.v1_systemone(request))


@when(parsers.parse("a systemone request {defect} is submitted"))
def _(ctx, defect, monkeypatch):
    body = {"model": "so-model", "state": {"k": 1}, "questions": ["q"]}
    json_error = None
    so_type = "system_one"
    if defect == "with invalid JSON":
        json_error = ValueError("bad json")
        body = None
    elif defect == "that is not a JSON object":
        body = ["not", "a", "dict"]
    elif defect == "with no model":
        body = {"state": {}, "questions": []}
    elif defect == "naming an unknown model":
        body = {"model": "nope", "state": {}, "questions": []}
    elif defect == "naming a non-systemone model":
        so_type = "llm"
    elif defect == "with no state":
        body = {"model": "so-model", "questions": []}
    elif defect == "with no questions":
        body = {"model": "so-model", "state": {}}
    elif defect == "with an empty model_override":
        body["model_override"] = "   "
    elif defect == "with a non-boolean route_only":
        body["route_only"] = "yes"
    elif defect == "with a non-boolean include_routing":
        body["include_routing"] = "yes"
    else:
        raise AssertionError(f"unknown systemone defect in feature: {defect}")
    monkeypatch.setattr(
        system_one_api, "model_settings", SimpleNamespace(models=_so_models(so_type))
    )
    ctx["run"] = AsyncMock(return_value={"model": "so-external", "answers": ["a"]})
    monkeypatch.setattr(system_one_api, "run_system_one", ctx["run"])
    _json_api_request(ctx, body, json_error=json_error)
    ctx["response"] = asyncio.run(system_one_api.v1_systemone(ctx["request"]))


@when(
    parsers.parse(
        'a systemone request carries model_override "{override}" and route flags'
    )
)
def _(ctx, override, monkeypatch):
    body = {
        "model": "so-model",
        "state": {"k": 1},
        "questions": ["q"],
        "model_override": override,
        "route_only": True,
        "include_routing": True,
    }
    monkeypatch.setattr(
        system_one_api, "model_settings", SimpleNamespace(models=_so_models())
    )
    ctx["run"] = AsyncMock(return_value={"model": "so-external", "answers": ["a"]})
    monkeypatch.setattr(system_one_api, "run_system_one", ctx["run"])
    _json_api_request(ctx, body)
    ctx["response"] = asyncio.run(system_one_api.v1_systemone(ctx["request"]))


@then(parsers.parse('the systemone response contains model "{model}"'))
def _(ctx, model):
    assert json.loads(ctx["response"].body.decode())["model"] == model


@then(
    parsers.parse(
        'run_system_one received model_override "{override}" route_only {route_only} include_routing {include_routing}'
    )
)
def _(ctx, override, route_only, include_routing):
    kwargs = ctx["run"].await_args.kwargs
    assert kwargs["model_override"] == override
    assert kwargs["route_only"] == (route_only == "True")
    assert kwargs["include_routing"] == (include_routing == "True")


@when(parsers.parse('the systemone backend raises {exc} "{text}"'))
def _(ctx, exc, text, monkeypatch):
    monkeypatch.setattr(
        system_one_api, "model_settings", SimpleNamespace(models=_so_models())
    )
    side_effect = _raise_of(exc, text)
    monkeypatch.setattr(
        system_one_api, "run_system_one", AsyncMock(side_effect=side_effect)
    )
    _json_api_request(ctx, {"model": "so-model", "state": {}, "questions": []})
    ctx["response"] = asyncio.run(system_one_api.v1_systemone(ctx["request"]))
