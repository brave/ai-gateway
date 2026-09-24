# BDD: deprecated /v1/conversation notice + /v1/passthrough forwarding.
# Duplication with unit tests:
# TODO: remove test/aichat/serve/conversation_api_test.py#test_defaults_to_english_when_no_system_language test/aichat/serve/conversation_api_test.py:16
# TODO: remove test/aichat/serve/conversation_api_test.py#test_localizes_known_language_and_still_includes_english test/aichat/serve/conversation_api_test.py:28
# TODO: remove test/aichat/serve/conversation_api_test.py#test_non_streaming_request_returns_static_notice test/aichat/serve/conversation_api_test.py:48
# TODO: remove test/aichat/serve/conversation_api_test.py#test_streaming_request_returns_single_sse_event test/aichat/serve/conversation_api_test.py:66
# TODO: remove test/aichat/serve/conversation_api_test.py#test_does_not_require_auth_headers_or_credentials test/aichat/serve/conversation_api_test.py:86
# TODO: remove test/aichat/serve/passthrough_api_test.py#test_invalid_body_raises_400 test/aichat/serve/passthrough_api_test.py:20
# TODO: remove test/aichat/serve/passthrough_api_test.py#test_unknown_backend_raises_400 test/aichat/serve/passthrough_api_test.py:27
# TODO: remove test/aichat/serve/passthrough_api_test.py#test_success_non_stream test/aichat/serve/passthrough_api_test.py:43
# TODO: remove test/aichat/serve/passthrough_api_test.py#test_error_dict_raises_http test/aichat/serve/passthrough_api_test.py:61
# TODO: remove test/aichat/serve/passthrough_api_test.py#test_streaming test/aichat/serve/passthrough_api_test.py:80

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, parsers, scenarios, then, when
from starlette.requests import Request

from aichat.serve import passthrough_api
from aichat.serve.api_server import app
from aichat.serve.constants import CONVERSATION_API_DEPRECATION_MESSAGES
from aichat.serve.conversation_api import get_conversation_deprecation_message

FEATURE = Path(__file__).parent / "features" / "conversation_passthrough.feature"
scenarios(str(FEATURE))


@pytest.fixture
def ctx():
    return {}


def _mock_request(body):
    request = MagicMock(spec=Request)
    request.json = AsyncMock(return_value=body)
    return request


def _base_body():
    return {
        "model": "mixtral-8x7b-instruct",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }


def _stub_backend(monkeypatch, ctx, converse_return):
    backend = MagicMock()
    backend.build_params.return_value = {"temperature": 0.9, "top_p": 0.9}
    backend.config.upstream_model = "casperhansen/mixtral-instruct-awq"
    backend.converse = AsyncMock(return_value=converse_return)
    get_backend = MagicMock(return_value=backend)
    monkeypatch.setattr(passthrough_api, "get_backend", get_backend)
    apply_sampling = MagicMock()
    monkeypatch.setattr(
        passthrough_api, "apply_claude_upstream_sampling_params", apply_sampling
    )
    ctx["backend"] = backend
    ctx["apply_sampling"] = apply_sampling
    return get_backend


# --- conversation deprecation notice -------------------------------------------


@given(parsers.parse('a conversation client with system language "{system_language}"'))
def given_conversation_language(system_language, ctx):
    ctx["system_language"] = None if system_language == "none" else system_language
    ctx["client"] = None


@given("a conversation client without credentials")
def given_conversation_client(ctx):
    ctx["test_client"] = TestClient(app)


@when("the deprecation notice is built")
def when_build_notice(ctx):
    ctx["notice"] = get_conversation_deprecation_message(ctx["system_language"])


@then(parsers.parse('the notice equals "{expected}"'))
def then_notice(expected, ctx):
    en = CONVERSATION_API_DEPRECATION_MESSAGES["en"]
    if expected == "en_only":
        assert ctx["notice"] == en
    elif expected == "fr_then_en":
        assert ctx["notice"] == (
            f"{CONVERSATION_API_DEPRECATION_MESSAGES['fr']}\n\n{en}"
        )
    else:
        pytest.fail(f"unknown expectation {expected}")


@when(
    parsers.parse(
        'a non-streaming conversation request is sent for language "{language}"'
    )
)
def when_conversation_non_streaming(language, ctx):
    ctx["response"] = ctx["test_client"].post(
        "/v1/conversation",
        json={
            "model": "automatic",
            "system_language": language,
            "stream": False,
            "events": [{"role": "user", "type": "chatMessage", "content": "hello"}],
        },
        headers={},
    )


@then(
    parsers.parse('the conversation response is a completion notice for "{language}"')
)
def then_conversation_notice(language, ctx):
    response = ctx["response"]
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "completion"
    assert body["model"] == "automatic"
    assert body["stop_reason"] == "stop_sequence"
    assert body["completion"] == (
        f"{CONVERSATION_API_DEPRECATION_MESSAGES[language]}\n\n"
        f"{CONVERSATION_API_DEPRECATION_MESSAGES['en']}"
    )


@when(parsers.parse('a streaming conversation request is sent for model "{model}"'))
def when_conversation_streaming(model, ctx):
    ctx["response"] = ctx["test_client"].post(
        "/v1/conversation",
        json={
            "model": model,
            "stream": True,
            "events": [{"role": "user", "type": "chatMessage", "content": "hello"}],
        },
    )


@then("the conversation stream ends with a DONE sentinel")
def then_conversation_done(ctx):
    raw_blocks = [b for b in ctx["response"].text.split("\n\n") if b]
    assert raw_blocks[-1] == "data: [DONE]"


@then("the first conversation event is an English completion notice")
def then_conversation_event(ctx):
    raw_blocks = [b for b in ctx["response"].text.split("\n\n") if b]
    data = json.loads(raw_blocks[0][len("data: ") :])
    assert data["type"] == "completion"
    assert data["stop_reason"] == "stop_sequence"
    assert data["completion"] == CONVERSATION_API_DEPRECATION_MESSAGES["en"]


# --- passthrough ------------------------------------------------------------------


@given(parsers.parse("a passthrough request with {problem}"))
def given_passthrough_problem(problem, monkeypatch, ctx):
    problem = problem.replace(" ", "_")
    if problem == "a_non-object_body":
        ctx["body"] = {"model": 1}
        ctx["get_backend"] = None
    elif problem == "a_non-bool_prompt_caching":
        ctx["body"] = {
            "model": "mixtral-8x7b-instruct",
            "messages": [{"role": "user", "content": "hello"}],
            "prompt_caching": "yes",
        }
        ctx["get_backend"] = None
    else:  # an_unknown_backend
        ctx["body"] = {
            "model": "unknown-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        get_backend = MagicMock(side_effect=ValueError("Unsupported backend"))
        monkeypatch.setattr(passthrough_api, "get_backend", get_backend)
        ctx["get_backend"] = get_backend


@given(parsers.parse('a passthrough request for model "{model}" without streaming'))
def given_passthrough_non_streaming(model, ctx):
    ctx["body"] = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }


@given(parsers.parse('a streaming passthrough request for model "{model}"'))
def given_passthrough_streaming(model, ctx):
    ctx["body"] = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }


@given(parsers.parse("a prompt caching flag of {flag}"))
def given_prompt_caching(flag, ctx):
    if flag != "None":
        ctx["body"]["prompt_caching"] = flag == "True"


@given(parsers.parse('extra parameters "{params}"'))
def given_extra_params(params, ctx):
    for item in params.split(", "):
        key, value = item.split("=")
        if key == "max_tokens":
            ctx["body"][key] = int(value)
        else:
            ctx["body"][key] = float(value)


@given(parsers.parse('a backend error with code "{code}"'))
def given_backend_error(code, monkeypatch, ctx):
    error = {"type": "error", "code": int(code), "content": "backend down"}
    ctx["get_backend"] = _stub_backend(monkeypatch, ctx, error)


@when("the passthrough request is processed")
def when_passthrough_processed(ctx, monkeypatch):
    if not ctx.get("backend") and ctx.get("get_backend") is None:
        completion = litellm.ModelResponse(id="resp-1", model="mixtral-8x7b-instruct")
        ctx["get_backend"] = _stub_backend(monkeypatch, ctx, completion)

    request = _mock_request(ctx["body"])

    def execute():
        try:
            ctx["result"] = asyncio.run(passthrough_api.v1_passthrough(request))
            ctx["error"] = None
        except Exception as exc:
            ctx["error"] = exc

    execute()


@then(parsers.parse("a passthrough HTTP error with status {status:d} is raised"))
def then_passthrough_http_error(status, ctx):
    error = ctx["error"]
    assert error is not None
    assert error.status_code == status


@then("the backend received the prompt caching flag")
def then_prompt_caching_forwarded(ctx):
    kwargs = ctx["backend"].build_params.call_args.kwargs
    assert kwargs["enable_prompt_caching"] is False


@then("the passthrough response is the backend completion")
def then_passthrough_completion(ctx):
    assert ctx["result"] is ctx["backend"].converse.return_value


@then("claude sampling params were applied to the backend params")
def then_sampling_applied(ctx):
    ctx["apply_sampling"].assert_called_once()
    args = ctx["apply_sampling"].call_args.args
    assert args[0] == "casperhansen/mixtral-instruct-awq"
    params = args[1]
    # temperature was explicitly supplied, so it survives the pruning step
    assert params["temperature"] == 0.5


@given(parsers.parse("a backend streaming {description}"))
def given_backend_stream(description, monkeypatch, ctx):
    if description == "two serializable chunks":
        chunks = [
            litellm.ModelResponse(id="c1", model="m"),
            litellm.ModelResponse(id="c2", model="m"),
        ]
    else:  # one unserializable chunk and one serializable chunk
        bad = MagicMock()
        bad.model_dump_json = MagicMock(side_effect=Exception("nope"))
        chunks = [bad, litellm.ModelResponse(id="ok", model="m")]
    ctx["stream_chunks"] = chunks

    async def stream_gen():
        for chunk in chunks:
            yield chunk

    ctx["get_backend"] = _stub_backend(monkeypatch, ctx, stream_gen())


@when("the passthrough stream is consumed")
def when_consume_passthrough_stream(ctx):
    result = ctx["result"]

    async def collect():
        return [event async for event in result.body_iterator]

    ctx["events"] = asyncio.run(collect())


@then(parsers.parse("the stream contains {description}"))
def then_stream_contains(description, ctx):
    joined = "".join(ctx["events"])
    if description == "both chunk payloads":
        assert "c1" in joined and "c2" in joined
    else:  # only the serializable chunk payload
        assert "ok" in joined and "c1" not in joined


@then("the stream ends with a DONE sentinel")
def then_passthrough_done(ctx):
    assert ctx["events"][-1] == "data: [DONE]\n\n"
