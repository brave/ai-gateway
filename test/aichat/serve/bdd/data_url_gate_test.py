"""Step definitions for the data URL fail-closed gate scenarios."""

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import api_key_chat_api, passthrough_api
from aichat.serve.media_client import SandboxOpError, SandboxWorkerError
from aichat.serve.services import conversation_title
from aichat.serve.services import data_url_gate as gate
from aichat.serve.services import pdf as pdf_service

scenarios("features/data_url_gate.feature")

PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
B64 = base64.b64encode(PDF).decode()
PDF_URL = f"data:application/pdf;base64,{B64}"


@pytest.fixture
def ctx():
    return {}


@given("a pdf payload")
def pdf_payload(ctx):
    ctx["b64"] = B64
    return ctx


def _file_part(url: str, filename: str = "a.bin") -> dict:
    return {"type": "file", "file": {"filename": filename, "file_data": url}}


class _FakeAnalyzer:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def __call__(self, args):
        self.calls.append(args)
        if self.error is not None:
            raise self.error
        return self.result


@given("a user message with a canonical pdf file part")
def user_message_with_pdf(ctx, monkeypatch):
    ctx["message"] = {"role": "user", "content": [_file_part(PDF_URL, "a.pdf")]}
    ctx["analyzer"] = _FakeAnalyzer(
        result={"passthrough": None, "total_pages": 1, "estimated_tokens": 1}
    )
    patcher = patch.object(pdf_service, "call_pdf_analyze", ctx["analyzer"])
    patcher.start()
    return ctx


@given(parsers.parse("the sandbox {outcome}"))
def sandbox_outcome(ctx, outcome):
    analyzer = ctx["analyzer"]
    if outcome == "rejects the pdf":
        analyzer.error = SandboxOpError("op failed in sandbox: ValueError: bad pdf")
    elif outcome == "dies unexpectedly":
        analyzer.error = RuntimeError("boom")
    elif outcome == "worker dies":
        analyzer.error = SandboxWorkerError("worker killed")


@given("the pdf size limit is 0 MB")
def zero_size_limit(ctx, monkeypatch):
    monkeypatch.setattr(
        pdf_service.external_service_settings, "max_pdf_file_size_mb", 0
    )


@when("pdf limits are enforced")
def enforce_pdf_limits(ctx):
    try:
        ctx["result"] = asyncio.run(
            pdf_service.process_messages_for_pdf_limits([ctx["message"]], 100000)
        )
    except Exception as e:
        ctx["error"] = e


@then("the pdf part is removed")
def pdf_part_removed(ctx):
    parts = [
        p
        for m in ctx["result"]
        for p in (m.get("content") or [])
        if p.get("type") == "file"
    ]
    assert not parts, f"unprocessed PDF part survived: {parts}"


@then("a hard sandbox error is raised")
def hard_error_raised(ctx):
    assert isinstance(ctx.get("error"), SandboxWorkerError), ctx.get("error")


@then(parsers.parse("the sandbox was invoked"))
def sandbox_invoked(ctx):
    assert ctx["analyzer"].calls, "PDF skipped sandbox processing"


def _patch_inspect(monkeypatch, verdict: str):
    if verdict == "confirms a safe binary payload":

        async def inspect(data_url):
            return {"sniffed_mime": "image/png", "is_binary_safe": True}

    elif verdict == "rejects the payload":

        async def inspect(data_url):
            return {"sniffed_mime": None, "is_binary_safe": False}

    else:  # unavailable

        async def inspect(data_url):
            raise SandboxWorkerError("unreachable")

    monkeypatch.setattr(gate, "call_media_inspect", inspect)


@when(parsers.parse('the gate runs on a message with a "{data_url}" part'))
def gate_runs(ctx, data_url, monkeypatch):
    verdict = ctx.get("verdict", "rejects the payload")
    _patch_inspect(monkeypatch, verdict)
    url = f"{data_url}{ctx['b64']}"
    ctx["message"] = {"role": "user", "content": [_file_part(url)]}
    ctx["result"] = asyncio.run(
        gate.remove_unprocessed_data_url_parts([ctx["message"]])
    )
    return ctx


@given("the media processor confirms a safe binary payload")
def processor_confirms(ctx):
    ctx["verdict"] = "confirms a safe binary payload"
    return ctx


@given("the media processor rejects the payload")
def processor_rejects(ctx):
    ctx["verdict"] = "rejects the payload"
    return ctx


@given("the media processor is unavailable")
def processor_unavailable(ctx):
    ctx["verdict"] = "is unavailable"
    return ctx


@then("the part is kept")
def part_kept(ctx):
    parts = ctx["result"][0]["content"]
    assert len(parts) == 1 and parts[0].get("type") == "file", parts


@then("the part is removed")
def part_removed(ctx):
    parts = ctx["result"][0]["content"]
    assert parts == [], f"data URL part survived: {parts}"


def _fake_backend_capture():
    captured = {}

    async def fake_converse(messages, stream=False, params=None, **kw):
        captured["messages"] = messages
        return MagicMock(model_dump=MagicMock(return_value={"id": "resp-1"}))

    backend = SimpleNamespace(
        build_params=lambda stream, tools, messages, enable_prompt_caching=None: {},
        config=SimpleNamespace(upstream_model="up/test-model"),
        converse=fake_converse,
    )
    return backend, captured


def _patch_inspect_pass_through(monkeypatch):
    async def inspect(data_url):
        return {"sniffed_mime": None, "is_binary_safe": False}

    monkeypatch.setattr(gate, "call_media_inspect", inspect)


@when("an api-key chat request with one text part is handled")
def api_key_text_part(ctx, monkeypatch):
    from fastapi.requests import Request

    _patch_inspect_pass_through(monkeypatch)
    backend, captured = _fake_backend_capture()
    monkeypatch.setattr(
        api_key_chat_api, "get_backend", MagicMock(return_value=backend)
    )
    monkeypatch.setattr(
        api_key_chat_api, "run_dynamic_leo", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        api_key_chat_api,
        "select_model_for_request",
        AsyncMock(return_value="test-model"),
    )
    monkeypatch.setattr(
        api_key_chat_api, "apply_claude_upstream_sampling_params", MagicMock()
    )

    request = MagicMock(spec=Request)
    request.json = AsyncMock(
        return_value={
            "model": "test-model",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
    )
    request.headers = {"x-forwarded-for": "1.2.3.4"}
    request.state.httpx_client = None

    ctx["response"] = asyncio.run(api_key_chat_api.handle_chat_completions(request))
    ctx["backend_messages"] = captured.get("messages")
    return ctx


@given("a canonical pdf file part")
def canonical_pdf_part(ctx):
    ctx["canonical_url"] = PDF_URL
    return ctx


@when("an api-key chat request with the pdf file part is handled")
def api_key_pdf_part(ctx, monkeypatch):
    from fastapi.requests import Request

    ctx["analyzer"] = _FakeAnalyzer(
        result={"passthrough": None, "total_pages": 1, "estimated_tokens": 1}
    )
    patcher = patch.object(pdf_service, "call_pdf_analyze", ctx["analyzer"])
    patcher.start()
    _patch_inspect_pass_through(monkeypatch)
    backend, captured = _fake_backend_capture()
    monkeypatch.setattr(
        api_key_chat_api, "get_backend", MagicMock(return_value=backend)
    )
    monkeypatch.setattr(
        api_key_chat_api, "run_dynamic_leo", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        api_key_chat_api,
        "select_model_for_request",
        AsyncMock(return_value="test-model"),
    )
    monkeypatch.setattr(
        api_key_chat_api, "apply_claude_upstream_sampling_params", MagicMock()
    )

    request = MagicMock(spec=Request)
    request.json = AsyncMock(
        return_value={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": [_file_part(ctx["canonical_url"], "a.pdf")]}
            ],
        }
    )
    request.headers = {"x-forwarded-for": "1.2.3.4"}
    request.state.httpx_client = None

    ctx["response"] = asyncio.run(api_key_chat_api.handle_chat_completions(request))
    ctx["backend_messages"] = captured.get("messages")
    return ctx


@when(parsers.parse('an api-key chat request with a "{data_url}" file part is handled'))
def api_key_file_part(ctx, data_url, monkeypatch):
    from fastapi.requests import Request

    url = f"{data_url}{ctx['b64']}"
    ctx["analyzer"] = _FakeAnalyzer(
        result={"passthrough": None, "total_pages": 1, "estimated_tokens": 1}
    )
    patcher = patch.object(pdf_service, "call_pdf_analyze", ctx["analyzer"])
    patcher.start()
    _patch_inspect_pass_through(monkeypatch)
    backend, captured = _fake_backend_capture()
    monkeypatch.setattr(
        api_key_chat_api, "get_backend", MagicMock(return_value=backend)
    )
    monkeypatch.setattr(
        api_key_chat_api, "run_dynamic_leo", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        api_key_chat_api,
        "select_model_for_request",
        AsyncMock(return_value="test-model"),
    )
    monkeypatch.setattr(
        api_key_chat_api, "apply_claude_upstream_sampling_params", MagicMock()
    )

    request = MagicMock(spec=Request)
    request.json = AsyncMock(
        return_value={
            "model": "test-model",
            "messages": [{"role": "user", "content": [_file_part(url)]}],
        }
    )
    request.headers = {"x-forwarded-for": "1.2.3.4"}
    request.state.httpx_client = None

    ctx["response"] = asyncio.run(api_key_chat_api.handle_chat_completions(request))
    ctx["backend_messages"] = captured.get("messages")
    return ctx


@when(parsers.parse('a passthrough request with a "{data_url}" file part is handled'))
def passthrough_file_part(ctx, data_url, monkeypatch):
    from fastapi.requests import Request

    url = f"{data_url}{ctx['b64']}"
    _patch_inspect_pass_through(monkeypatch)
    backend, captured = _fake_backend_capture()
    monkeypatch.setattr(passthrough_api, "get_backend", MagicMock(return_value=backend))
    monkeypatch.setattr(
        passthrough_api, "require_internal_models_api_key", lambda _: None
    )

    request = MagicMock(spec=Request)
    request.json = AsyncMock(
        return_value={
            "model": "test-model",
            "messages": [{"role": "user", "content": [_file_part(url)]}],
        }
    )
    request.headers = {}

    ctx["response"] = asyncio.run(passthrough_api.v1_passthrough(request))
    ctx["backend_messages"] = captured.get("messages")
    return ctx


@then(parsers.parse("the backend received {count:d} content part"))
def backend_received_parts(ctx, count):
    assert ctx["backend_messages"] is not None, "backend.converse was not awaited"
    content = ctx["backend_messages"][0]["content"]
    parts = list(content)
    assert len(parts) == count, f"expected {count} part(s), got {len(parts)}: {parts}"


@when("a title request carries a pdf part and a title part with text")
def title_request_with_pdf(ctx, monkeypatch):
    from aichat.protocol.open_ai_protocol import Request as OpenAIRequest

    backend, captured = _fake_backend_capture()
    monkeypatch.setattr(
        conversation_title, "get_backend", MagicMock(return_value=backend)
    )
    from test.aichat.serve.bdd.helpers import dataclass_model_config

    monkeypatch.setattr(
        conversation_title.model_settings,
        "model_triaging",
        {"conversation_title": "test-model"},
    )
    monkeypatch.setattr(
        conversation_title.model_settings,
        "models",
        {"test-model": dataclass_model_config()},
    )

    request = OpenAIRequest.model_validate(
        {
            "model": "test-model",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _file_part(PDF_URL, "a.pdf"),
                        {"type": "brave-conversation-title", "text": "Summarize"},
                    ],
                }
            ],
        }
    )
    ctx["response"] = asyncio.run(
        conversation_title.complete_conversation_title_chat(
            request=request,
            prompts=None,
            process_streaming_response=None,
        )
    )
    ctx["backend_messages"] = captured.get("messages")
    return ctx


@then("the title model saw only the synthetic transcript")
def title_saw_transcript(ctx):
    msgs = ctx["backend_messages"]
    assert len(msgs) == 1, msgs
    assert msgs[0]["content"] is not None
    for p in msgs[0]["content"] if isinstance(msgs[0]["content"], list) else []:
        assert p.get("type") != "file", "PDF forwarded to title model"
    assert "Summarize" in str(msgs[0]["content"])
