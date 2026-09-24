# BDD coverage for aichat/serve/services/pdf.py.
#
# TODO: remove test/aichat/serve/services/pdf_test.py#test_is_pdf_file_data test/aichat/serve/services/pdf_test.py:31
# TODO: remove test/aichat/serve/services/pdf_test.py#test_decode_pdf_bytes test/aichat/serve/services/pdf_test.py:46
# TODO: remove test/aichat/serve/services/pdf_test.py#test_compute_max_extraction_tokens test/aichat/serve/services/pdf_test.py:54
# TODO: remove test/aichat/serve/services/pdf_test.py#test_process_pdf_text_content_parts_sync test/aichat/serve/services/pdf_test.py:70
# TODO: remove test/aichat/serve/services/pdf_test.py#test_process_pdf_text_content_parts test/aichat/serve/services/pdf_test.py:140
# TODO: remove test/aichat/serve/services/pdf_test.py#test_process_messages_for_pdf_limits test/aichat/serve/services/pdf_test.py:171

import asyncio
import base64
from unittest.mock import AsyncMock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.services import pdf as pdf_service
from aichat.serve.services.media_sandbox import SandboxOpError, SandboxWorkerError

FEATURE = "features/pdf_service.feature"
scenarios(FEATURE)

PDF_PREFIX = "data:application/pdf;base64,"


@pytest.fixture
def ctx():
    return {}


class _FakePool:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def call(self, op, args):
        self.calls.append((op, args))
        if self.error is not None:
            raise self.error
        return self.result


def _pdf_data_url(payload: bytes = b"%PDF-fake") -> str:
    return f"{PDF_PREFIX}{base64.b64encode(payload).decode()}"


@given("the pdf service harness")
def pdf_harness(ctx, monkeypatch):
    ctx["pdf_part"] = {
        "type": "file",
        "file": {"file_data": _pdf_data_url(b"%PDF-fake")},
    }
    ctx["file_info"] = ctx["pdf_part"]["file"]
    ctx["pool"] = _FakePool()
    monkeypatch.setattr(pdf_service, "get_pool", lambda: ctx["pool"])
    return ctx


@when(parsers.parse("the file data {data_desc} is inspected"))
def inspect_file_data(ctx, data_desc):
    if data_desc == "is a pdf data url":
        data = f"{PDF_PREFIX}{base64.b64encode(b'%PDF').decode()}"
    elif data_desc == "is a png data url":
        data = "data:image/png;base64,AAAA"
    else:
        data = "plain text"
    ctx["detection"] = pdf_service._is_pdf_file_data(data)


@then(parsers.parse("the pdf detection result is {expected}"))
def pdf_detection_result(ctx, expected):
    assert ctx["detection"] == (expected == "True")


@when("pdf bytes are encoded to a data url and decoded back")
def pdf_roundtrip(ctx):
    original = b"%PDF-roundtrip-bytes"
    url = pdf_service._encode_pdf_to_data_url(original)
    ctx["decoded"] = pdf_service._decode_pdf_bytes(url)


@then("the decoded bytes match the original")
def decoded_bytes_match(ctx):
    assert ctx["decoded"] == b"%PDF-roundtrip-bytes"


@when(parsers.parse("the conversation token limit is {limit}"))
def conversation_limit(ctx, limit):
    value = None if limit == "none" else int(limit)
    ctx["budget"] = pdf_service._compute_max_extraction_tokens(value)


@then(parsers.parse("the max extraction tokens are {tokens:d}"))
def max_extraction_tokens(ctx, tokens):
    assert ctx["budget"] == tokens


@when(parsers.parse("pdf bytes are analyzed with budget {budget:d}"))
def analyze_pdf(ctx, budget):
    ctx["analysis"] = asyncio.run(pdf_service._analyze_pdf(b"%PDF-analyzed", budget))


@then(
    parsers.parse(
        'the sandbox receives op "{op}" with encoding "{encoding}"'
        " max_allowed_pages {pages:d}"
    )
)
def sandbox_payload_assert(ctx, op, encoding, pages):
    calls = ctx["pool"].calls
    assert len(calls) == 1
    got_op, args = calls[0]
    assert got_op == op
    assert args["pdf_b64"] == base64.b64encode(b"%PDF-analyzed").decode("ascii")
    assert args["encoding_name"] == encoding
    assert args["max_extraction_tokens"] == 4800
    assert args["max_allowed_pages"] == pages


@when("the sandbox worker dies during pdf analysis")
def worker_dies_pdf_analysis(ctx, monkeypatch):
    ctx["pool"].error = SandboxWorkerError("worker died")
    with pytest.raises(SandboxWorkerError) as exc:
        asyncio.run(pdf_service._analyze_pdf(b"%PDF-analyzed", 4800))
    ctx["error"] = exc.value


@then("the analysis raises SandboxWorkerError")
def analysis_raises_worker_error(ctx):
    assert isinstance(ctx["error"], SandboxWorkerError)


ANALYSES = {
    "encrypted passthrough": {
        "passthrough": "encrypted",
        "total_pages": 3,
    },
    "all_text with content": {
        "passthrough": None,
        "total_pages": 90,
        "all_text": "hello pdf body",
    },
    "all_text without content": {
        "passthrough": None,
        "total_pages": 90,
        "all_text": "   ",
    },
    "under budget without native limit": {
        "passthrough": None,
        "total_pages": 2,
        "estimated_tokens": 10,
    },
    "native limit with overflow text": {
        "passthrough": None,
        "total_pages": 3,
        "estimated_tokens": 5000,
        "native_page_limit": 2,
        "truncated_pdf_b64": base64.b64encode(b"%PDF-truncated").decode(),
        "overflow_text": "overflow body",
    },
    "native limit without overflow": {
        "passthrough": None,
        "total_pages": 3,
        "estimated_tokens": 5000,
        "native_page_limit": 2,
        "truncated_pdf_b64": base64.b64encode(b"%PDF-truncated").decode(),
        "overflow_text": "",
    },
}


@given(parsers.parse("a pdf analysis {analysis_desc}"))
def a_pdf_analysis(ctx, analysis_desc):
    ctx["analysis"] = dict(ANALYSES[analysis_desc])


@when("the analysis result is assembled")
def assemble_analysis(ctx):
    result = []
    pdf_service.process_pdf_text_content_parts_sync(
        ctx["analysis"], ctx["file_info"], ctx["pdf_part"], result
    )
    ctx["assembled"] = result


@then(parsers.parse("the assembled result is {expected}"))
def assembled_result(ctx, expected):
    part = ctx["pdf_part"]
    result = ctx["assembled"]
    if expected == "the original part":
        assert result == [part]
    elif expected == "a brave-pdf-text part with the text":
        assert result == [
            {"type": "brave-pdf-text", "text": ctx["analysis"]["all_text"]}
        ]
    elif expected == "a truncated file part plus a brave-pdf-text part":
        assert len(result) == 2
        assert result[0]["type"] == "file"
        assert result[0]["file"]["file_data"].startswith(PDF_PREFIX)
        assert result[0]["file"]["file_data"] != ctx["pdf_part"]["file"]["file_data"]
        assert result[1] == {
            "type": "brave-pdf-text",
            "text": ctx["analysis"]["overflow_text"],
        }
    elif expected == "a truncated file part only":
        assert len(result) == 1
        assert result[0]["type"] == "file"
        assert result[0]["file"]["file_data"].startswith(PDF_PREFIX)
    else:
        raise AssertionError(f"unknown expected result: {expected}")


@when(parsers.parse("content parts {parts_desc} are processed"))
def process_parts(ctx, parts_desc, monkeypatch):
    if parts_desc == "contain only text parts":
        parts = [{"type": "text", "text": "hi"}]
        ctx["expected_parts"] = parts
    elif parts_desc == "contain a non-pdf file":
        parts = [{"type": "file", "file": {"file_data": "data:image/png;base64,AAAA"}}]
        ctx["expected_parts"] = parts
    elif parts_desc == "exceed the size limit":
        parts = [dict(ctx["pdf_part"])]
        ctx["expected_parts"] = parts
        monkeypatch.setattr(external_service_settings, "max_pdf_file_size_mb", 0.000001)
    elif parts_desc == "hit a sandbox op error":
        parts = [dict(ctx["pdf_part"])]
        ctx["expected_parts"] = parts
        ctx["pool"].error = SandboxOpError("pdf_analyze failed in sandbox")
    elif parts_desc == "hit an unexpected error":
        parts = [dict(ctx["pdf_part"])]
        ctx["expected_parts"] = parts
        ctx["pool"].error = RuntimeError("boom")
    else:
        raise AssertionError(f"unknown parts description: {parts_desc}")
    ctx["processed"] = asyncio.run(
        pdf_service.process_pdf_text_content_parts(parts, 4800)
    )


@then(parsers.parse("the processed result is {expected}"))
def processed_result(ctx, expected):
    if (
        expected == "the parts pass through unchanged"
        or expected == "the pdf part passes through unchanged"
    ):
        assert ctx["processed"] == ctx["expected_parts"]
    else:
        raise AssertionError(f"unknown expected: {expected}")


@when("the sandbox worker dies during content part processing")
def worker_dies_content_parts(ctx):
    ctx["pool"].error = SandboxWorkerError("worker died")
    with pytest.raises(SandboxWorkerError) as exc:
        asyncio.run(
            pdf_service.process_pdf_text_content_parts([dict(ctx["pdf_part"])], 4800)
        )
    ctx["error"] = exc.value


@then("processing raises SandboxWorkerError")
def processing_raises_worker_error(ctx):
    assert isinstance(ctx["error"], SandboxWorkerError)


@when("a pdf content part is processed with a successful analysis")
def process_pdf_part_success(ctx):
    ctx["pool"].result = {
        "passthrough": None,
        "total_pages": 90,
        "all_text": "hello text",
    }
    ctx["processed"] = asyncio.run(
        pdf_service.process_pdf_text_content_parts([dict(ctx["pdf_part"])], 4800)
    )


@then("the sandbox op was called once and the parts contain the brave-pdf-text")
def sandbox_called_once_with_brave_pdf_text(ctx):
    assert len(ctx["pool"].calls) == 1
    assert ctx["processed"] == [{"type": "brave-pdf-text", "text": "hello text"}]


@when(parsers.parse("messages {messages_desc} are preprocessed"))
def preprocess_messages(ctx, messages_desc, monkeypatch):
    pdf_part = dict(ctx["pdf_part"])
    if messages_desc == "are assistant-only":
        messages = [
            {"role": "assistant", "content": [dict(pdf_part)]},
        ]
    elif messages_desc == "have a user message with plain text":
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
    elif messages_desc == "have a user message with a pdf file":
        messages = [
            {"role": "user", "content": [dict(pdf_part)]},
        ]
    else:
        raise AssertionError(f"unknown messages description: {messages_desc}")

    parts_mock = AsyncMock()
    monkeypatch.setattr(pdf_service, "process_pdf_text_content_parts", parts_mock)
    ctx["messages"] = asyncio.run(
        pdf_service.process_messages_for_pdf_limits(messages, 6400)
    )
    ctx["parts_mock"] = parts_mock


@then(parsers.parse("the sandbox processing {outcome}"))
def sandbox_processing_outcome(ctx, outcome):
    if outcome == "is skipped entirely":
        ctx["parts_mock"].assert_not_awaited()
    elif outcome == "runs on the pdf parts":
        ctx["parts_mock"].assert_awaited_once()
    else:
        raise AssertionError(f"unknown outcome: {outcome}")
