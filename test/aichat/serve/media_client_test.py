import httpx
import pytest
import pytest_asyncio

from aichat.serve import media_client
from aichat.serve.media_client import (
    SandboxOpError,
    SandboxWorkerError,
    call_pdf_analyze,
    call_stt_decode,
    shutdown_client,
)
from aichat.serve.metrics import MEDIA_REQUEST_RETRY_TOTAL

PDF_URL = "http://ai-gateway-media-processor:8000/v1/pdf/analyze"
STT_URL = "http://ai-gateway-media-processor:8000/v1/stt/decode"

pytestmark = pytest.mark.asyncio


def _retry_count(path: str, outcome: str) -> float:
    return MEDIA_REQUEST_RETRY_TOTAL.labels(path=path, outcome=outcome)._value.get()


@pytest_asyncio.fixture(autouse=True)
async def _reset_media_client():
    await shutdown_client()
    yield
    await shutdown_client()


async def test_pdf_analyze_returns_json(httpx_mock):
    httpx_mock.add_response(url=PDF_URL, json={"total_pages": 2})
    assert await call_pdf_analyze({"pdf_b64": "abc"}) == {"total_pages": 2}


async def test_stt_decode_returns_json(httpx_mock):
    httpx_mock.add_response(url=STT_URL, json={"n_samples": 4, "pcm_b64": "aaaa"})
    assert await call_stt_decode({"audio_b64": "abcd"}) == {
        "n_samples": 4,
        "pcm_b64": "aaaa",
    }


async def test_connect_error_retries_once_then_succeeds(httpx_mock):
    path = "/v1/pdf/analyze"
    before_attempted = _retry_count(path, "attempted")
    before_succeeded = _retry_count(path, "succeeded")
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    httpx_mock.add_response(url=PDF_URL, json={"total_pages": 1})

    assert await call_pdf_analyze({"pdf_b64": "abc"}) == {"total_pages": 1}
    assert len(httpx_mock.get_requests()) == 2
    assert _retry_count(path, "attempted") == before_attempted + 1
    assert _retry_count(path, "succeeded") == before_succeeded + 1


async def test_connect_error_exhausted_records_failure(httpx_mock):
    path = "/v1/pdf/analyze"
    before_failed = _retry_count(path, "failed")
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))

    with pytest.raises(SandboxWorkerError, match="unreachable") as exc_info:
        await call_pdf_analyze({"pdf_b64": "abc"})

    assert isinstance(exc_info.value, SandboxWorkerError)
    assert not isinstance(exc_info.value, SandboxOpError)
    assert len(httpx_mock.get_requests()) == 2
    assert _retry_count(path, "failed") == before_failed + 1


async def test_read_error_is_not_retried(httpx_mock):
    path = "/v1/pdf/analyze"
    before_attempted = _retry_count(path, "attempted")
    httpx_mock.add_exception(httpx.ReadError("connection dropped"))

    with pytest.raises(SandboxWorkerError, match="unreachable") as exc_info:
        await call_pdf_analyze({"pdf_b64": "abc"})

    assert "ReadError" in str(exc_info.value)
    assert len(httpx_mock.get_requests()) == 1
    assert _retry_count(path, "attempted") == before_attempted


async def test_timeout_is_not_retried_and_is_distinct(httpx_mock):
    httpx_mock.add_exception(httpx.ReadTimeout("slow"))

    with pytest.raises(SandboxWorkerError, match="timed out") as exc_info:
        await call_pdf_analyze({"pdf_b64": "abc"})

    message = str(exc_info.value)
    assert "ReadTimeout" in message
    assert "unreachable" not in message
    assert len(httpx_mock.get_requests()) == 1


async def test_422_maps_to_op_error(httpx_mock):
    httpx_mock.add_response(url=PDF_URL, status_code=422, json={"error": "bad pdf"})

    with pytest.raises(SandboxOpError, match="bad pdf"):
        await call_pdf_analyze({"pdf_b64": "abc"})


async def test_422_truncates_long_error(httpx_mock):
    httpx_mock.add_response(url=PDF_URL, status_code=422, json={"error": "x" * 500})

    with pytest.raises(SandboxOpError) as exc_info:
        await call_pdf_analyze({"pdf_b64": "abc"})

    assert str(exc_info.value) == "x" * media_client._MAX_ERROR_CHARS


async def test_422_non_json_is_worker_error(httpx_mock):
    httpx_mock.add_response(url=PDF_URL, status_code=422, text="not-json")

    with pytest.raises(SandboxWorkerError, match="non-JSON") as exc_info:
        await call_pdf_analyze({"pdf_b64": "abc"})

    assert not isinstance(exc_info.value, SandboxOpError)
    assert "not-json" not in str(exc_info.value)


async def test_200_non_json_is_worker_error(httpx_mock):
    httpx_mock.add_response(url=PDF_URL, status_code=200, text="not-json")

    with pytest.raises(SandboxWorkerError, match="non-JSON") as exc_info:
        await call_pdf_analyze({"pdf_b64": "abc"})

    assert "not-json" not in str(exc_info.value)


async def test_200_non_object_json_is_worker_error(httpx_mock):
    httpx_mock.add_response(url=PDF_URL, status_code=200, json=["not", "object"])

    with pytest.raises(SandboxWorkerError, match="non-object") as exc_info:
        await call_pdf_analyze({"pdf_b64": "abc"})

    assert not isinstance(exc_info.value, SandboxOpError)


async def test_error_status_omits_response_body(httpx_mock):
    secret = "user-document-fragment"
    httpx_mock.add_response(url=PDF_URL, status_code=500, text=secret)

    with pytest.raises(SandboxWorkerError, match="returned 500") as exc_info:
        await call_pdf_analyze({"pdf_b64": "abc"})

    assert secret not in str(exc_info.value)


async def test_request_over_limit_is_not_sent(httpx_mock, monkeypatch):
    monkeypatch.setattr(media_client, "_MAX_REQUEST_BYTES", 32)

    with pytest.raises(SandboxWorkerError, match="request exceeds"):
        await call_pdf_analyze({"pdf_b64": "x" * 64})

    assert httpx_mock.get_requests() == []


async def test_response_over_limit_raises(httpx_mock, monkeypatch):
    monkeypatch.setattr(media_client, "_MAX_RESPONSE_BYTES", 8)
    httpx_mock.add_response(url=PDF_URL, json={"total_pages": 1})

    with pytest.raises(SandboxWorkerError, match="response exceeds"):
        await call_pdf_analyze({"pdf_b64": "abc"})


async def test_shutdown_client_closes_and_allows_reuse(httpx_mock):
    httpx_mock.add_response(url=PDF_URL, json={"total_pages": 1})
    httpx_mock.add_response(url=PDF_URL, json={"total_pages": 2})

    assert await call_pdf_analyze({"pdf_b64": "abc"}) == {"total_pages": 1}
    assert media_client._client is not None
    await shutdown_client()
    assert media_client._client is None
    assert await call_pdf_analyze({"pdf_b64": "abc"}) == {"total_pages": 2}
