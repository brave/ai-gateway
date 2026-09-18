"""Tests for PDF service (Bedrock 100-page limit splitting)."""

import asyncio
import base64
import io
from unittest.mock import ANY, patch

from pypdf import PdfWriter

from aichat.serve.services import pdf


def _make_pdf_bytes(num_pages: int) -> bytes:
    """Create a minimal valid PDF with num_pages blank pages."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(612, 792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _pdf_data_url(pdf_bytes: bytes) -> str:
    return f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}"


def _run(coro):
    return asyncio.run(coro)


class TestIsPdfFileData:
    def test_true_for_pdf_data_url(self):
        assert pdf._is_pdf_file_data("data:application/pdf;base64,YWJj") is True

    def test_false_for_empty(self):
        assert pdf._is_pdf_file_data("") is False

    def test_false_for_other_data_url(self):
        assert pdf._is_pdf_file_data("data:image/png;base64,abc") is False

    def test_false_for_plain_text(self):
        assert pdf._is_pdf_file_data("not a data url") is False


class TestDecodePdfBytes:
    def test_decodes_valid_data_url(self):
        raw = b"binary \x00 pdf content"
        b64 = base64.b64encode(raw).decode()
        data_url = f"{pdf.PDF_DATA_URL_PREFIX}{b64}"
        assert pdf._decode_pdf_bytes(data_url) == raw


class TestComputeMaxExtractionTokens:
    def test_uses_75_percent_of_limit(self):
        assert pdf._compute_max_extraction_tokens(100_000) == 75_000

    def test_none_uses_default(self):
        assert pdf._compute_max_extraction_tokens(None) == int(
            pdf.DEFAULT_TOKEN_LIMIT * pdf.TOKEN_BUDGET_RATIO
        )

    def test_caps_at_default_token_limit(self):
        # Premium limits (e.g. 512K) must not exceed the actual model context.
        assert pdf._compute_max_extraction_tokens(512_000) == int(
            pdf.DEFAULT_TOKEN_LIMIT * pdf.TOKEN_BUDGET_RATIO
        )


class TestProcessPdfTextContentParts:
    def test_non_file_part_unchanged(self):
        part = {"type": "text", "text": "hello"}
        result = asyncio.run(
            pdf.process_pdf_text_content_parts([part], max_extraction_tokens=50_000)
        )
        assert result == [part]

    def test_file_part_non_pdf_unchanged(self):
        part = {
            "type": "file",
            "file": {"filename": "x.txt", "file_data": "data:text/plain;base64,aGk="},
        }
        result = asyncio.run(
            pdf.process_pdf_text_content_parts([part], max_extraction_tokens=50_000)
        )
        assert result == [part]

    def test_pdf_under_page_limit_unchanged(self):
        pdf_bytes = _make_pdf_bytes(50)
        data_url = _pdf_data_url(pdf_bytes)
        part = {"type": "file", "file": {"filename": "doc.pdf", "file_data": data_url}}
        analysis = {
            "passthrough": None,
            "total_pages": 50,
            "estimated_tokens": 10,
        }
        with (
            patch(
                "aichat.serve.services.pdf.external_service_settings"
            ) as mock_settings,
            patch("aichat.serve.services.pdf._analyze_pdf") as mock_analyze,
        ):
            mock_settings.max_pdf_file_size_mb = 20
            mock_analyze.return_value = analysis
            result = asyncio.run(pdf.process_pdf_text_content_parts([part], 50_000))
        assert len(result) == 1
        assert result[0]["type"] == "file"
        assert result[0]["file"]["file_data"] == data_url

    def test_pdf_over_page_limit_converted_to_text(self):
        pdf_bytes = _make_pdf_bytes(100)
        data_url = _pdf_data_url(pdf_bytes)
        part = {"type": "file", "file": {"filename": "big.pdf", "file_data": data_url}}
        analysis = {
            "passthrough": None,
            "total_pages": 100,
            "all_text": "Page 1 text\nPage 2 text",
            "all_was_truncated": False,
        }
        with (
            patch(
                "aichat.serve.services.pdf.external_service_settings"
            ) as mock_settings,
            patch("aichat.serve.services.pdf._analyze_pdf") as mock_analyze,
        ):
            mock_settings.max_pdf_file_size_mb = 20
            mock_analyze.return_value = analysis
            result = asyncio.run(pdf.process_pdf_text_content_parts([part], 100_000))
        assert len(result) == 1
        assert result[0]["type"] == "brave-pdf-text"
        assert result[0]["text"] == "Page 1 text\nPage 2 text"
        mock_analyze.assert_awaited_once_with(ANY, 100_000)

    def test_pdf_over_page_limit_no_text_passes_through(self):
        # Blank/image-only PDFs with no extractable text are passed through so
        # Bedrock can attempt native processing.
        pdf_bytes = _make_pdf_bytes(100)
        data_url = _pdf_data_url(pdf_bytes)
        part = {"type": "file", "file": {"filename": "big.pdf", "file_data": data_url}}
        with (
            patch(
                "aichat.serve.services.pdf.external_service_settings"
            ) as mock_settings,
            patch("aichat.serve.services.pdf._analyze_pdf") as mock_analyze,
        ):
            mock_settings.max_pdf_file_size_mb = 20
            mock_analyze.return_value = {
                "passthrough": None,
                "total_pages": 100,
                "all_text": "",
                "all_was_truncated": False,
            }
            result = asyncio.run(pdf.process_pdf_text_content_parts([part], 100_000))
        assert len(result) == 1
        assert result[0]["type"] == "file"
        assert result[0]["file"]["file_data"] == data_url

    def test_pdf_over_size_limit_passed_through(self):
        pdf_bytes = _make_pdf_bytes(100)
        data_url = _pdf_data_url(pdf_bytes)
        part = {"type": "file", "file": {"filename": "big.pdf", "file_data": data_url}}
        with patch(
            "aichat.serve.services.pdf.external_service_settings"
        ) as mock_settings:
            mock_settings.max_pdf_file_size_mb = 0
            result = asyncio.run(pdf.process_pdf_text_content_parts([part], 100_000))
        assert len(result) == 1
        assert result[0]["file"]["file_data"] == data_url


class TestProcessMessagesForPdfLimits:
    def test_messages_without_pdf_unchanged(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": "Hello"},
        ]
        result = asyncio.run(
            pdf.process_messages_for_pdf_limits(
                messages, conversation_token_limit=128_000
            )
        )
        assert result == messages

    def test_user_message_with_pdf_over_limit_converted_to_text(self):
        pdf_bytes = _make_pdf_bytes(100)
        data_url = _pdf_data_url(pdf_bytes)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this"},
                    {
                        "type": "file",
                        "file": {"filename": "x.pdf", "file_data": data_url},
                    },
                ],
            },
        ]
        with (
            patch(
                "aichat.serve.services.pdf.external_service_settings"
            ) as mock_settings,
            patch("aichat.serve.services.pdf._analyze_pdf") as mock_analyze,
        ):
            mock_settings.max_pdf_file_size_mb = 20
            mock_analyze.return_value = {
                "passthrough": None,
                "total_pages": 100,
                "all_text": "extracted text",
                "all_was_truncated": False,
            }
            result = asyncio.run(
                pdf.process_messages_for_pdf_limits(
                    messages, conversation_token_limit=128_000
                )
            )
        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "brave-pdf-text"
        assert content[1]["text"] == "extracted text"
