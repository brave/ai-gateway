from __future__ import annotations

import base64
import io
import logging

import tiktoken
from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)

TRUNCATION_NOTICE = (
    "\n\n[Note: The remaining PDF content was truncated because "
    "the extracted text exceeded the model's context limit. "
    "The content above covers pages {first_overflow_page} through {last_included_page} "
    "of {total_pages} total pages.]"
)

_encodings: dict[str, object] = {}


def _get_encoding(name: str):
    enc = _encodings.get(name)
    if enc is None:
        enc = tiktoken.get_encoding(name)
        _encodings[name] = enc
    return enc


def preload() -> None:
    for name in ("o200k_base", "cl100k_base"):
        _get_encoding(name)


def _safe_extract_page_text(reader: PdfReader, page_num: int) -> str:
    try:
        return reader.pages[page_num].extract_text() or ""
    except Exception:
        logger.warning(
            "Failed to extract text from PDF page %d; treating as empty",
            page_num + 1,
            exc_info=True,
        )
        return ""


def _tokens_for_page(reader: PdfReader, page_num: int, encoding) -> int:
    page_text = _safe_extract_page_text(reader, page_num)
    if not page_text.strip():
        return 0
    return len(encoding.encode(page_text))


def _estimate_native_pdf_tokens(reader: PdfReader, end_page: int, encoding) -> int:
    total = 0
    for page_num in range(min(end_page, len(reader.pages))):
        total += _tokens_for_page(reader, page_num, encoding)
    return total


def _find_max_pages_within_budget(
    reader: PdfReader, token_budget: int, encoding
) -> int:
    tokens_so_far = 0
    for page_num in range(len(reader.pages)):
        tokens_so_far += _tokens_for_page(reader, page_num, encoding)
        if tokens_so_far > token_budget:
            return max(page_num, 1)
    return len(reader.pages)


def _create_truncated_pdf(reader: PdfReader, max_pages: int) -> bytes:
    writer = PdfWriter()
    for page_num in range(min(max_pages, len(reader.pages))):
        writer.add_page(reader.pages[page_num])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _extract_text_from_pages(
    reader: PdfReader, start_page: int, max_tokens: int, encoding
) -> tuple[str, bool]:
    text_parts: list[str] = []
    tokens_used = 0
    last_included_page = start_page
    truncated = False

    for page_num in range(start_page, len(reader.pages)):
        page_text = _safe_extract_page_text(reader, page_num)
        if not page_text.strip():
            continue

        formatted = f"--- Page {page_num + 1} ---\n{page_text}"
        page_tokens = len(encoding.encode(formatted))

        if tokens_used + page_tokens > max_tokens and text_parts:
            truncated = True
            break

        text_parts.append(formatted)
        tokens_used += page_tokens
        last_included_page = page_num + 1

    result = "\n\n".join(text_parts)
    if truncated:
        result += TRUNCATION_NOTICE.format(
            first_overflow_page=start_page + 1,
            last_included_page=last_included_page,
            total_pages=len(reader.pages),
        )
    return result, truncated


def op_pdf_analyze(
    *,
    pdf_b64: str,
    encoding_name: str,
    max_extraction_tokens: int,
    max_allowed_pages: int,
    **_ignored,
) -> dict:
    encoding = _get_encoding(encoding_name)
    pdf_bytes = base64.b64decode(pdf_b64)
    reader = PdfReader(io.BytesIO(pdf_bytes))

    if reader.is_encrypted and reader.decrypt("") == 0:
        return {"passthrough": "encrypted"}

    total_pages = len(reader.pages)
    result: dict = {"passthrough": None, "total_pages": total_pages}

    if total_pages > max_allowed_pages:
        text, was_truncated = _extract_text_from_pages(
            reader, 0, max_extraction_tokens, encoding
        )
        result["all_text"] = text
        result["all_was_truncated"] = was_truncated
        return result

    estimated_tokens = 0
    for page_num in range(total_pages):
        estimated_tokens += _tokens_for_page(reader, page_num, encoding)
    result["estimated_tokens"] = estimated_tokens
    if estimated_tokens <= max_extraction_tokens:
        return result

    native_page_limit = _find_max_pages_within_budget(
        reader, max_extraction_tokens, encoding
    )
    native_pdf_tokens = _estimate_native_pdf_tokens(reader, native_page_limit, encoding)
    remaining_budget = max(max_extraction_tokens - native_pdf_tokens, 0)
    truncated_pdf_bytes = _create_truncated_pdf(reader, native_page_limit)
    overflow_text, was_truncated = _extract_text_from_pages(
        reader, native_page_limit, remaining_budget, encoding
    )
    result["native_page_limit"] = native_page_limit
    result["truncated_pdf_b64"] = base64.b64encode(truncated_pdf_bytes).decode()
    result["overflow_text"] = overflow_text
    result["overflow_was_truncated"] = was_truncated
    return result


def op_ping(**_ignored) -> dict:
    return {"pong": True}


OP_HANDLERS = {
    "ping": op_ping,
    "pdf_analyze": op_pdf_analyze,
}
