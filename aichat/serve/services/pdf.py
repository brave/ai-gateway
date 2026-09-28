import base64
import logging

from aichat.llm.llm_settings import llm_settings
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.media_client import (
    SandboxOpError,
    SandboxWorkerError,
    call_pdf_analyze,
)
from aichat.serve.metrics import PDF_FILE_PART_ENCOUNTERED

logger = logging.getLogger(__name__)

BEDROCK_PDF_PAGE_LIMIT = 100
MAX_ALLOWED_PAGES = 85
TOKEN_BUDGET_RATIO = 0.75
DEFAULT_TOKEN_LIMIT = 128_000

PDF_DATA_URL_PREFIX = "data:application/pdf;base64,"

TRUNCATION_NOTICE = (
    "\n\n[Note: The remaining PDF content was truncated because "
    "the extracted text exceeded the model's context limit. "
    "The content above covers pages {first_overflow_page} through {last_included_page} "
    "of {total_pages} total pages.]"
)


def _is_pdf_file_data(file_data: str) -> bool:
    return file_data.startswith(PDF_DATA_URL_PREFIX)


def _decode_pdf_bytes(file_data: str) -> bytes:
    b64_data = file_data[len(PDF_DATA_URL_PREFIX) :]
    return base64.b64decode(b64_data)


def _encode_pdf_to_data_url(pdf_bytes: bytes) -> str:
    b64_str = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"{PDF_DATA_URL_PREFIX}{b64_str}"


def _compute_max_extraction_tokens(conversation_token_limit: int | None) -> int:
    # Cap at DEFAULT_TOKEN_LIMIT so that a high conversation_token_limit_premium
    # (e.g. 512K) does not produce a budget larger than the actual Bedrock
    # model context window (~200K for Claude).
    limit = min(conversation_token_limit or DEFAULT_TOKEN_LIMIT, DEFAULT_TOKEN_LIMIT)
    return int(limit * TOKEN_BUDGET_RATIO)


async def _analyze_pdf(pdf_bytes: bytes, max_extraction_tokens: int) -> dict:
    try:
        return await call_pdf_analyze(
            {
                "pdf_b64": base64.b64encode(pdf_bytes).decode("ascii"),
                "encoding_name": llm_settings.default_tokenizer,
                "max_extraction_tokens": max_extraction_tokens,
                "max_allowed_pages": MAX_ALLOWED_PAGES,
            }
        )
    except SandboxWorkerError as e:
        logger.error("Media sandbox worker failed hard: %s", e)
        raise


def process_pdf_text_content_parts_sync(
    analysis: dict,
    file_info: dict,
    part: dict,
    result: list[dict],
) -> None:
    if analysis.get("passthrough") == "encrypted":
        logger.warning(
            "PDF is password-protected and cannot be decrypted; passing through as-is"
        )
        result.append(part)
        return

    total_pages = analysis["total_pages"]

    if "all_text" in analysis or total_pages > MAX_ALLOWED_PAGES:
        text = analysis.get("all_text", "")
        if not text.strip():
            logger.warning(
                "PDF has %d pages but no extractable text; passing through as-is",
                total_pages,
            )
            result.append(part)
            return
        result.append({"type": "brave-pdf-text", "text": text})
        return

    estimated_tokens = analysis.get("estimated_tokens", 0)
    if "native_page_limit" not in analysis:
        result.append(part)
        return

    native_page_limit = analysis["native_page_limit"]
    logger.info(
        "PDF has %d pages but estimated %d tokens exceeds budget; "
        "truncating native PDF to %d pages and extracting remaining text",
        total_pages,
        estimated_tokens,
        native_page_limit,
    )
    truncated_data_url = _encode_pdf_to_data_url(
        base64.b64decode(analysis["truncated_pdf_b64"])
    )
    result.append(
        {
            "type": "file",
            "file": {**file_info, "file_data": truncated_data_url},
        }
    )
    overflow_text = analysis.get("overflow_text", "")
    if overflow_text.strip():
        result.append({"type": "brave-pdf-text", "text": overflow_text})


async def process_pdf_text_content_parts(
    content_parts: list[dict], max_extraction_tokens: int
) -> list[dict]:
    result = []

    for part in content_parts:
        if part.get("type") != "file":
            result.append(part)
            continue

        file_info = part.get("file", {})
        file_data = file_info.get("file_data", "")

        if not _is_pdf_file_data(file_data):
            result.append(part)
            continue

        PDF_FILE_PART_ENCOUNTERED.inc()

        try:
            pdf_bytes = _decode_pdf_bytes(file_data)
            max_size_bytes = (
                external_service_settings.max_pdf_file_size_mb * 1024 * 1024
            )
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)

            if len(pdf_bytes) > max_size_bytes:
                logger.warning(
                    f"PDF file size ({pdf_size_mb:.1f} MB) exceeds "
                    f"max allowed size ({external_service_settings.max_pdf_file_size_mb} MB), "
                    f"skipping PDF processing"
                )
                result.append(part)
                continue

            analysis = await _analyze_pdf(pdf_bytes, max_extraction_tokens)
            process_pdf_text_content_parts_sync(analysis, file_info, part, result)

        except SandboxOpError:
            logger.exception(
                "Sandboxed PDF parsing failed (malformed input?), "
                "passing through as-is"
            )
            result.append(part)
        except SandboxWorkerError:
            raise
        except Exception:
            logger.exception(
                "Failed to process PDF for page splitting, passing through as-is"
            )
            result.append(part)

    return result


async def process_messages_for_pdf_limits(
    messages: list[dict], conversation_token_limit: int | None
) -> list[dict]:
    max_extraction_tokens = _compute_max_extraction_tokens(conversation_token_limit)
    processed = []

    for message in messages:
        content = message.get("content")

        if message.get("role") == "user" and isinstance(content, list):
            has_pdf_file = any(
                part.get("type") == "file"
                and _is_pdf_file_data(part.get("file", {}).get("file_data", ""))
                for part in content
            )

            if has_pdf_file:
                new_content = await process_pdf_text_content_parts(
                    content, max_extraction_tokens
                )
                message = {**message, "content": new_content}

        processed.append(message)

    return processed
