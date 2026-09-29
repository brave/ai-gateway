"""Fail-closed gate for client-supplied data URLs.

Security rule (review of PR #79): any data URL that is not passed through
the media processor must be removed from the message before it reaches a
provider. A whitelist of known safe binary MIME types bypasses the
processor — verified against litellm 1.93.0 converters: image/png, jpeg,
gif and webp data URLs convert to clean provider image blocks, while
everything else (svg, audio-as-file, json, octet-stream, text) either
errors on Bedrock or is forwarded raw as a document.

Non-whitelisted data URLs are sniffed by the media processor
(``/v1/media/inspect``): a part is kept only when the processor confirms
the payload really is a known-safe binary type. Any processor failure
removes the part (fail-closed).
"""

import logging

from aichat.serve.media_client import call_media_inspect
from aichat.serve.metrics import DATA_URL_PART_REMOVED
from aichat.serve.services.pdf import _is_pdf_file_data

logger = logging.getLogger(__name__)

# Known safe binary files: litellm converts these to provider image blocks
# on both bedrock_converse and anthropic backends (see the security review).
SAFE_BINARY_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp"}
)


def _data_url_mime(data_url: str) -> str | None:
    """Exact MIME of a data URL ("data:<mime>[;params],<payload>"), else None."""
    if not data_url.startswith("data:"):
        return None
    head = data_url.partition(",")[0]
    return head[5:].split(";", 1)[0]


def _should_keep_data_url(mime: str | None) -> bool:
    return mime is not None and mime in SAFE_BINARY_MIME_TYPES


async def _inspect_data_url(data_url: str) -> bool:
    """True when the media processor confirms a known-safe binary payload."""
    try:
        verdict = await call_media_inspect(data_url)
    except Exception:
        logger.warning("media inspect unavailable; failing closed (removing part)")
        return False
    return bool(verdict.get("is_binary_safe"))


async def _keep_file_part(part: dict) -> bool:
    file_data = part.get("file", {}).get("file_data", "")
    if _is_pdf_file_data(file_data):
        # PDFs go through the media processor sandbox flow (pdf service),
        # which is fail-closed on processing failures.
        return True
    mime = _data_url_mime(file_data)
    if _should_keep_data_url(mime):
        return True
    if mime is None:
        # Not a data URL (remote URL etc.): not part of this gate.
        return True
    if await _inspect_data_url(file_data):
        return True
    DATA_URL_PART_REMOVED.inc()
    logger.warning("removed file part with unprocessed data URL (mime=%s)", mime)
    return False


async def _keep_image_part(part: dict) -> bool:
    url = part.get("image_url", {}).get("url", "")
    mime = _data_url_mime(url)
    if _should_keep_data_url(mime):
        return True
    if mime is None:
        return True
    if await _inspect_data_url(url):
        return True
    DATA_URL_PART_REMOVED.inc()
    logger.warning("removed image part with unprocessed data URL (mime=%s)", mime)
    return False


async def remove_unprocessed_data_url_parts(messages: list[dict]) -> list[dict]:
    """Drop every data-URL content part that the media processor did not pass.

    Walks user messages with list content (same scope as the PDF limits
    flow) and removes file/image parts whose data URLs are neither on the
    known-safe-binary whitelist nor confirmed safe by the media processor.
    """
    result = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, list):
            kept_parts = []
            for part in content:
                part_type = part.get("type")
                if part_type == "file":
                    if await _keep_file_part(part):
                        kept_parts.append(part)
                elif part_type == "image_url":
                    if await _keep_image_part(part):
                        kept_parts.append(part)
                else:
                    kept_parts.append(part)
            message = {**message, "content": kept_parts}
        result.append(message)
    return result
