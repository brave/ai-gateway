import logging
from copy import deepcopy

from aichat.llm.metrics import TOKEN_TRIMMING_AMOUNT, TOKEN_TRIMMING_TOTAL
from aichat.prompts.file_extracted_text import TYPE as FILE_EXTRACTED_TEXT_TYPE
from aichat.prompts.page_excerpt import TYPE as PAGE_EXCERPT_TYPE
from aichat.prompts.page_text import TYPE as PAGE_TEXT_TYPE
from aichat.prompts.pdf_text_content import TYPE as PDF_TEXT_CONTENT_TYPE
from aichat.prompts.request_summary import TYPE as REQUEST_SUMMARY_TYPE
from aichat.prompts.search_results import TYPE as SEARCH_RESULTS_TYPE
from aichat.prompts.video_transcript import TYPE as VIDEO_TRANSCRIPT_TYPE
from aichat.serve.services.compaction import calculate_max_input_tokens
from aichat.serve.services.compaction_settings import compaction_settings
from aichat.serve.services.models import ModelConfig
from aichat.serve.utils import DEFAULT_TOKENIZER, calculate_message_tokens

"""
Token trimming service for OpenAI API.

Handles trimming of large content parts within messages to fit within token limits.
This is complementary to compaction - while compaction summarizes old conversation turns,
trimming truncates large attached content (page text, search results, etc.) in current messages.
"""

logger = logging.getLogger(__name__)

TRIMMABLE_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        PAGE_TEXT_TYPE,
        PAGE_EXCERPT_TYPE,
        SEARCH_RESULTS_TYPE,
        VIDEO_TRANSCRIPT_TYPE,
        FILE_EXTRACTED_TEXT_TYPE,
        PDF_TEXT_CONTENT_TYPE,
        REQUEST_SUMMARY_TYPE,
    }
)


def truncate_text(text: str, excess_tokens: int) -> tuple[str, int]:
    encoded = DEFAULT_TOKENIZER.encode(text)
    allowed = max(0, len(encoded) - excess_tokens)
    removed = len(encoded) - allowed
    return DEFAULT_TOKENIZER.decode(encoded[:allowed]), removed


def is_trimmable_content_part(part: dict) -> bool:
    return part.get("type") in TRIMMABLE_CONTENT_TYPES


def trim_tool_messages(
    messages: list[dict],
    max_tokens: int,
) -> tuple[list[dict], int]:
    """
    Trim tool message string content to fit within a token budget.

    Iterates through messages from earliest to latest, truncating the string
    content of ``role: tool`` messages until the total token count is within
    ``max_tokens``.  Only messages that exceed the budget are modified; the
    others are returned as-is.

    This is useful standalone — e.g. before sending messages to the compaction
    model — as well as being called internally by ``trim_messages_to_fit``.

    Args:
        messages: List of OpenAI message dicts.  Will not be modified; a deep
            copy is made when trimming is required.
        max_tokens: Token budget.  Pass 0 or a negative value to skip trimming.

    Returns:
        Tuple of (messages, tokens_trimmed).  ``messages`` is the original list
        when no trimming was needed, or a modified deep copy otherwise.
    """
    if max_tokens is None or max_tokens <= 0:
        return messages, 0

    total_tokens = calculate_message_tokens(messages)
    if total_tokens <= max_tokens:
        return messages, 0

    working_messages = deepcopy(messages)
    excess_tokens = total_tokens - max_tokens
    _, trimmed = _trim_tool_messages(working_messages, excess_tokens)
    return working_messages, trimmed


def _trim_tool_messages(
    messages: list[dict],
    excess_tokens: int,
) -> tuple[int, int]:
    """
    Mutate *messages* in-place, trimming tool message content until
    ``excess_tokens`` reaches zero.

    Args:
        messages: Mutable list of message dicts (already deep-copied by caller).
        excess_tokens: How many tokens need to be removed.

    Returns:
        Tuple of (remaining_excess_tokens, total_trimmed_tokens).
    """
    total_trimmed = 0
    for message in messages:
        if excess_tokens <= 0:
            break

        if message.get("role") != "tool":
            continue

        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue

        try:
            truncated_content, tokens_removed = truncate_text(content, excess_tokens)

            if tokens_removed > 0:
                message["content"] = (
                    truncated_content
                    + "\n[... tool response truncated to fit conversation limit]"
                )
                excess_tokens -= tokens_removed
                total_trimmed += tokens_removed

                logger.debug(
                    f"Trimmed {tokens_removed} tokens from tool message "
                    f"(remaining excess: {excess_tokens})"
                )
        except Exception as e:
            logger.warning(f"Failed to trim tool message content: {e}")
            continue

    return excess_tokens, total_trimmed


def _trim_text_parts(
    messages: list[dict],
    excess_tokens: int,
) -> tuple[int, int]:
    """
    Mutate *messages* in-place, truncating oversized ``type:"text"`` content
    parts until ``excess_tokens`` reaches zero or all eligible parts have been
    capped to ``compaction_settings.max_text_part_tokens``.

    This is Pass 3 in ``trim_messages_to_fit`` and targets large plain-text
    blobs (e.g. assistant prose, PDF text repeated in a generic text part) that
    are not covered by the Brave-type or tool-message passes.

    Args:
        messages: Mutable list of message dicts (already deep-copied by caller).
        excess_tokens: How many tokens still need to be removed.

    Returns:
        Tuple of (remaining_excess_tokens, total_trimmed_tokens).
    """
    max_text_part_tokens = compaction_settings.max_text_part_tokens
    total_trimmed = 0
    for message in messages:
        if excess_tokens <= 0:
            break
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if excess_tokens <= 0:
                break
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = part.get("text", "")
            if not text:
                continue
            part_tokens = len(DEFAULT_TOKENIZER.encode(text))
            if part_tokens <= max_text_part_tokens:
                continue
            tokens_to_remove = min(part_tokens - max_text_part_tokens, excess_tokens)
            try:
                truncated, tokens_removed = truncate_text(text, tokens_to_remove)
                if tokens_removed > 0:
                    part["text"] = (
                        truncated
                        + "\n[... content truncated to fit conversation limit]"
                    )
                    excess_tokens -= tokens_removed
                    total_trimmed += tokens_removed
                    logger.debug(
                        f"Trimmed {tokens_removed} tokens from oversized text part "
                        f"in {message.get('role')} message "
                        f"(remaining excess: {excess_tokens})"
                    )
            except Exception as e:
                logger.warning(f"Failed to trim text part: {e}")
    return excess_tokens, total_trimmed


def _log_trim_insufficient(
    messages: list[dict],
    total_tokens_before: int,
    total_trimmed: int,
    excess_tokens: int,
) -> None:
    """
    Emit a warning log when trimming was unable to bring the conversation
    within budget, including a per-role token breakdown to aid debugging.

    Args:
        messages: The (possibly modified) working message list.
        total_tokens_before: Token count before any trimming.
        total_trimmed: Total tokens removed across all passes.
        excess_tokens: Tokens still over budget after all passes.
    """
    remaining_total = total_tokens_before - total_trimmed
    per_role: dict[str, int] = {}
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content")
        msg_tokens = 4
        if isinstance(content, str):
            msg_tokens += len(DEFAULT_TOKENIZER.encode(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text:
                        msg_tokens += len(DEFAULT_TOKENIZER.encode(text))
        per_role[role] = per_role.get(role, 0) + msg_tokens
    logger.warning(
        f"Token trimming insufficient: {remaining_total} tokens remain "
        f"({excess_tokens} over budget). Per-role breakdown: {per_role}"
    )


def trim_messages_to_fit(
    messages: list[dict],
    max_input_tokens: int,
) -> tuple[list[dict], int, int]:
    """
    Trim content parts within messages to fit within token limit.

    Iterates through messages from earliest to latest, truncating trimmable
    content parts until the token count is within the limit.

    Args:
        messages: List of OpenAI message dicts (will not be modified)
        max_input_tokens: Maximum number of tokens allowed

    Returns:
        Tuple of (trimmed_messages, total_tokens_before, total_trimmed_tokens)
    """
    if max_input_tokens is None or max_input_tokens <= 0:
        return messages, 0, 0

    total_tokens = calculate_message_tokens(messages)

    # If we're within limits, no trimming needed
    if total_tokens <= max_input_tokens:
        return messages, total_tokens, 0

    # Deep copy to avoid modifying original messages
    working_messages = deepcopy(messages)

    excess_tokens = total_tokens - max_input_tokens
    total_trimmed_tokens = 0

    logger.info(
        f"Token trimming needed: {total_tokens} tokens > {max_input_tokens} limit "
        f"(excess: {excess_tokens})"
    )

    # Pass 1: trim Brave custom content parts in multimodal messages (earliest to latest)
    for message in working_messages:
        if excess_tokens <= 0:
            break

        content = message.get("content")

        # Only process messages with list content (multimodal messages)
        if not isinstance(content, list):
            continue

        # Look for trimmable content parts
        for part in content:
            if excess_tokens <= 0:
                break

            if not isinstance(part, dict) or not is_trimmable_content_part(part):
                continue

            # Get the content field (could be 'content' or 'text' depending on type)
            content_field = None
            if "content" in part:
                content_field = "content"
            elif "text" in part:
                content_field = "text"

            if content_field is None:
                continue

            original_content = part[content_field]
            if not original_content:
                continue

            # Truncate this content part
            try:
                truncated_content, tokens_removed = truncate_text(
                    original_content, excess_tokens
                )

                if tokens_removed > 0:
                    part[content_field] = (
                        truncated_content
                        + "\n[... content truncated to fit conversation limit]"
                    )
                    excess_tokens -= tokens_removed
                    total_trimmed_tokens += tokens_removed

                    logger.debug(
                        f"Trimmed {tokens_removed} tokens from {part.get('type')} "
                        f"content part (remaining excess: {excess_tokens})"
                    )
            except Exception as e:
                logger.warning(f"Failed to trim content part: {e}")
                continue

    # Pass 2: trim tool message string content
    excess_tokens, trimmed_from_tools = _trim_tool_messages(
        working_messages, excess_tokens
    )
    total_trimmed_tokens += trimmed_from_tools

    # Pass 3: cap oversized plain type:"text" content parts that survived the
    # previous passes (e.g. large assistant prose or PDF text that ended up in
    # a generic text part rather than a Brave custom type).
    if excess_tokens > 0:
        excess_tokens, trimmed_from_text = _trim_text_parts(
            working_messages, excess_tokens
        )
        total_trimmed_tokens += trimmed_from_text

    if total_trimmed_tokens > 0:
        logger.info(
            f"Trimmed {total_trimmed_tokens} tokens from {len(messages)} messages "
            f"({total_tokens} -> {total_tokens - total_trimmed_tokens} tokens)"
        )

    if excess_tokens > 0:
        _log_trim_insufficient(
            working_messages, total_tokens, total_trimmed_tokens, excess_tokens
        )

    return working_messages, total_tokens, total_trimmed_tokens


def maybe_trim_messages(
    messages: list[dict],
    model_config: ModelConfig,
    is_premium: bool,
    model: str = "",
) -> tuple[list[dict], int, int]:
    """
    Trim large content parts within messages to fit within the model's token limit.

    This is the main entry point for token trimming from the API layer.

    Args:
        messages: List of OpenAI message dicts
        model_config: Model configuration with token limits
        is_premium: Whether this is a premium user
        model: Model name used for metrics labels

    Returns:
        Tuple of (messages, total_tokens, trimmed_tokens)
        - If trimming occurred: (trimmed_messages, original_tokens, tokens_trimmed)
        - If no trimming: (original_messages, total_tokens, 0)
    """
    max_input_tokens, _ = calculate_max_input_tokens(
        model_config, is_premium, margin=compaction_settings.trimming_input_margin
    )

    trimmed_messages, total_tokens, trimmed_tokens = trim_messages_to_fit(
        messages=messages,
        max_input_tokens=max_input_tokens,
    )

    if trimmed_tokens > 0:
        logger.info(f"Applied token trimming: trimmed {trimmed_tokens} tokens")
        TOKEN_TRIMMING_TOTAL.labels(model).inc()
        TOKEN_TRIMMING_AMOUNT.labels(model).observe(trimmed_tokens)

    return trimmed_messages, total_tokens, trimmed_tokens
