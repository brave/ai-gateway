"""
Conversation compaction module for OpenAI API.

Handles detection of when compaction is needed, selection of messages to compact,
and generation of summaries via LLM.
"""

import asyncio
import json
import logging

from aichat.llm.metrics import (
    COMPACTION_ENDING_TOKEN_COUNT,
    COMPACTION_STARTING_TOKEN_COUNT,
    COMPACTION_TOTAL,
)
from aichat.prompts.compaction_summary import (
    build_fallback_compaction_prompt,
    build_jaguar_compaction_prompt,
)
from aichat.protocol.open_ai_protocol import (
    Capability,
    CapabilityOptions,
    has_capability,
)
from aichat.responses import CompactionMetadata
from aichat.serve.services.backend import get_backend
from aichat.serve.services.compaction_settings import compaction_settings
from aichat.serve.services.models import ModelConfig
from aichat.serve.utils import calculate_message_tokens, count_tokens

logger = logging.getLogger(__name__)


# =============================================================================
# OpenAI API Compaction Functions
# =============================================================================
# These functions work with OpenAI message format (list of dicts).


def check_compaction_needed(
    messages: list[dict],
    max_input_tokens: int,
) -> tuple[bool, int, int]:
    """
    Check if compaction is needed for OpenAI-format messages.

    Args:
        messages: List of OpenAI message dicts
        max_input_tokens: Maximum tokens allowed for input

    Returns:
        tuple of (needs_compaction, current_tokens, threshold_tokens)
    """
    if max_input_tokens is None or max_input_tokens <= 0:
        return (False, 0, 0)

    current_tokens = calculate_message_tokens(messages)
    threshold_tokens = int(max_input_tokens * compaction_settings.compaction_threshold)
    return (current_tokens >= threshold_tokens, current_tokens, threshold_tokens)


def can_compact(messages: list[dict]) -> bool:
    """
    Check if compaction can be performed (enough messages to compact).

    Args:
        messages: List of OpenAI message dicts

    Returns:
        True if compaction is possible
    """
    user_assistant_messages = [
        m for m in messages if m.get("role") in ("user", "assistant")
    ]
    return (
        len(user_assistant_messages)
        > compaction_settings.compaction_min_preserved_turns
    )


def calculate_max_input_tokens(
    model_config: ModelConfig,
    is_premium: bool,
    margin: float | None = None,
) -> tuple[int, int]:
    """
    Calculate the maximum input tokens allowed before compaction/trimming triggers.

    Args:
        model_config: Model configuration with token limits
        is_premium: Whether this is a premium user
        margin: Safety margin to apply (defaults to compaction_input_margin)

    Returns:
        Tuple of (max_input_tokens, conversation_max_tokens)
    """
    if margin is None:
        margin = compaction_settings.compaction_input_margin

    conversation_max_tokens = (
        model_config.conversation_token_limit_premium
        if is_premium
        else model_config.conversation_token_limit
    ) or compaction_settings.compaction_fallback_max_tokens

    max_tokens = (
        model_config.max_tokens or compaction_settings.compaction_fallback_output_tokens
    )
    max_input_tokens = int((conversation_max_tokens - max_tokens) * margin)
    return (max_input_tokens, conversation_max_tokens)


def select_messages_for_compaction(
    messages: list[dict],
) -> tuple[list[dict], list[dict], list[int]]:
    """
    Select which messages to compact vs preserve for OpenAI format.

    Preserves:
    - All system messages
    - Last N user/assistant messages (configurable via compaction_min_preserved_turns)
    - Tool messages associated with preserved assistant messages

    Compacts:
    - Older user/assistant messages (and their tool messages)

    Args:
        messages: List of OpenAI message dicts

    Returns:
        tuple of (messages_to_compact, messages_to_preserve, compacted_indices)
    """
    user_assistant_messages = [
        m for m in messages if m.get("role") in ("user", "assistant")
    ]

    min_turns = compaction_settings.compaction_min_preserved_turns
    if len(user_assistant_messages) <= min_turns:
        return ([], messages, [])

    # Preserve last N user/assistant messages
    preserved_ua = user_assistant_messages[-min_turns:]
    to_compact_ua = user_assistant_messages[:-min_turns]

    # Preserve tool messages that belong to preserved assistant messages
    preserved_tool_call_ids = set()
    for msg in preserved_ua:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    preserved_tool_call_ids.add(tc.get("id"))

    to_compact_set = {id(m) for m in to_compact_ua}

    for msg in messages:
        if (
            msg.get("role") == "tool"
            and msg.get("tool_call_id") not in preserved_tool_call_ids
        ):
            to_compact_set.add(id(msg))

    messages_to_compact = []
    messages_to_preserve = []
    # Track original indices of compacted messages
    compacted_indices = []

    # Keep the original order of the messages
    for i, msg in enumerate(messages):
        if id(msg) in to_compact_set:
            messages_to_compact.append(msg)
            compacted_indices.append(i)
        else:
            messages_to_preserve.append(msg)

    return (messages_to_compact, messages_to_preserve, compacted_indices)


def build_compaction_content(messages_to_compact: list[dict]) -> str:
    """
    Build content string from OpenAI messages for summarization.

    Args:
        messages_to_compact: Messages to summarize

    Returns:
        Formatted conversation string
    """
    content_parts = []
    for msg in messages_to_compact:
        role = msg.get("role", "unknown").capitalize()
        content = _extract_text_from_content(msg.get("content", ""))

        if content:
            content_parts.append(f"{role}: {content}")

    return "\n\n".join(content_parts)


def _extract_text_from_content(content: str | list) -> str:
    """Extract plain text from message content, ignoring non-text parts."""
    if isinstance(content, list):
        text_parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(text_parts)
    if isinstance(content, str):
        return content
    return ""


def _prepare_messages_for_compaction(messages: list[dict]) -> list[dict]:
    """Return message copies sanitized for the Jaguar compaction model."""
    limit = compaction_settings.compaction_max_images

    image_positions: list[tuple[int, int]] = []
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for part_idx, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_positions.append((msg_idx, part_idx))

    to_remove: set[tuple[int, int]] = set()
    if limit > 0 and len(image_positions) > limit:
        to_remove = set(image_positions[: len(image_positions) - limit])

    prepared = []
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            prepared.append(msg)
            continue

        new_content = []
        changed = False
        for part_idx, part in enumerate(content):
            if not isinstance(part, dict):
                changed = True
                continue

            part_type = part.get("type")
            if part_type not in ("text", "image_url"):
                changed = True
                continue
            if (msg_idx, part_idx) in to_remove:
                changed = True
                continue
            if (
                part_type == "image_url"
                and isinstance(part.get("image_url"), dict)
                and "detail" in part["image_url"]
            ):
                image_url = {
                    k: v for k, v in part["image_url"].items() if k != "detail"
                }
                new_content.append({**part, "image_url": image_url})
                changed = True
            else:
                new_content.append(part)

        if changed and not new_content:
            prepared.append({**msg, "content": ""})
        elif changed:
            prepared.append({**msg, "content": new_content})
        else:
            prepared.append(msg)

    return prepared


def _compaction_token_budget(context_window: int) -> int:
    return (
        int(context_window * compaction_settings.compaction_context_safety_factor)
        - compaction_settings.compaction_summary_max_tokens
    )


def _fallback_compaction_token_budget() -> int:
    return (
        int(
            compaction_settings.compaction_fallback_model_context_window
            * compaction_settings.compaction_context_safety_factor
        )
        - compaction_settings.compaction_summary_max_tokens
    )


def _jaguar_compaction_prompt() -> str:
    return build_jaguar_compaction_prompt(
        compaction_settings.compaction_summary_max_tokens,
        compaction_settings.jaguar_style,
        compaction_settings.jaguar_preserve,
        compaction_settings.jaguar_focus,
        compaction_settings.jaguar_recency,
        compaction_settings.jaguar_language,
    )


async def _call_compaction_model(
    model: str,
    messages: list[dict],
    context_window: int,
) -> str:
    backend = get_backend(model)
    params = backend.build_params(
        stream=False,
        tools=[],
        messages=messages,
        context_window_override=context_window,
    )
    params["temperature"] = compaction_settings.compaction_summary_temperature
    params["max_tokens"] = compaction_settings.compaction_summary_max_tokens

    response = await backend.converse(messages, stream=False, params=params)
    if isinstance(response, dict):
        raise RuntimeError(  # noqa: TRY004 - runtime failure, not a type error
            f"Compaction summary generation failed for {model}: "
            f"backend returned error response: {response}"
        )
    if not response.choices:
        raise RuntimeError(
            f"Compaction summary generation failed for {model}: empty response"
        )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(
            f"Compaction summary generation failed for {model}: empty content"
        )
    return content


def _build_jaguar_compaction_messages(messages_to_compact: list[dict]) -> list[dict]:
    token_budget = _compaction_token_budget(
        compaction_settings.compaction_model_context_window
    )
    prepared = _prepare_messages_for_compaction(list(messages_to_compact))
    summary_request = {"role": "user", "content": _jaguar_compaction_prompt()}

    working = prepared
    while working:
        messages = working + [summary_request]
        if count_tokens(messages) <= token_budget:
            return messages
        working = working[1:]
        logger.debug(
            "Truncated oldest message from Jaguar compaction input; "
            "%s messages remaining",
            len(working),
        )

    raise RuntimeError(
        "No messages remain after truncating Jaguar compaction input to fit "
        "model context window"
    )


def _build_fallback_compaction_messages(messages_to_compact: list[dict]) -> list[dict]:
    token_budget = _fallback_compaction_token_budget()
    working = list(messages_to_compact)

    while working:
        content = build_compaction_content(working)
        prompt = build_fallback_compaction_prompt(
            content, compaction_settings.compaction_summary_max_tokens
        )
        messages = [{"role": "user", "content": prompt}]
        if count_tokens(messages) <= token_budget:
            return messages
        working = working[1:]
        logger.debug(
            "Truncated oldest message from fallback compaction input; "
            "%s messages remaining",
            len(working),
        )

    raise RuntimeError(
        "No messages remain after truncating fallback compaction input to fit "
        "model context window"
    )


def _compaction_chunk_token_limit(context_window: int) -> int:
    return min(
        compaction_settings.compaction_chunk_max_tokens,
        _compaction_token_budget(context_window),
    )


def _jaguar_chunk_token_limit() -> int:
    return _compaction_chunk_token_limit(
        compaction_settings.compaction_model_context_window
    )


def _fallback_chunk_token_limit() -> int:
    return min(
        compaction_settings.compaction_fallback_chunk_max_tokens,
        _fallback_compaction_token_budget(),
    )


def _chunk_messages_for_compaction(
    messages_to_compact: list[dict], chunk_limit: int
) -> list[list[dict]]:
    total_compact_tokens = calculate_message_tokens(messages_to_compact)
    if total_compact_tokens <= chunk_limit:
        return [messages_to_compact]
    return _chunk_messages_by_tokens(messages_to_compact, chunk_limit)


def _chunk_messages_by_tokens(
    messages: list[dict], max_tokens: int
) -> list[list[dict]]:
    if not messages:
        return []

    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []
    current_tokens = 0

    for msg in messages:
        msg_tokens = calculate_message_tokens([msg])
        if current_chunk and current_tokens + msg_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(msg)
        current_tokens += msg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _section_label(index: int, total: int) -> str | None:
    if total <= 1:
        return None
    if total == 2:
        return "start" if index == 0 else "end"
    if index == 0:
        return "start"
    if index == total - 1:
        return "end"
    return "middle"


def _extract_summary_text(msg: dict) -> str:
    content = msg.get("content", "")
    start = content.find("<context>\n")
    end = content.find("\n</context>")
    if start != -1 and end != -1:
        return content[start + len("<context>\n") : end]
    return content


def _is_summary_message(msg: dict) -> bool:
    return "<context>" in msg.get("content", "")


def create_summary_message(summary: str, section: str | None = None) -> dict:
    """
    Create an OpenAI-format system message containing the compaction summary.

    Args:
        summary: The generated summary text
        section: Optional conversation section label (e.g. start, middle, end)

    Returns:
        OpenAI message dict
    """
    section_note = ""
    if section:
        section_note = f"This is a compaction of the {section} of the conversation.\n\n"
    return {
        "role": "system",
        "content": (
            section_note
            + "The text in <context> tags is what you already know from earlier "
            "in this conversation. Treat it as your own memory and use it "
            "silently to keep the conversation coherent.\n\n"
            "Speak naturally, as if you simply remember these details. "
            "Refer to things the way a person recalling a chat would, e.g. "
            '"earlier you mentioned..." rather than "according to the '
            'summary...". Do not describe, quote, or mention that this '
            "context exists.\n\n"
            f"<context>\n{summary}\n</context>"
        ),
    }


async def _summarize_jaguar_chunk(messages_to_compact: list[dict]) -> str:
    messages = _build_jaguar_compaction_messages(messages_to_compact)
    return await _call_compaction_model(
        compaction_settings.compaction_model,
        messages,
        compaction_settings.compaction_model_context_window,
    )


async def _summarize_haiku_chunk(messages_to_compact: list[dict]) -> str:
    messages = _build_fallback_compaction_messages(messages_to_compact)
    return await _call_compaction_model(
        compaction_settings.compaction_fallback_model,
        messages,
        compaction_settings.compaction_fallback_model_context_window,
    )


async def generate_compaction_summary(messages_to_compact: list[dict]) -> str:
    """Generate a Jaguar compaction summary for a single chunk."""
    return await _summarize_jaguar_chunk(messages_to_compact)


async def generate_summary_from_single_message(content: str) -> str | None:
    """
    Generate a shorter summary of existing summary text.

    Tries Jaguar first, then falls back to the legacy Haiku prompt.
    """
    try:
        messages = [
            {"role": "user", "content": content},
            {"role": "user", "content": _jaguar_compaction_prompt()},
        ]
        return await _call_compaction_model(
            compaction_settings.compaction_model,
            messages,
            compaction_settings.compaction_model_context_window,
        )
    except Exception:
        logger.warning(
            "Jaguar re-summarization failed, falling back to %s",
            compaction_settings.compaction_fallback_model,
        )
        try:
            prompt = build_fallback_compaction_prompt(
                content, compaction_settings.compaction_summary_max_tokens
            )
            return await _call_compaction_model(
                compaction_settings.compaction_fallback_model,
                [{"role": "user", "content": prompt}],
                compaction_settings.compaction_fallback_model_context_window,
            )
        except Exception:
            return None


async def _gather_chunk_summaries(
    chunks: list[list[dict]],
    summarize_chunk,
) -> list[str | Exception]:
    max_parallel = max(1, compaction_settings.compaction_max_parallel_requests)
    semaphore = asyncio.Semaphore(max_parallel)

    async def run(chunk: list[dict]):
        async with semaphore:
            try:
                return await summarize_chunk(chunk)
            except Exception as e:
                return e

    return await asyncio.gather(*(run(chunk) for chunk in chunks))


async def _summarize_with_haiku(
    messages_to_compact: list[dict],
) -> list[tuple[int, str, int]]:
    chunk_limit = _fallback_chunk_token_limit()
    chunks = _chunk_messages_for_compaction(messages_to_compact, chunk_limit)
    results = await _gather_chunk_summaries(chunks, _summarize_haiku_chunk)
    total = len(chunks)
    summaries: list[str] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            raise RuntimeError(  # noqa: TRY004 - runtime failure, not a type error
                f"Haiku compaction chunk {i + 1}/{total} failed"
            ) from result
        summaries.append(result)

    surviving_total = len(summaries)
    return [(i, summary, surviving_total) for i, summary in enumerate(summaries)]


async def _summarize_compaction_chunks(
    messages_to_compact: list[dict],
    chunks: list[list[dict]],
) -> list[tuple[int, str, int]]:
    """
    Summarize chunks with Jaguar. If the final (most recent) chunk fails,
    re-chunk the full history for Haiku. Otherwise skip failed earlier chunks.
    """
    results = await _gather_chunk_summaries(chunks, _summarize_jaguar_chunk)
    if isinstance(results[-1], Exception):
        return await _summarize_with_haiku(messages_to_compact)

    summaries: list[str] = []
    for result in results:
        if isinstance(result, Exception):
            continue
        summaries.append(result)

    surviving_total = len(summaries)
    return [(i, summary, surviving_total) for i, summary in enumerate(summaries)]


async def compact_messages(
    messages: list[dict],
) -> tuple[list[dict], str, list[int]]:
    """
    Perform compaction on OpenAI-format messages.

    Args:
        messages: Original message list

    Returns:
        tuple of (compacted_messages, summary, compacted_indices)
    """
    messages_to_compact, messages_to_preserve, compacted_indices = (
        select_messages_for_compaction(messages)
    )

    if not messages_to_compact:
        return (messages, "", [])

    chunk_limit = _jaguar_chunk_token_limit()
    chunks = _chunk_messages_for_compaction(messages_to_compact, chunk_limit)
    if len(chunks) > 1:
        logger.info(
            "Splitting compaction input into %d chunks (%d tokens, limit %d)",
            len(chunks),
            calculate_message_tokens(messages_to_compact),
            chunk_limit,
        )

    chunk_summaries = await _summarize_compaction_chunks(messages_to_compact, chunks)
    summary_messages = [
        create_summary_message(summary, section=_section_label(i, total))
        for i, summary, total in chunk_summaries
    ]

    combined_summary = "\n\n".join(summary for _, summary, _ in chunk_summaries)

    # Build new message list
    # Structure: [system messages...] + [summary chunk(s)...] + [preserved messages]
    system_messages = [m for m in messages_to_preserve if m.get("role") == "system"]
    non_system_preserved = [
        m for m in messages_to_preserve if m.get("role") != "system"
    ]

    compacted_messages = system_messages + summary_messages + non_system_preserved

    return (compacted_messages, combined_summary, compacted_indices)


async def _resummarize_compacted_messages(
    compacted_messages: list[dict],
    summary: str,
) -> tuple[list[dict], str]:
    """
    Re-summarize an existing summary to make it shorter.

    Args:
        compacted_messages: Current compacted message list
        summary: Current summary text

    Returns:
        tuple of (new_compacted_messages, new_summary) or (original, original) if failed
    """
    summary_messages = [m for m in compacted_messages if _is_summary_message(m)]
    if summary_messages:
        combined = "\n\n".join(_extract_summary_text(m) for m in summary_messages)
        summary = combined or summary

    try:
        shorter_summary = await generate_summary_from_single_message(summary)
    except Exception:
        logger.warning("Re-summarization failed, keeping original summary")
        return (compacted_messages, summary)

    if not shorter_summary:
        return (compacted_messages, summary)

    summary_messages = [m for m in compacted_messages if _is_summary_message(m)]

    # Rebuild message list with a single shorter summary
    summary_message = create_summary_message(shorter_summary)
    system_messages = [
        m
        for m in compacted_messages
        if m.get("role") == "system" and not _is_summary_message(m)
    ]
    other_messages = [
        m
        for m in compacted_messages
        if not _is_summary_message(m) and m.get("role") != "system"
    ]

    return (system_messages + [summary_message] + other_messages, shorter_summary)


async def maybe_compact_conversation(
    messages: list[dict],
    model: str,
    model_config: ModelConfig,
    is_premium: bool,
    skip_checks: bool = False,
) -> tuple[list[dict], CompactionMetadata | None]:
    """
    Check if compaction is needed and perform it if so.

    This is the main entry point for compaction from the API layer.

    Args:
        messages: List of OpenAI message dicts
        model: Model name for token counting
        model_config: Model configuration with token limits
        is_premium: Whether this is a premium user
        skip_checks: If True, skip validation checks (assumes caller already validated)

    Returns:
        tuple of (messages, CompactionMetadata or None)
        - If compaction occurred: (compacted_messages, metadata)
        - If no compaction: (original_messages, None)
    """
    # Calculate token limits
    max_input_tokens, _conversation_max_tokens = calculate_max_input_tokens(
        model_config, is_premium
    )

    if skip_checks:
        # When skip_checks is True, caller claims they've validated, but we still
        # guard against invalid state to prevent downstream errors
        if not can_compact(messages):
            logger.warning(
                "skip_checks=True but can_compact=False, skipping compaction"
            )
            return (messages, None)

        # We still need current_tokens for metrics
        current_tokens = calculate_message_tokens(messages)
        logger.info(
            f"Compaction triggered (pre-validated): {current_tokens} tokens, "
            f"num_messages={len(messages)}"
        )
    else:
        # Check if compaction needed
        needs_compaction, current_tokens, threshold = check_compaction_needed(
            messages=messages,
            max_input_tokens=max_input_tokens,
        )
        logger.info(
            f"Compaction check: {current_tokens} tokens, threshold={threshold}, "
            f"max_input_tokens={max_input_tokens}, needs_compaction={needs_compaction}, "
            f"can_compact={can_compact(messages)}, num_messages={len(messages)}"
        )

        if not needs_compaction or not can_compact(messages):
            return (messages, None)

        logger.info(
            f"Compaction triggered: {current_tokens} tokens >= {threshold} threshold"
        )

    # Perform compaction
    compacted_messages, summary, compacted_indices = await compact_messages(
        messages=messages,
    )
    tokens_after = calculate_message_tokens(compacted_messages)
    logger.info(
        f"Compacted messages: {current_tokens} -> {tokens_after} tokens "
        f"(compacted {len(compacted_indices)} messages)"
    )

    # Trim the preserved tail so that huge recent turns (e.g. a large tool
    # result or assistant response with PDF text) don't survive compaction
    # intact.  We target max_input_tokens so the result is usable for the
    # next LLM call, not just technically within the full context window.
    if tokens_after > max_input_tokens:
        from aichat.serve.services.trimming import (
            trim_messages_to_fit,
        )  # local import avoids circular dependency

        compacted_messages, _, trimmed_tail = trim_messages_to_fit(
            compacted_messages, max_input_tokens
        )
        if trimmed_tail > 0:
            tokens_after = calculate_message_tokens(compacted_messages)
            logger.info(
                f"Trimmed {trimmed_tail} tokens from preserved tail after compaction "
                f"(now {tokens_after} tokens)"
            )

    # Re-summarize if still above max_input_tokens (the 60%-of-limit threshold
    # used for the compaction trigger) rather than the full conversation window,
    # so the result is actually useful for the next turn.
    compacted_messages, summary, tokens_after = await _resummarize_until_within_limit(
        compacted_messages,
        summary,
        tokens_after,
        max_input_tokens,
        compaction_settings.compaction_max_resummarize_attempts,
    )

    # Record successful compaction metric
    COMPACTION_TOTAL.labels(model=model).inc()

    # Record compaction metrics
    COMPACTION_STARTING_TOKEN_COUNT.labels(model=model).observe(
        calculate_message_tokens(messages)
    )
    COMPACTION_ENDING_TOKEN_COUNT.labels(model=model).observe(tokens_after)

    # Create metadata for client notification
    compaction_metadata = CompactionMetadata(
        summary=summary,
        compacted_message_indices=compacted_indices,
        tokens_before=current_tokens,
        tokens_after=tokens_after,
    )

    return (compacted_messages, compaction_metadata)


async def _resummarize_until_within_limit(
    compacted_messages: list[dict],
    summary: str,
    tokens_after: int,
    token_limit: int,
    attempts_remaining: int,
) -> tuple[list[dict], str, int]:
    if tokens_after <= token_limit or attempts_remaining <= 0:
        return compacted_messages, summary, tokens_after

    logger.info(
        "Re-summarizing (attempt %s/%s): %s tokens > %s limit",
        attempts_remaining - 1,
        attempts_remaining,
        tokens_after,
        token_limit,
    )
    compacted_messages, summary = await _resummarize_compacted_messages(
        compacted_messages, summary
    )
    tokens_after = calculate_message_tokens(compacted_messages)
    return await _resummarize_until_within_limit(
        compacted_messages,
        summary,
        tokens_after,
        token_limit,
        attempts_remaining - 1,
    )


def create_compaction_event(event_type: str, data: dict | None = None) -> str:
    """
    Create an SSE-formatted compaction event.

    Args:
        event_type: The type of compaction event (e.g., 'compaction_starting', 'compaction_failed')
        data: Optional additional data to include in the event

    Returns:
        SSE-formatted event string
    """
    event = {"type": event_type}
    if data:
        event.update(data)
    return f"data: {json.dumps(event)}\n\n"


# Capabilities for which compaction is disabled.
# content_agent is blocked because compaction may degrade task performance.
COMPACTION_BLOCKED_CAPABILITIES: list[Capability] = [
    Capability.content_agent,
]


def should_compact(
    messages: list[dict],
    model_config: ModelConfig,
    is_premium: bool,
    is_streaming: bool,
    capability: CapabilityOptions = None,
) -> bool:
    """
    Check if compaction will be needed for the given messages without performing it.

    Args:
        messages: List of OpenAI message dicts
        model_config: Model configuration with token limits
        is_premium: Whether this is a premium user
        is_streaming: Whether this is a streaming request
        capability: The brave_capability of the request, if any

    Returns:
        True if compaction will be needed (only for streaming requests)
    """
    # Only pre-check compaction for streaming requests
    if not is_streaming:
        return False

    if not is_premium:
        return False

    if any(
        has_capability(capability, blocked)
        for blocked in COMPACTION_BLOCKED_CAPABILITIES
    ):
        return False

    max_input_tokens, _ = calculate_max_input_tokens(model_config, is_premium)
    needs_compaction, _, _ = check_compaction_needed(messages, max_input_tokens)
    return needs_compaction and can_compact(messages)
