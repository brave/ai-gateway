import logging
from hashlib import sha256

from aichat.protocol.open_ai_protocol import (
    AssistantMessage,
    FileContentPart,
    FileUrlContentPart,
    ImageContentPart,
    MessageUnion,
    TextContentPart,
    ToolMessage,
)
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.services.models import ModelConfig

logger = logging.getLogger(__name__)


def _get_image_limit(model_config: ModelConfig) -> int | None:
    """Return the maximum image count for the given model, or None if there is no limit."""
    if model_config.backend in ("bedrock", "bedrock_mantle"):
        return external_service_settings.bedrock_image_limit
    return external_service_settings.global_image_limit


def _get_document_limit(model_config: ModelConfig) -> int | None:
    """Return the maximum document count for the given model, or None if there is no limit."""
    if model_config.backend in ("bedrock", "bedrock_mantle"):
        return external_service_settings.bedrock_document_limit
    return None


def _short_hash(data: str) -> str:
    return sha256(data.encode()).hexdigest()[:8]


def preprocess_messages(
    messages: list[MessageUnion], model_config: ModelConfig
) -> list[MessageUnion]:
    """
    Preprocess messages before they reach the model:
      1. Drop the oldest images when the total count exceeds the per-model limit.
      2. Drop the oldest documents when the total count exceeds the per-model limit.
      3. Drop older duplicates when multiple attachments share identical content
         (Bedrock only: LiteLLM names Bedrock documents by content hash, so identical
         content always produces the same name regardless of filename).
      4. Rename duplicate file attachment names to '{name}-{counter}-{hash}'.
      5. Strip tool-related messages when the model does not support tools.
      6. Strip file parts when the model does not support files.
    """
    messages = _enforce_image_limit(messages, model_config)
    messages = _enforce_document_limit(messages, model_config)
    if model_config.backend == "bedrock":
        messages = _deduplicate_attachment_content(messages)
    messages = _deduplicate_attachment_names(messages)
    if not model_config.tool_support:
        messages = _strip_tool_messages(messages, model_config)
    if not model_config.file_support:
        messages = _strip_file_parts(messages, model_config)
    if not model_config.image_support:
        messages = _strip_image_parts(messages)
    return messages


def _enforce_image_limit(
    messages: list[MessageUnion], model_config: ModelConfig
) -> list[MessageUnion]:
    limit = _get_image_limit(model_config)
    if limit is None:
        return messages

    # Collect (message_index, part_index) for every ImageContentPart, oldest-first.
    image_positions: list[tuple[int, int]] = []
    for msg_idx, message in enumerate(messages):
        if isinstance(message.content, list):
            for part_idx, part in enumerate(message.content):
                if isinstance(part, ImageContentPart):
                    image_positions.append((msg_idx, part_idx))

    excess = len(image_positions) - limit
    if excess <= 0:
        return messages

    logger.info(
        "Removing %d oldest image(s) to stay within limit of %d for model %s",
        excess,
        limit,
        model_config.model_id,
    )

    to_remove: set[tuple[int, int]] = set(image_positions[:excess])
    result = list(messages)

    for msg_idx, message in enumerate(result):
        if not isinstance(message.content, list):
            continue
        new_content = [
            part
            for part_idx, part in enumerate(message.content)
            if (msg_idx, part_idx) not in to_remove
        ]
        if len(new_content) != len(message.content):
            result[msg_idx] = message.model_copy(update={"content": new_content})

    return result


def _enforce_document_limit(
    messages: list[MessageUnion], model_config: ModelConfig
) -> list[MessageUnion]:
    limit = _get_document_limit(model_config)
    if limit is None:
        return messages

    # Collect (message_index, part_index) for every document part, oldest-first.
    doc_positions: list[tuple[int, int]] = []
    for msg_idx, message in enumerate(messages):
        if isinstance(message.content, list):
            for part_idx, part in enumerate(message.content):
                if isinstance(part, (FileContentPart, FileUrlContentPart)):
                    doc_positions.append((msg_idx, part_idx))

    excess = len(doc_positions) - limit
    if excess <= 0:
        return messages

    logger.warning(
        "Removing %d oldest document(s) to stay within limit of %d for model %s",
        excess,
        limit,
        model_config.model_id,
    )

    to_remove: set[tuple[int, int]] = set(doc_positions[:excess])
    result = list(messages)

    for msg_idx, message in enumerate(result):
        if not isinstance(message.content, list):
            continue
        new_content = [
            part
            for part_idx, part in enumerate(message.content)
            if (msg_idx, part_idx) not in to_remove
        ]
        if len(new_content) != len(message.content):
            result[msg_idx] = message.model_copy(update={"content": new_content})

    return result


def _deduplicate_attachment_content(messages: list[MessageUnion]) -> list[MessageUnion]:
    """
    Remove older FileContentPart attachments that share the same file_data as a
    later attachment.

    LiteLLM names Bedrock document blocks by hashing the file content, so two
    attachments with identical bytes always produce the same Bedrock document name
    regardless of their filename.  Bedrock then rejects the request with
    "Messages can't contain duplicate document names."

    We scan newest-first and keep only the last occurrence of each unique
    file_data, dropping the earlier (stale) copies.
    """
    # Scan newest-first to determine which positions are superseded duplicates.
    # Store a short hash of file_data rather than the raw bytes to avoid keeping
    # potentially large base64 strings in memory.
    seen_content: set[str] = set()
    to_remove: set[tuple[int, int]] = set()

    for msg_idx in range(len(messages) - 1, -1, -1):
        message = messages[msg_idx]
        if not isinstance(message.content, list):
            continue
        for part_idx in range(len(message.content) - 1, -1, -1):
            part = message.content[part_idx]
            if not isinstance(part, FileContentPart):
                continue
            content_key = _short_hash(part.file.file_data)
            if content_key in seen_content:
                to_remove.add((msg_idx, part_idx))
            else:
                seen_content.add(content_key)

    if not to_remove:
        return messages

    logger.info(
        "Removing %d duplicate document(s) with identical content for Bedrock",
        len(to_remove),
    )

    result = list(messages)
    for msg_idx, message in enumerate(result):
        if not isinstance(message.content, list):
            continue
        new_content = [
            part
            for part_idx, part in enumerate(message.content)
            if (msg_idx, part_idx) not in to_remove
        ]
        if len(new_content) != len(message.content):
            # new_content may be empty if the message contained only duplicate
            # file parts; that is valid and intentional.
            result[msg_idx] = message.model_copy(update={"content": new_content})

    return result


def _deduplicate_attachment_names(messages: list[MessageUnion]) -> list[MessageUnion]:
    """
    When multiple FileContentPart attachments share the same filename, rename every
    duplicate (2nd, 3rd, …) to '{original_name}-{occurrence}-{short_hash}'.
    The first occurrence keeps its original name.
    """
    # Identify filenames that appear more than once across all messages.
    name_counts: dict[str, int] = {}
    for message in messages:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            if isinstance(part, FileContentPart):
                name_counts[part.file.filename] = (
                    name_counts.get(part.file.filename, 0) + 1
                )

    duplicate_names = {name for name, count in name_counts.items() if count > 1}
    if not duplicate_names:
        return messages

    logger.info("Deduplicating attachment filenames: %s", duplicate_names)

    name_occurrence: dict[str, int] = {}
    result = list(messages)

    for msg_idx, message in enumerate(result):
        if not isinstance(message.content, list):
            continue

        new_content = list(message.content)
        changed = False

        for part_idx, part in enumerate(new_content):
            if not isinstance(part, FileContentPart):
                continue
            filename = part.file.filename
            if filename not in duplicate_names:
                continue

            occurrence = name_occurrence.get(filename, 0)
            name_occurrence[filename] = occurrence + 1

            if occurrence == 0:
                # The first occurrence keeps its original name.
                continue

            file_hash = _short_hash(part.file.file_data or filename)
            new_filename = f"{filename}-{occurrence}-{file_hash}"
            new_file = part.file.model_copy(update={"filename": new_filename})
            new_content[part_idx] = part.model_copy(update={"file": new_file})
            changed = True

        if changed:
            result[msg_idx] = message.model_copy(update={"content": new_content})

    return result


def _strip_file_parts(
    messages: list[MessageUnion], model_config: ModelConfig
) -> list[MessageUnion]:
    """
    Replace all FileContentPart and FileUrlContentPart entries with a text
    note for models that do not support file attachments.

    Replacing rather than silently dropping gives the model enough context to
    inform the user that their file could not be processed.
    """
    result = list(messages)
    replaced = 0

    for msg_idx, message in enumerate(result):
        if not isinstance(message.content, list):
            continue
        new_content = []
        for part in message.content:
            if isinstance(part, (FileContentPart, FileUrlContentPart)):
                new_content.append(
                    TextContentPart(
                        type="text",
                        text=f"[File attachment '{part.file.filename}' was removed: this model does not support file attachments.]",
                    )
                )
                replaced += 1
            else:
                new_content.append(part)
        if replaced:
            result[msg_idx] = message.model_copy(update={"content": new_content})

    if replaced:
        logger.info(
            "Replaced %d file part(s) with text notes for model %s (file_support=False)",
            replaced,
            model_config.model_id,
        )

    return result


def _strip_image_parts(
    messages: list[MessageUnion],
) -> list[MessageUnion]:
    """
    Replace ImageContentPart entries with a text note for models that do not
    support image attachments.

    Replacing rather than silently dropping gives the model enough context to
    inform the user that their image could not be processed.
    """
    result = list(messages)
    replaced = 0

    for msg_idx, message in enumerate(result):
        if not isinstance(message.content, list):
            continue
        new_content = []
        changed = False
        for part in message.content:
            if isinstance(part, ImageContentPart):
                new_content.append(
                    TextContentPart(
                        type="text",
                        text="[Image attachment was removed: this model does not support image attachments.]",
                    )
                )
                replaced += 1
                changed = True
            else:
                new_content.append(part)
        if changed:
            result[msg_idx] = message.model_copy(update={"content": new_content})

    return result


def _strip_tool_messages(
    messages: list[MessageUnion], model_config: ModelConfig
) -> list[MessageUnion]:
    """
    Remove tool-related messages for models that do not support tool usage.

    Specifically:
    - Drop all ToolMessage entries (role="tool" / tool results).
    - Strip tool_calls from AssistantMessage entries; drop the message entirely
      if it has no remaining content after stripping.
    """
    result: list[MessageUnion] = []
    removed = 0

    for message in messages:
        if isinstance(message, ToolMessage):
            removed += 1
            continue

        if isinstance(message, AssistantMessage) and message.tool_calls:
            if message.content is not None:
                result.append(message.model_copy(update={"tool_calls": None}))
            else:
                removed += 1
            continue

        result.append(message)

    if removed:
        logger.info(
            "Stripped %d tool-related message(s) for model %s (tool_support=False)",
            removed,
            model_config.model_id,
        )

    return result
