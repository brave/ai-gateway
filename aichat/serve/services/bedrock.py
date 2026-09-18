import hashlib
import logging
import re
from json import JSONDecodeError, loads

from aichat.protocol.open_ai_protocol import Tool
from aichat.serve.constants import CONTENT_FILTER_MESSAGE
from aichat.serve.services.bedrock_settings import bedrock_settings
from aichat.serve.utils import get_token_count_estimate

logger = logging.getLogger(__name__)


# Bedrock requires tool_use.id to match this pattern. Any other characters
# (e.g. dots, colons) cause a BadRequestError from the Converse API.
_BEDROCK_TOOL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_BEDROCK_TOOL_ID_DISALLOWED = re.compile(r"[^a-zA-Z0-9_-]")

# Bedrock also caps tool_use.name and tool_use.id at 64 chars (e.g. WebMCP
# tool names that embed a site URL can exceed this).
_BEDROCK_MAX_LEN = 64
_BEDROCK_HASH_LEN = 8


def _enforce_bedrock_max_length(value: str) -> str:
    """Truncate to Bedrock's 64-char limit, keeping a hash suffix for uniqueness.

    Deterministic in the original value, so independently sanitizing a
    tool_calls[].id and its paired tool_call_id still yields matching ids.
    """
    if len(value) <= _BEDROCK_MAX_LEN:
        return value
    digest = hashlib.sha256(value.encode()).hexdigest()[:_BEDROCK_HASH_LEN]
    prefix_len = _BEDROCK_MAX_LEN - _BEDROCK_HASH_LEN - 1
    return f"{value[:prefix_len]}_{digest}"


def sanitize_tool_call_id_for_bedrock(tool_call_id: str | None) -> str | None:
    """Rewrite a tool_use.id to satisfy Bedrock's charset and length constraints.

    Returns the input unchanged when it is falsy or already valid, so
    sanitization is a no-op for IDs that satisfy ``^[a-zA-Z0-9_-]+$`` and are
    at most 64 characters.
    """
    if not tool_call_id or not isinstance(tool_call_id, str):
        return tool_call_id
    sanitized = (
        tool_call_id
        if _BEDROCK_TOOL_ID_PATTERN.match(tool_call_id)
        else _BEDROCK_TOOL_ID_DISALLOWED.sub("_", tool_call_id)
    )
    return _enforce_bedrock_max_length(sanitized)


def sanitize_tool_name_for_bedrock(tool_name: str | None) -> str | None:
    """Rewrite a toolUse.name to satisfy Bedrock's charset and length constraints.

    Bedrock enforces the same ``[a-zA-Z0-9_-]+`` constraint and 64-char limit
    on tool names as on tool_use IDs. Returns the input unchanged when it is
    falsy or already valid.
    """
    if not tool_name or not isinstance(tool_name, str):
        return tool_name
    sanitized = (
        tool_name
        if _BEDROCK_TOOL_ID_PATTERN.match(tool_name)
        else _BEDROCK_TOOL_ID_DISALLOWED.sub("_", tool_name)
    )
    return _enforce_bedrock_max_length(sanitized)


def completion_content_for_finish_reason(
    finish_reason: str | None, content: str | None
) -> str | None:
    """
    Return the content to use for a completion. When the model returns
    content_filter (e.g. Bedrock safety filter), returns a user-facing message
    instead of empty/truncated content.
    """
    if finish_reason == "content_filter":
        return CONTENT_FILTER_MESSAGE
    return content


def apply_content_filter_to_completion_response(response: dict) -> None:
    """
    When the model returns content_filter (e.g. Bedrock safety filter), replace
    the completion message content with a user-facing message. Mutates
    response in place. No-op if response is not a dict or lacks choices.
    """
    if not isinstance(response, dict) or not response.get("choices"):
        return
    choice = response["choices"][0]
    if choice.get("finish_reason") != "content_filter":
        return
    if choice.get("message") is not None:
        choice["message"] = dict(choice["message"])
    else:
        choice["message"] = {}
    choice["message"]["content"] = CONTENT_FILTER_MESSAGE


def map_tool_role_to_assistant(messages: list[dict]) -> list[dict]:
    """For Bedrock models that require it: change message role 'tool' -> 'assistant'."""
    if not messages:
        return messages
    return [
        {**msg, "role": "assistant"} if msg.get("role") == "tool" else msg
        for msg in messages
    ]


def _assistant_content_is_nonempty(content) -> bool:
    """True when an assistant message carries meaningful text content."""
    if not content:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if text and str(text).strip():
                    return True
            elif part:
                return True
        return False
    return True


def split_assistant_content_with_tool_calls(messages: list[dict]) -> list[dict]:
    """
    Split assistant turns holding both content and tool_calls into the canonical
    order: assistant{tool_calls} -> tool{result} -> assistant{content}.

    Streaming tools like deep_research stream their answer as assistant content,
    so the turn gets persisted as a single message with both the answer and the
    tool_call. That self-contradictory shape makes some Bedrock models (notably
    Kimi-k2.5) return empty completions on later turns.

    Only splits assistant messages immediately followed by a tool result; dangling
    tool calls are left untouched.
    """
    if not messages:
        return messages

    result: list[dict] = []
    i = 0
    n = len(messages)
    while i < n:
        message = messages[i]
        if (
            message.get("role") == "assistant"
            and message.get("tool_calls")
            and _assistant_content_is_nonempty(message.get("content"))
            and i + 1 < n
            and messages[i + 1].get("role") == "tool"
        ):
            result.append({k: v for k, v in message.items() if k != "content"})
            i += 1
            while i < n and messages[i].get("role") == "tool":
                result.append(messages[i])
                i += 1
            result.append({"role": "assistant", "content": message.get("content")})
        else:
            result.append(message)
            i += 1
    return result


def _merge_split_tool_calls(
    tool_calls: list[dict], model_id: str | None = None
) -> list[dict]:
    """
    Recover tool calls that an upstream model split across multiple entries.

    vLLM's Qwen tool-call streaming sometimes emits a continuation entry with
    an empty `function.name` and empty `id` whose `arguments` are the rest of
    the previous entry's JSON. Concatenate those onto the prior tool call and
    re-validate; keep the merged result if it parses, otherwise drop the
    continuation.
    """
    merged: list[dict] = []

    for tool_call in tool_calls:
        function_def = tool_call.get("function") or {}
        name = function_def.get("name") or ""
        tc_id = tool_call.get("id") or ""
        is_continuation = not name and not tc_id and merged

        if not is_continuation:
            merged.append(tool_call)
            continue

        prev = merged[-1]
        prev_function = prev.get("function") or {}
        prev_args = prev_function.get("arguments") or ""
        cont_args = function_def.get("arguments") or ""

        if not isinstance(prev_args, str) or not isinstance(cont_args, str):
            logger.warning(
                "Dropping non-string continuation arguments for tool_call_id="
                f"'{prev.get('id')}'"
            )
            continue

        candidate = prev_args + cont_args
        try:
            loads(candidate)
        except (JSONDecodeError, ValueError):
            logger.warning(
                "Dropping unrecoverable split tool-call continuation for "
                f"tool_call_id='{prev.get('id')}'"
            )
            continue

        new_function = {**prev_function, "arguments": candidate}
        merged[-1] = {**prev, "function": new_function}

    return merged


def filter_tool_call_result_pairing(
    messages: list[dict], model_id: str | None = None
) -> list[dict]:
    """
    Validate that all tool results have corresponding valid tool calls in the conversation history.
    Filters out:
    - Tool calls with invalid JSON in their arguments
    - Tool results for invalid/malformed tool calls
    - Orphaned tool results that have no matching tool call

    Also runs a pre-pass to merge "split" tool calls (continuation entries
    with empty name/id) before JSON validation. The optional model_id is used
    purely as the metric label when malformed tool calls are merged or dropped.
    """
    try:
        if not messages:
            return messages

        cleaned_messages = []
        valid_tool_call_ids = set()

        for message in messages:
            role = message.get("role")

            # Track tool call IDs from assistant messages, validating JSON
            if role == "assistant" and message.get("tool_calls"):
                valid_tool_calls = []
                valid_tool_call_ids = set()

                pre_merged_tool_calls = _merge_split_tool_calls(
                    message.get("tool_calls", []), model_id=model_id
                )

                for tool_call in pre_merged_tool_calls:
                    tool_call_id = tool_call.get("id")
                    if not tool_call_id:
                        continue

                    sanitized_id = sanitize_tool_call_id_for_bedrock(tool_call_id)
                    if sanitized_id != tool_call_id:
                        tool_call = {**tool_call, "id": sanitized_id}
                        tool_call_id = sanitized_id

                    function_def = tool_call.get("function", {})
                    fn_name = (
                        function_def.get("name")
                        if isinstance(function_def, dict)
                        else None
                    )
                    sanitized_name = sanitize_tool_name_for_bedrock(fn_name)
                    if sanitized_name != fn_name:
                        function_def = {**function_def, "name": sanitized_name}
                        tool_call = {**tool_call, "function": function_def}

                    arguments = function_def.get("arguments", "{}")

                    try:
                        if isinstance(arguments, str):
                            loads(arguments)
                        valid_tool_calls.append(tool_call)
                        valid_tool_call_ids.add(tool_call_id)
                    except (JSONDecodeError, ValueError):
                        logger.warning(
                            f"Removing tool call with tool_call_id='{tool_call_id}' "
                            f"due to invalid JSON in arguments"
                        )

                if valid_tool_calls:
                    cleaned_messages.append({**message, "tool_calls": valid_tool_calls})
                elif message.get("content") is not None:
                    cleaned_messages.append(
                        {k: v for k, v in message.items() if k != "tool_calls"}
                    )

            elif role == "tool":
                tool_call_id = message.get("tool_call_id")
                sanitized_id = sanitize_tool_call_id_for_bedrock(tool_call_id)
                if sanitized_id != tool_call_id:
                    message = {**message, "tool_call_id": sanitized_id}
                    tool_call_id = sanitized_id
                if tool_call_id in valid_tool_call_ids:
                    cleaned_messages.append(message)
                else:
                    logger.warning(
                        f"Removing tool result with tool_call_id='{tool_call_id}' "
                    )
            else:
                cleaned_messages.append(message)

        return cleaned_messages
    except Exception:
        logger.error(
            "Error validating tool call/result pairing. Returning original messages."
        )
        return messages


def format_tools_for_bedrock(tools: list[Tool]) -> list[Tool]:
    """
    Bedrock rejects tools without explicit additionalProperties in their
    parameter schema. This function ensures all tools have this field set.
    """
    formatted_tools = []

    for tool in tools:
        tool_dict = tool.model_dump()
        # All tools are expected to be in OpenAI format with type="function"
        function_def = tool_dict["function"]
        if not isinstance(function_def, dict):
            if hasattr(function_def, "model_dump"):
                function_def = function_def.model_dump()
            else:
                function_def = dict(function_def)
        fn_name = function_def.get("name")
        sanitized_name = sanitize_tool_name_for_bedrock(fn_name)
        if sanitized_name != fn_name:
            function_def["name"] = sanitized_name
            tool_dict["function"] = function_def

        params = function_def.get("parameters")

        # Bedrock requires all parameter schemas to have explicit 'additionalProperties',
        # 'type', and 'properties' fields.
        if not isinstance(params, dict):
            function_def["parameters"] = {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }
        else:
            # tool_dict is already a copy, so we can modify params directly
            if "type" not in params:
                params["type"] = "object"
            if "properties" not in params:
                params["properties"] = {}
            # Always set additionalProperties to True for Bedrock compatibility
            params["additionalProperties"] = True

        formatted_tools.append(tool_dict)

    return formatted_tools


# Bedrock/Anthropic allow at most 4 cache checkpoints per request.
MAX_CACHE_POINTS = 4


_EPHEMERAL = {"type": "ephemeral"}

# Roles cached via a separate role-targeted checkpoint (the system prompt).
# We never place an index-based history checkpoint on these — doing so would
# duplicate the role point and waste one of the four cache checkpoints.
_ROLE_CACHED_SEPARATELY = {"system", "developer"}


def get_cache_control_injection_points(
    messages, cache_system_prompt: bool = True, reserved_points: int = 0
):
    """
    Build ``cache_control_injection_points`` for Bedrock prompt caching.

    Prompt caching on Bedrock is *prefix*-based: a cache checkpoint caches
    everything before it (tools -> system -> messages, in that order). We place
    checkpoints at stable boundaries so each new turn reuses the largest
    possible prefix from the previous request:

      1. The system prompt (caches the tools+system prefix), if a system
         message is present and ``cache_system_prompt`` is set. litellm has no
         "system" location, but tagging the system message by role makes the
         Bedrock transform emit a system ``cachePoint``.
      2. Up to two conversation-history prefixes (see ``get_prefix_cache_indices``):
         a stable early anchor and the latest boundary, which together give
         strong turn-over-turn reuse.

    ``reserved_points`` lets the caller reserve checkpoints for cache points
    placed outside this function (e.g. the tools checkpoint set directly on the
    tools list), so the total never exceeds ``MAX_CACHE_POINTS``.
    """
    budget = MAX_CACHE_POINTS - reserved_points
    if budget <= 0:
        return []

    injection_points = []

    if cache_system_prompt and _has_system_message(messages):
        injection_points.append(
            {"location": "message", "role": "system", "control": dict(_EPHEMERAL)}
        )
        budget -= 1

    for idx in get_prefix_cache_indices(messages, max_count=budget):
        injection_points.append(
            {"location": "message", "index": idx, "control": dict(_EPHEMERAL)}
        )

    return injection_points


def _has_system_message(messages) -> bool:
    # Require content: a content-less system message produces no content block,
    # so a role-targeted checkpoint on it would be ineffective and waste budget.
    return any(
        message.get("role") == "system" and message.get("content")
        for message in messages or []
    )


def get_prefix_cache_indices(messages, max_count: int = 2) -> list[int]:
    """
    Indices at which to place conversation-history cache checkpoints, based on
    *cumulative* prefix token counts rather than individual message size.

    Walking from the start and summing tokens, a message is "eligible" once the
    running prefix total reaches ``bedrock_settings.bedrock_cache_min_tokens`` (so the cached prefix
    clears Bedrock's minimum). We return, within ``max_count``:

      - the earliest eligible index (a stable anchor that rarely moves), and
      - the latest eligible index (advances as the conversation grows, so the
        whole prior conversation is a cache hit on the next turn).

    Only messages with content are considered, since a checkpoint can only be
    attached to a message that produces a content block.
    """
    if max_count <= 0 or not messages:
        return []

    cumulative = 0
    eligible: list[int] = []
    for i, message in enumerate(messages):
        content = message.get("content")
        if not content:
            continue
        # The system/developer prompt is physically part of the cached prefix,
        # so it still counts toward the cumulative total...
        cumulative += get_token_count_estimate(content)
        # ...but it is cached separately via a role-targeted checkpoint, so we
        # never place an index-based history checkpoint on it.
        if message.get("role") in _ROLE_CACHED_SEPARATELY:
            continue
        if cumulative >= bedrock_settings.bedrock_cache_min_tokens:
            eligible.append(i)

    if not eligible:
        return []

    indices = [eligible[0]]
    if max_count >= 2 and eligible[-1] != eligible[0]:
        indices.append(eligible[-1])
    return indices[:max_count]


def add_cache_control_to_tools(tools: list) -> list:
    """
    Tag the last tool with ``cache_control`` so the litellm Bedrock transform
    emits a ``cachePoint`` after the tool definitions, caching the entire tools
    prefix. Tools are identical on every request, so this is a per-request
    saving whenever tools are sent.

    ``cache_control_injection_points`` cannot target tools (litellm only honors
    the "message" location), so the checkpoint must be set on the tool dict
    directly.

    Returns a new list with a copy of the last tool tagged; the input list and
    its tool dicts are left unmodified, so shared tool definitions never leak a
    ``cache_control`` key across requests. Returns ``tools`` unchanged when it is
    empty or the last entry is not a dict.
    """
    if not tools:
        return tools
    last = tools[-1]
    if not isinstance(last, dict):
        return tools
    tagged_last = {**last, "cache_control": dict(_EPHEMERAL)}
    return [*tools[:-1], tagged_last]
