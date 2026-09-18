import asyncio
import ctypes
import ctypes.util
import functools
import logging
import os
import time
from typing import Any

import shortuuid
import tiktoken
from fastapi.responses import StreamingResponse

from aichat.llm.llm_settings import llm_settings
from aichat.protocol.anthropic_api_protocol import CompletionStreamResponse
from aichat.serve.conversation_settings import conversation_settings
from aichat.serve.services.model_settings import model_settings

SYSTEM_ROLE = "system"
ASSISTANT_ROLE = "assistant"
USER_ROLE = "user"
TOOL_ROLE = "tool"

MEDIA_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]

NON_TEXT_TOKEN_ESTIMATES = {
    "file": conversation_settings.file_token_estimate,
    "file_url": conversation_settings.file_token_estimate,
    "image_url": conversation_settings.image_token_estimate,
    "input_audio": conversation_settings.audio_token_estimate,
    "video_url": conversation_settings.video_token_estimate,
}

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = tiktoken.get_encoding(llm_settings.default_tokenizer)


@functools.cache
def _get_upstream_to_model_id_mapping() -> dict[str, str]:
    """
    Create a mapping from upstream model names to model IDs.
    This allows us to map LiteLLM's returned model names back to our friendly model IDs.
    """
    mapping = {}
    for model_id, config in model_settings.models.items():
        if config.get("type", "llm") != "llm":
            continue

        if config.get("backend") == "bedrock":
            inference_profile = config.get("inference_profile")
            profile_value = os.getenv(inference_profile) if inference_profile else None
            if profile_value:
                mapping[profile_value] = model_id
        else:
            upstream_model = config.get("upstream_model")
            if upstream_model:
                mapping[upstream_model] = model_id
    return mapping


def normalize_model_name(litellm_model: str) -> str:
    """
    Convert LiteLLM's model string to our friendly model ID.

    LiteLLM returns models like "Qwen/Qwen3-14B" or "hosted_vllm/Qwen/Qwen3-14B",
    and we want to map them back to friendly names like "qwen-14b-instruct".
    """
    if not litellm_model:
        return None

    mapping = _get_upstream_to_model_id_mapping()

    if litellm_model in mapping:
        return mapping[litellm_model]

    if "/" in litellm_model:
        parts = litellm_model.split("/", 1)
        if len(parts) > 1 and parts[1] in mapping:
            return mapping[parts[1]]

    model_name = litellm_model.split("/")[-1]
    if model_name in mapping:
        return mapping[model_name]

        return model_name


def parse_last_user_input(text: str) -> str:
    text = text.strip()

    # Constants
    INST_START = "[INST]"
    INST_END = "[/INST]"
    TRANSCRIPT_END = "</transcript>\n"
    SYS_END = "<</SYS>>\n"
    ARTICLE_END = "</article>\n"
    AI_RESPONSE_SEED = "Here is your response:"

    # Find the last occurrence of INST_END (the last user input will end just before this)
    end_index = text.rfind(INST_END)

    # If there's any content after INST_END, return an empty string.
    after_inst_end_to_end = text[end_index + len(INST_END) :]
    if after_inst_end_to_end.strip() != AI_RESPONSE_SEED:
        # If AI_RESPONSE_SEED is not after INST_END, then this must be the prompt
        # that generates suggested questions.
        return ""

    # The last user input will be between the last instance of INST_END and
    # the last instance of either:
    # - SYS_END - If it's the first message on a page without an article or video
    # - ARTICLE_END - If it's the first message on a page with an article
    # - TRANSCRIPT_END - If it's the first message on a page with a video
    # - INST_START - If it's the second or greater message in a conversation on any page

    # Find the latest occurence among these potential start indexes for last user input.
    start_inst_index = text.rfind(INST_START) + len(INST_START) + 1
    sys_end_index = text.rfind(SYS_END) + len(SYS_END) + 1
    query_text_index = text.rfind(ARTICLE_END) + len(ARTICLE_END) + 1
    transcript_end_index = text.rfind(TRANSCRIPT_END) + len(TRANSCRIPT_END) + 1

    start_index = max(
        start_inst_index, sys_end_index, query_text_index, transcript_end_index
    )
    if start_index == -1:
        return ""

    # Return the substring between the start index we've found up to the last occurence
    # of INST_END.
    return text[start_index:end_index].strip()


async def static_content_generator(static_content: str, delay: float = 0.05):
    """
    Generates a stream of static content in chunks with a delay between each chunk.
    The stream contains just the content, and is not formatted as a JSON SSE response.

    E.g.  "Hello, this is a test" -> "Hello", "this ", "is a ", "test"
    """
    chunk_size = 5
    current_size = 0

    while True:
        partial_content = static_content[current_size : current_size + chunk_size]
        current_size += chunk_size
        finished = current_size > len(static_content)
        yield partial_content
        if finished:
            return
        await asyncio.sleep(delay)


async def generate_static_content_generator(
    model_name: str, static_content: str, delay: float = 0.05
):
    """
    Generates a stream valid SSE responses from static content.

    E.g. "Hello, this is a test" ->
    data: {"completion": "Hello", "model": "model_name", "stop_reason": null, "stop": null, "truncated": false, "log_id": "cmpl-uuid", "exception": null}
    data: {"completion": "this ", "model": "model_name", "stop_reason": null, "stop": null, "truncated": false, "log_id": "cmpl-uuid", "exception": null}
    data: {"completion": "is a ", "model": "model_name", "stop_reason": null, "stop": null, "truncated": false, "log_id": "cmpl-uuid", "exception": null}
    data: {"completion": "test", "model": "model_name", "stop_reason": null, "stop": null, "truncated": false, "log_id": "cmpl-uuid", "exception": null}
    data: [DONE]
    """
    generator = static_content_generator(static_content, delay)
    text = ""
    async for partial_content in generator:
        text += partial_content
        chunk = CompletionStreamResponse(
            completion=text,
            model=model_name,
            stop_reason=None,
            stop=None,
            truncated=False,
            log_id=f"cmpl-{shortuuid.random()}",
            exception=None,
        )

        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

    yield "data: [DONE]\n\n"


def parse_free_model_names(models: dict[str, Any]):
    free_models = []
    for model_name, attributes in models.items():
        if attributes.get("type", "llm") != "llm":
            continue
        if attributes.get("free", False):
            free_models.append(model_name)

    return free_models


def get_real_ip(x_forwarded_for: str) -> str:
    # Split the x_forwarded_for string and take the last IP address
    return x_forwarded_for.split(",")[-1].strip() if x_forwarded_for else "UNKNOWN"


def get_rate_limiting_key(model: str, rate_key: str, interval_in_seconds: int):
    return f"{model}:{rate_key}:{int(time.time() // interval_in_seconds)}"


async def _sse_generator(generator):
    try:
        async for data in generator:
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        if hasattr(generator, "aclose"):
            await generator.aclose()


def SSEResponse(generator, headers=None):
    return StreamingResponse(
        _sse_generator(generator), media_type="text/event-stream", headers=headers
    )


def count_tokens(messages: list[dict]) -> int:
    """
    Count tokens in OpenAI-format messages using tiktoken.

    Args:
        messages: List of OpenAI message dicts

    Returns:
        Token count
    """
    try:
        # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo compatible)
        encoding = tiktoken.get_encoding("cl100k_base")

        total_tokens = 0
        for message in messages:
            # Count message overhead tokens (role, etc.)
            total_tokens += 4  # every message has <|start|>role<|end|> overhead
            for value in message.values():
                if isinstance(value, str):
                    total_tokens += len(encoding.encode(value))
                elif isinstance(value, list):
                    # Handle multimodal content
                    for part in value:
                        if isinstance(part, dict) and part.get("type") == "text":
                            total_tokens += len(encoding.encode(part.get("text", "")))
        total_tokens += 2  # every reply is primed with <|start|>assistant
        return total_tokens
    except Exception:
        # Fallback: estimate ~4 chars per token
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4


def calculate_message_tokens(messages: list[dict]) -> int:
    """
    Count tokens in OpenAI-format messages, including all content part types.

    Unlike count_tokens(), this counts tokens from all content part types
    (brave-page-text, brave-search-results, etc.) not just type=="text" parts,
    which is necessary to accurately measure how much trimming is needed and to
    give the compaction trigger a true picture of conversation size.

    Args:
        messages: List of OpenAI message dicts

    Returns:
        Total token count
    """
    total_tokens = 0
    for message in messages:
        total_tokens += 4  # rough estimate for message structure overhead

        content = message.get("content")
        if isinstance(content, str):
            total_tokens += len(DEFAULT_TOKENIZER.encode(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type in NON_TEXT_TOKEN_ESTIMATES:
                        total_tokens += NON_TEXT_TOKEN_ESTIMATES[part_type]
                        continue
                    text = part.get("text") or part.get("content")
                    if text:
                        total_tokens += len(DEFAULT_TOKENIZER.encode(text))

        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                if isinstance(tool_call, dict):
                    total_tokens += len(
                        DEFAULT_TOKENIZER.encode(str(tool_call.get("function", {})))
                    )

    return total_tokens


def get_malloc_trim():
    """Load glibc's malloc_trim if available (Linux only).
    malloc_trim(0) tells glibc to release free memory back to the OS,
    combating RSS growth from heap fragmentation in long-running processes."""
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        return None
    try:
        libc = ctypes.CDLL(libc_name, use_errno=True)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim.argtypes = [ctypes.c_size_t]
            libc.malloc_trim.restype = ctypes.c_int
            return libc.malloc_trim
    except OSError:
        pass
    return None


_malloc_trim = get_malloc_trim()


async def periodic_malloc_trim(interval: int = 120):
    """Periodically call malloc_trim(0) to return freed heap pages to the OS."""
    while True:
        await asyncio.sleep(interval)
        if _malloc_trim:
            _malloc_trim(0)


type Tokenizable = str | list["Tokenizable"]


def get_token_count_estimate(content: Tokenizable, tokenizer: Any = None) -> int:
    """
    Returns the number of tokens for the given content using the provided tokenizer.

    Args:
        content (Tokenizable): The text content or nested lists of text to count tokens for.
        tokenizer (Any): The tokenizer used for encoding the text.

    Returns:
        int: The total number of tokens in the provided content.

    Notes:
        - If specific tokenizer is not needed, tiktoken will be used for faster encoding
    """
    if tokenizer is None:
        tokenizer = DEFAULT_TOKENIZER

    if isinstance(content, list):
        return sum(get_token_count_estimate(item, tokenizer) for item in content)

    # Handle Content objects (TextContent, ImageURLContent, InputAudioContent)
    if isinstance(content, dict) and "type" in content:
        if content["type"] == "text" and "text" in content:
            # TextContent object - extract the text field
            return get_token_count_estimate(content["text"], tokenizer)
        if content["type"] in NON_TEXT_TOKEN_ESTIMATES:
            return NON_TEXT_TOKEN_ESTIMATES[content["type"]]
        return 0

    # Handle string content
    if isinstance(content, str):
        if isinstance(tokenizer, tiktoken.core.Encoding):
            encoded = tokenizer.encode(content)
        else:
            encoded = tokenizer.encode(content, add_special_tokens=False)
        return len(encoded)

    # If content is neither string, list, nor recognized dict, return 0
    return 0
