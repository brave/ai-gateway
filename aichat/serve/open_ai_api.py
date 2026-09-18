import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Generator

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)
from pydantic import TypeAdapter

from aichat.llm.metrics import (
    DEEP_RESEARCH_CAPABILITY_TOTAL,
    EMPTY_STREAMING_RESPONSE_TOTAL,
    GENERATED_LENGTH,
    MID_STREAM_ERROR_TOTAL,
    TIME_TO_FIRST_RESPONSE_TOKEN,
    TIME_TO_TOOLS_USE,
)
from aichat.prompts.prompts import prompts
from aichat.protocol.open_ai_protocol import (
    OPENAI_RESPONSE_OBJECTS,
    AssistantMessage,
    Capability,
    CapabilityOptions,
    ErrorCode,
    FileContentPart,
    FileUrlContentPart,
    ImageContentPart,
    InputAudioContentPart,
    Message,
    MessageUnion,
    Tool,
    ToolCall,
    ToolCallFunction,
    VideoContentPart,
    has_capability,
)
from aichat.protocol.open_ai_protocol import (
    Request as OpenAIRequest,
)
from aichat.responses import (
    CompactionMetadata,
    ContentReceipt,
    SecurityScan,
)
from aichat.serve.analytics import send_analytics_request
from aichat.serve.backend.litellm import handle_litellm_error
from aichat.serve.common_api import (
    check_requests_common,
    create_completion_common_params,
    create_error_response,
)
from aichat.serve.constants import CONTENT_FILTER_MESSAGE
from aichat.serve.conversation_settings import conversation_settings
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.mcp_integration import initialize_mcp_for_request
from aichat.serve.mcp_tool_execution import (
    STREAMING_TOOLS,
    execute_tools_and_stream_events,
    handle_streaming_tool_calls,
    simplify_messages_for_llm,
    simplify_tool_message_for_llm,
)
from aichat.serve.message_preprocessing import preprocess_messages
from aichat.serve.open_ai_adapter import (
    ChunkEmitter,
    OpenAIToolCallBufferManager,
    StreamingInjectionScanEmitter,
    handle_alignment_buffering,
    maybe_perform_prompt_injection_scan,
    process_non_streaming_alignment_check,
    process_non_streaming_prompt_injection_scan,
)
from aichat.serve.server_settings import server_settings
from aichat.serve.services.backend import get_backend
from aichat.serve.services.bedrock import (
    apply_content_filter_to_completion_response,
)
from aichat.serve.services.compaction import (
    create_compaction_event,
    maybe_compact_conversation,
    should_compact,
)
from aichat.serve.services.compaction_settings import compaction_settings
from aichat.serve.services.conversation_title import (
    complete_conversation_title_chat,
    last_message_includes_conversation_title,
)
from aichat.serve.services.dynamic_leo.settings import dynamic_leo_settings
from aichat.serve.services.dynamic_leo.signals import run_dynamic_leo
from aichat.serve.services.dynamic_leo.tool_filter import (
    should_apply_dynamic_leo_tool_filter,
)
from aichat.serve.services.mcp.mcp_settings import mcp_settings
from aichat.serve.services.model_selection import (
    AndroclesPrefetch,
    select_model_for_request,
)
from aichat.serve.services.models import (
    ModelConfig,
    get_model_config,
    normalize_model_name,
)
from aichat.serve.services.near import verify_and_get_ohttp_config
from aichat.serve.services.pdf import process_messages_for_pdf_limits
from aichat.serve.services.search import (
    InlineSearchHelper,
)
from aichat.serve.services.security_settings import security_settings
from aichat.serve.services.trimming import maybe_trim_messages
from aichat.serve.tool_parser import (
    ToolCallAccumulator,
    parse_streaming_tool_calls,
    report_duplicate_tool_call_ids,
)
from aichat.serve.truncation_continuation import (
    maybe_non_stream_truncation_recovery,
    plan_stream_truncation_recovery,
    record_stream_truncation_recovery,
    stream_truncation_continuation,
)
from aichat.serve.utils import get_real_ip

logger = logging.getLogger(__name__)


v1_router = APIRouter()

message_adapter = TypeAdapter(MessageUnion)


def _new_metrics_state(start_time: float | None = None) -> dict:
    """Shared per-request state so TTFT/tools-use are observed once, even
    across recursive follow-up calls in the tool-calling loop."""
    return {
        "start_time": start_time if start_time is not None else time.time(),
        "first_token_recorded": False,
        "tools_use_recorded": False,
        "truncation_continuation_depth": 0,
    }


@v1_router.post("/chat/completions", response_model=None)
async def v1_chat_completions(
    raw_request: Request,
    request: OpenAIRequest,
    background_tasks: BackgroundTasks,
    fastapi_response: Response,
    common: dict = Depends(create_completion_common_params),
) -> StreamingResponse | JSONResponse:
    request_start_time = time.time()
    rate_key = get_real_ip(common.get("x_forwarded_for"))

    if last_message_includes_conversation_title(request.messages):
        return await complete_conversation_title_chat(
            request=request,
            prompts=prompts,
            process_streaming_response=process_streaming_response,
        )

    # Classify for model triage and Dynamic Leo MCP tool gating
    dynamic_leo_prefetch = await run_dynamic_leo(request.messages)

    is_premium = common["is_premium_host"] and common["has_valid_premium_credential"]
    premium_fallback = getattr(
        raw_request.state, "model_override_premium_fallback", None
    )
    override_active = getattr(raw_request.state, "model_override", False) and (
        not is_premium or bool(premium_fallback)
    )
    requested_model = server_settings.placeholder_model if override_active else None

    request.model = await select_model_for_request(
        model=common["model"],
        messages=request.messages,
        brave_capability=getattr(request, "brave_capability", None),
        is_premium=common["is_premium_host"] and common["has_valid_premium_credential"],
        last_user_message_content=get_last_user_message_content(request.messages),
        media_type=detect_media_content(request.messages),
        rate_key=rate_key,
        httpx_client=getattr(raw_request.state, "httpx_client", None),
        androcles_prefetch=dynamic_leo_prefetch,
    )

    common["model"] = request.model

    raw_request.state.capability = request.brave_capability

    error_check_ret = await check_requests_common(raw_request, common)
    if error_check_ret:
        return error_check_ret

    # check_requests_common may have swapped in a premium rate-limit
    # fallback model; propagate it to request.model so downstream
    # get_backend()/get_model_config() calls see the real serving model.
    request.model = common["model"]

    if len(request.messages) > conversation_settings.max_conversation_rounds:
        return create_error_response(
            ErrorCode.EXCEEDED_CONTEXT_LENGTH,
            f"message length must be less than the conversation rounds maximum of {conversation_settings.max_conversation_rounds} - 'messages'",
            api_version=2,
        )

    backend = get_backend(request.model)
    model_config = get_model_config(request.model)
    is_premium = common["is_premium_host"] and common["has_valid_premium_credential"]

    request.messages = preprocess_messages(request.messages, model_config)

    if not model_config.tool_support and request.tools:
        request.tools = None

    # Start prompt injection scan concurrently (runs in parallel with compaction/LLM)
    injection_scan_task = maybe_perform_prompt_injection_scan(
        request.messages, request.brave_capability
    )

    messages = [m.model_dump() for m in request.messages]

    # Trim large content parts first (fast, local operation) before checking
    # whether compaction is needed. This avoids sending bloated messages to
    # the compaction LLM and may eliminate the need for compaction entirely.
    trimmed_messages, total_tokens, trimmed_tokens = maybe_trim_messages(
        messages=messages,
        model_config=model_config,
        is_premium=is_premium,
        model=request.model,
    )

    if trimmed_tokens > 0:
        messages = trimmed_messages
        request.messages = [
            message_adapter.validate_python(m) for m in trimmed_messages
        ]

    # Hard ceiling: reject requests that are still pathologically large after
    # trimming rather than forwarding them to the LLM (and compaction).
    if total_tokens > compaction_settings.absolute_max_tokens:
        return create_error_response(
            ErrorCode.EXCEEDED_CONTEXT_LENGTH,
            f"Conversation is too large ({total_tokens} tokens). "
            f"Please start a new conversation or reduce the amount of attached content.",
        )

    # Send a compaction_starting event before compaction begins.
    capability = getattr(request, "brave_capability", None)
    if should_compact(messages, model_config, is_premium, request.stream, capability):
        response_headers = {}
        if request.model.startswith("near-"):
            response_headers["Brave-NEAR-verified"] = str(
                bool(await verify_and_get_ohttp_config(request.model))
            ).lower()

        return StreamingResponse(
            _stream_with_compaction(
                request=request,
                messages=messages,
                model_config=model_config,
                is_premium=is_premium,
                backend=backend,
                injection_scan_task=injection_scan_task,
                trimmed_tokens=trimmed_tokens,
                requested_model=requested_model,
                dynamic_leo=dynamic_leo_prefetch,
                request_start_time=request_start_time,
            ),
            media_type="text/event-stream",
            headers=response_headers,
        )

    # Non-streaming or streaming without compaction - use original flow
    tools, _, mcp_executor = await prepare_tools_for_request(
        client_tools=request.tools,
        model_config=model_config,
        mcp_tools_exclude=request.brave_mcp_tools_exclude,
        mcp_tools_include=request.brave_mcp_tools_include,
        brave_capability=request.brave_capability,
        dynamic_leo=dynamic_leo_prefetch,
    )

    messages = await augment_messages(
        messages=request.messages,
        tools=tools,
        model_config=model_config,
        is_premium=is_premium,
        brave_capability=request.brave_capability,
    )

    messages = simplify_messages_for_llm(messages)

    params = backend.build_params(request.stream, tools, messages)

    if request.model.startswith("near-"):
        params["api_key"] = external_service_settings.near_api_key

    # Send analytics request in the background
    background_tasks.add_task(
        send_analytics_request,
        messages=messages,
        model=request.model,
    )

    response = await backend.converse(messages, request.stream, params)

    if request.stream:
        response_headers = {}

        if request.model.startswith("near-"):
            response_headers["Brave-NEAR-verified"] = str(
                bool(await verify_and_get_ohttp_config(request.model))
            ).lower()

        return StreamingResponse(
            process_streaming_response(
                response,
                request,
                tools,
                mcp_executor,
                backend,
                model_config,
                prompts,
                None,
                injection_scan_task=injection_scan_task,
                trimmed_tokens=trimmed_tokens,
                requested_model=requested_model,
                metrics_state=_new_metrics_state(request_start_time),
            ),
            media_type="text/event-stream",
            headers=response_headers,
        )
    else:
        if response.get("type") == "error":
            raise HTTPException(
                status_code=int(response.get("code", 50000) / 100),
                detail=response.get("content"),
            )

        if request.model.startswith("near-"):
            fastapi_response.headers["Brave-NEAR-verified"] = str(
                bool(await verify_and_get_ohttp_config(request.model))
            ).lower()

        # Perform alignment check for non-streaming tool calls
        if security_settings.alignment_checking_enabled:
            response = await process_non_streaming_alignment_check(
                response, request.messages, injection_scan_task=injection_scan_task
            )

        response = await process_non_streaming_prompt_injection_scan(
            response, injection_scan_task
        )

        apply_content_filter_to_completion_response(response)

        if hasattr(response, "model") and response.model:
            normalized = normalize_model_name(
                response.model, requested_model_id=request.model
            )
            if normalized:
                response.model = normalized
            if requested_model:
                response.model = requested_model

        if getattr(response, "choices", None):
            generated_content = getattr(
                getattr(response.choices[0], "message", None), "content", None
            )
            if generated_content:
                GENERATED_LENGTH.labels(request.model).observe(len(generated_content))

        response = await maybe_non_stream_truncation_recovery(
            response=response,
            request=request,
            llm_messages=messages,
            tools=tools,
            params=params,
            backend=backend,
            model_config=model_config,
        )

        return response


async def augment_messages(
    messages: list[MessageUnion],
    tools: list[Tool],
    model_config: ModelConfig,
    is_premium: bool = False,
    brave_capability: CapabilityOptions = None,
) -> list[dict]:
    """
    Augment messages with system prompts.

    Args:
        messages: List of message objects
        tools: List of tools for the conversation
        model_config: Model configuration
        is_premium: Whether the user has premium access
        brave_capability: Capabilities advertised by the client, used by
            prompts to gate directives on client support

    Returns:
        List of augmented message dicts
    """
    messages_dict = []
    for message in messages:
        messages_dict.append(message.model_dump(exclude_none=True))

    token_limit = (
        model_config.conversation_token_limit_premium
        if is_premium
        else model_config.conversation_token_limit
    )
    messages_dict = await process_messages_for_pdf_limits(messages_dict, token_limit)

    for prompt in prompts.prompts:
        messages_dict = prompt.augment(
            messages_dict,
            model_config=model_config,
            tools=tools,
            brave_capability=brave_capability,
        )

    return messages_dict


async def _stream_with_compaction(
    request: OpenAIRequest,
    messages: list[dict],
    model_config: ModelConfig,
    is_premium: bool,
    backend,
    injection_scan_task=None,
    trimmed_tokens: int = 0,
    requested_model: str | None = None,
    dynamic_leo: AndroclesPrefetch | None = None,
    request_start_time: float | None = None,
) -> AsyncIterator[str]:
    """
    Stream response with compaction, yielding compaction_starting event first.

    Token trimming is performed by the caller before this function is invoked,
    so messages are already right-sized when compaction runs.

    This generator:
    1. Yields a compaction_starting SSE event
    2. Performs compaction (summarizes old conversation turns)
    3. Yields all events from process_streaming_response
    """
    yield create_compaction_event("compaction_starting")
    logger.info("Sent compaction_starting event")

    try:
        compacted_messages, compaction_metadata = await maybe_compact_conversation(
            messages=messages,
            model=request.model,
            model_config=model_config,
            is_premium=is_premium,
            skip_checks=True,
        )

        if compaction_metadata:
            request.messages = [
                message_adapter.validate_python(m) for m in compacted_messages
            ]

        tools, _tool_names, mcp_executor = await prepare_tools_for_request(
            client_tools=request.tools,
            model_config=model_config,
            mcp_tools_exclude=request.brave_mcp_tools_exclude,
            mcp_tools_include=request.brave_mcp_tools_include,
            brave_capability=request.brave_capability,
            dynamic_leo=dynamic_leo,
        )

        messages = await augment_messages(
            messages=request.messages,
            tools=tools,
            model_config=model_config,
            is_premium=is_premium,
            brave_capability=request.brave_capability,
        )

        messages = simplify_messages_for_llm(messages)

        params = backend.build_params(request.stream, tools, messages)

        if request.model.startswith("near-"):
            params["api_key"] = external_service_settings.near_api_key

        response = await backend.converse(messages, request.stream, params)

        async for event in process_streaming_response(
            response,
            request,
            tools,
            mcp_executor,
            backend,
            model_config,
            prompts,
            compaction_metadata,
            injection_scan_task=injection_scan_task,
            trimmed_tokens=trimmed_tokens,
            requested_model=requested_model,
            metrics_state=_new_metrics_state(request_start_time),
        ):
            yield event

    except Exception:
        logger.exception("Error during compaction streaming")
        yield create_compaction_event(
            "compaction_failed",
            {"error": "An internal error occurred during compaction"},
        )
        yield "data: [DONE]\n\n"
        return


async def prepare_tools_for_request(
    client_tools: list[Tool] | None,
    model_config,
    mcp_tools_exclude: list[str] | None = None,
    mcp_tools_include: list[str] | None = None,
    brave_capability: CapabilityOptions = None,
    dynamic_leo: AndroclesPrefetch | None = None,
) -> tuple[list[Tool], list[str], object | None]:
    """
    Prepare tools for the request by combining client tools with MCP tools.
    Returns:
        Tuple of (tools, tool_names, mcp_executor)
    """
    client_tools = client_tools or []
    mcp_tools = []
    mcp_executor = None

    # Fetch MCP tools if enabled and model supports tools
    if mcp_settings.mcp_enabled and model_config.tool_support:
        try:
            matched_categories = (
                dynamic_leo.matched_categories if dynamic_leo is not None else None
            )
            dynamic_leo_tool_filtering = should_apply_dynamic_leo_tool_filter(
                enable_dynamic_leo=dynamic_leo_settings.enable_dynamic_leo,
                prompt_caching_support=model_config.prompt_caching_support,
            )
            mcp_tools, mcp_executor = await initialize_mcp_for_request(
                model_config=model_config,
                matched_categories=matched_categories,
                dynamic_leo_enabled=dynamic_leo_tool_filtering,
            )

            tools_to_exclude = mcp_tools_exclude or []
            tools_to_include = mcp_tools_include or []

            # Exclusion takes precedence
            if "all" in tools_to_exclude:
                mcp_tools = []
            elif tools_to_exclude:
                mcp_tools = [
                    tool
                    for tool in mcp_tools
                    if tool.function.name not in tools_to_exclude
                ]
            elif tools_to_include and "all" not in tools_to_include:
                mcp_tools = [
                    tool for tool in mcp_tools if tool.function.name in tools_to_include
                ]

            if not has_capability(brave_capability, Capability.deep_research):
                mcp_tools = [
                    tool for tool in mcp_tools if tool.function.name != "deep_research"
                ]

            logger.info(f"Loaded {len(mcp_tools)} MCP tools")

        except (httpx.HTTPError, httpx.TimeoutException):
            logger.exception("Failed to fetch MCP tools due to network error")
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.exception("Failed to parse MCP tools response")

    tools = client_tools + mcp_tools
    tool_names = [tool.function.name for tool in tools]
    logger.info(f"Total tools being sent to model: {tool_names}")

    return tools, tool_names, mcp_executor


async def convert_litellm_response_to_openai_chat_completion(
    generator: Generator,
    model: str | None = None,
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """
    Transform litellm streaming response to ensure proper format
    and handle reasoning output.
    """
    # Track reasoning block state
    in_reasoning_block = False

    async for chunk in generator:
        if hasattr(chunk, "model") and chunk.model:
            normalized = normalize_model_name(chunk.model, requested_model_id=model)
            if normalized:
                chunk.model = normalized

        # Log usage if available and logging enabled
        if server_settings.log_level == "DEBUG" and hasattr(chunk, "usage"):
            logger.debug(f"Litellm usage: {chunk.usage}")
        if hasattr(chunk, "usage") and chunk.usage:
            yield chunk
            continue

        # Skip processing if no choices or empty choices
        if not hasattr(chunk, "choices") or not chunk.choices:
            yield chunk
            continue

        # Check if first choice is None
        if chunk.choices[0] is None:
            yield chunk
            continue

        choice: Choice = chunk.choices[0]
        delta: ChoiceDelta = choice.delta

        # Handle reasoning content (if available)
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_text = delta.reasoning_content
            # Add <think> tag on first reasoning chunk
            if not in_reasoning_block:
                in_reasoning_block = True
                reasoning_text = "<think>" + reasoning_text

            reasoning_chunk = create_content_chunk(chunk, reasoning_text)
            if reasoning_chunk:
                yield reasoning_chunk
                continue

        # Handle regular content - close reasoning block if needed
        if delta.content and in_reasoning_block:
            delta.content = "</think>" + delta.content
            in_reasoning_block = False

        # Handle tool calls
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                if tool_call.function and tool_call.function.arguments == "{}":
                    # If the arguments delta is "{}", change it to
                    # empty string to prevent clients from concatenating
                    # it and creating invalid JSON.
                    tool_call.function.arguments = ""

        yield chunk


def create_content_chunk(
    original_chunk: ChatCompletionChunk, content: str
) -> ChatCompletionChunk | None:
    """
    Create a new chunk with specific content, preserving the original
    chunk structure.
    """
    if not original_chunk.choices:
        return None

    # Modify the original chunk directly since it won't be used again
    original_chunk.choices[0].delta.content = content
    # Clear reasoning content if present
    if hasattr(original_chunk.choices[0].delta, "reasoning_content"):
        original_chunk.choices[0].delta.reasoning_content = ""
    return original_chunk


def _is_mid_stream_fallback_error(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ == "MidStreamFallbackError":
            return True
        current = current.__cause__ or current.__context__
    return False


async def process_streaming_response(
    response: AsyncIterator,
    request: OpenAIRequest,
    original_client_tools: list[str | Tool],
    mcp_executor=None,
    backend=None,
    model_config=None,
    prompts=None,
    compaction_info: CompactionMetadata | None = None,
    injection_scan_task=None,
    trimmed_tokens: int | None = None,
    requested_model: str | None = None,
    metrics_state: dict | None = None,
) -> AsyncIterator[str]:
    """
    Process streaming response, execute tools, and format as SSE events

    Yields:
        SSE-formatted strings for each streaming event
    """
    if metrics_state is None:
        metrics_state = _new_metrics_state()
    response = convert_litellm_response_to_openai_chat_completion(
        response, request.model
    )
    tool_accumulator = ToolCallAccumulator()
    tool_calls_in_response = []
    assistant_tool_calls_for_message = []
    conversation_log_id = None
    actual_model = None
    deep_research_triggered = False
    tool_buffer_manager = OpenAIToolCallBufferManager()
    assistant_content_accumulator = ""
    chunk_count = 0
    last_finish_reason = None
    chunk_emitter = ChunkEmitter(compaction_info)
    inline_search_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=1.0, read=2.0, write=1.0, pool=1.0)
    )
    inline_search_helper = InlineSearchHelper(inline_search_client)

    try:
        pi_emitter = StreamingInjectionScanEmitter(injection_scan_task)
        scan_event = create_openai_response(pi_emitter.scan_started_event())
        if scan_event:
            yield chunk_emitter.emit(scan_event)

        async for chunk in response:
            chunk_count += 1
            for inline_search in inline_search_helper.pop_completed_searches():
                yield chunk_emitter.emit(create_openai_response(inline_search))

            pi_result_event = create_openai_response(pi_emitter.try_get_result())
            if pi_result_event:
                yield chunk_emitter.emit(pi_result_event)

            if conversation_log_id is None and chunk.id:
                conversation_log_id = chunk.id
            if actual_model is None and chunk.model:
                actual_model = chunk.model.replace(" ", "")

            if hasattr(chunk, "usage") and chunk.usage:
                usage_chunk_dict = chunk.model_dump(exclude_none=True)
                if "model" in usage_chunk_dict:
                    usage_chunk_dict["model"] = usage_chunk_dict["model"].replace(
                        " ", ""
                    )
                    if requested_model:
                        usage_chunk_dict["model"] = requested_model
                usage_chunk_json = json.dumps(usage_chunk_dict, separators=(",", ":"))
                yield chunk_emitter.emit(f"data: {usage_chunk_json}\n\n")

                yield chunk_emitter.emit(
                    create_openai_response(
                        ContentReceipt(
                            total_tokens=chunk.usage.total_tokens,
                            trimmed_tokens=trimmed_tokens or 0,
                        )
                    )
                )
                continue

            if hasattr(chunk, "choices") and chunk.choices:
                if chunk.choices[0] is not None:
                    stop_reason = chunk.choices[0].finish_reason
                    if stop_reason:
                        last_finish_reason = stop_reason

                    # When the model returns content_filter (e.g. Bedrock safety filter),
                    # emit a chunk with a user-facing message instead of empty/truncated content.
                    if stop_reason == "content_filter":
                        chunk = create_content_chunk(chunk, CONTENT_FILTER_MESSAGE)
                        if chunk and chunk.choices:
                            chunk.choices[0].finish_reason = "content_filter"
                        assistant_content_accumulator = CONTENT_FILTER_MESSAGE
                    else:
                        # Accumulate assistant content for alignment checking
                        if (
                            hasattr(chunk.choices[0].delta, "content")
                            and chunk.choices[0].delta.content
                        ):
                            if not metrics_state["first_token_recorded"]:
                                # "uses_search" has no clean analog with MCP tools;
                                # kept as a fixed value so the histogram's label set
                                # (shared with the legacy /conversation pipeline)
                                # doesn't need to change.
                                TIME_TO_FIRST_RESPONSE_TOKEN.labels(
                                    request.model, "unknown"
                                ).observe(time.time() - metrics_state["start_time"])
                                metrics_state["first_token_recorded"] = True

                            assistant_content_accumulator += chunk.choices[
                                0
                            ].delta.content
                            inline_search_helper.handle_received_completion(
                                assistant_content_accumulator
                            )

                        if (
                            chunk.choices[0].delta.tool_calls
                            or stop_reason == "tool_calls"
                        ):
                            completed_tools = parse_streaming_tool_calls(
                                tool_accumulator,
                                chunk.choices[0].delta.tool_calls,
                                stop_reason,
                            )

                            if completed_tools:
                                if not metrics_state["tools_use_recorded"]:
                                    TIME_TO_TOOLS_USE.labels(request.model).observe(
                                        time.time() - metrics_state["start_time"]
                                    )
                                    metrics_state["tools_use_recorded"] = True

                                tool_calls_in_response.extend(completed_tools)

                                for tool_call in completed_tools:
                                    assistant_tool_calls_for_message.append(
                                        ToolCall(
                                            id=tool_call.id
                                            or f"call_{tool_call.index}",
                                            type="function",
                                            function=ToolCallFunction(
                                                name=tool_call.function_name,
                                                arguments=tool_call.arguments or "{}",
                                            ),
                                        )
                                    )

                        # Handle tool call buffering for alignment checking
                        if security_settings.alignment_checking_enabled:
                            (
                                should_continue,
                                buffered_chunks,
                                started_tool_name,
                            ) = await handle_alignment_buffering(
                                tool_buffer_manager,
                                chunk,
                                stop_reason,
                                request.messages,
                                mcp_executor,
                                backend,
                                tool_calls_in_response,
                                assistant_content_accumulator or None,
                                injection_scan_task=injection_scan_task,
                            )
                            if started_tool_name is not None:
                                yield chunk_emitter.emit(
                                    create_openai_response(
                                        SecurityScan(
                                            type="alignment_check",
                                            tool_name=started_tool_name,
                                        )
                                    )
                                )
                            for chunk_str in buffered_chunks:
                                yield chunk_emitter.emit(chunk_str)
                            if should_continue:
                                continue

                chunk_dict = chunk.model_dump(exclude_none=True)
                if "model" in chunk_dict:
                    chunk_dict["model"] = chunk_dict["model"].replace(" ", "")
                    if requested_model:
                        chunk_dict["model"] = requested_model

                should_skip = False
                if (
                    mcp_executor
                    and backend
                    and chunk_dict.get("choices", [{}])[0].get("finish_reason")
                    == "tool_calls"
                ):
                    logger.info(
                        "Skipping finish_reason chunk for server-side tool execution"
                    )
                    should_skip = True

                if not should_skip:
                    chunk_json = json.dumps(chunk_dict, separators=(",", ":"))
                    logger.debug(f"Chunk JSON: {chunk_json}")
                    yield chunk_emitter.emit(f"data: {chunk_json}\n\n")

        if not assistant_content_accumulator and not tool_calls_in_response:
            if chunk_count == 0:
                empty_reason = "no_chunks"
            elif tool_accumulator.tool_calls:
                empty_reason = "incomplete_tool_call"
            elif last_finish_reason:
                empty_reason = last_finish_reason
            else:
                empty_reason = "unknown"
            EMPTY_STREAMING_RESPONSE_TOTAL.labels(request.model, empty_reason).inc()

        if tool_calls_in_response:
            report_duplicate_tool_call_ids(tool_calls_in_response, request.model)

        if tool_calls_in_response and mcp_executor and backend:
            mcp_tool_calls = []
            streaming_tool_calls = []
            client_tool_calls = []
            for tool_call in tool_calls_in_response:
                if tool_call.function_name in STREAMING_TOOLS:
                    streaming_tool_calls.append(tool_call)
                    if tool_call.function_name == "deep_research":
                        deep_research_triggered = True
                        DEEP_RESEARCH_CAPABILITY_TOTAL.labels(
                            request.model, "True", str(len(request.messages))
                        ).inc()
                elif await mcp_executor.is_mcp_tool(tool_call.function_name):
                    mcp_tool_calls.append(tool_call)
                else:
                    client_tool_calls.append(tool_call)

            # Handle streaming tools (like deep_research) first
            if streaming_tool_calls:
                async for chunk_str in handle_streaming_tool_calls(
                    streaming_tool_calls=streaming_tool_calls,
                    conversation_log_id=conversation_log_id,
                    actual_model=actual_model,
                    mcp_tool_calls=mcp_tool_calls,
                    client_tool_calls=client_tool_calls,
                    assistant_tool_calls_for_message=assistant_tool_calls_for_message,
                    request=request,
                    backend=backend,
                    model_config=model_config,
                    prompts_obj=prompts,
                    original_client_tools=original_client_tools,
                    metrics_state=metrics_state,
                ):
                    yield chunk_str
                if not mcp_tool_calls and not client_tool_calls:
                    yield "data: [DONE]\n\n"
                    return

            if mcp_tool_calls:
                logger.info(
                    f"Executing {len(mcp_tool_calls)} MCP tool calls server-side "
                    f"({len(client_tool_calls)} client-side tools will be "
                    f"forwarded)"
                )

                tool_result_messages = []
                async for (
                    event_str,
                    tool_output_or_message,
                ) in execute_tools_and_stream_events(
                    mcp_tool_calls,
                    mcp_executor,
                    conversation_log_id or "unknown",
                    actual_model,
                    request.model,
                ):
                    if event_str:
                        yield event_str
                    if tool_output_or_message:
                        if isinstance(tool_output_or_message, dict):
                            chunk_json = json.dumps(
                                tool_output_or_message, separators=(",", ":")
                            )
                            yield f"data: {chunk_json}\n\n"
                        else:
                            tool_result_messages.append(tool_output_or_message)

                if tool_result_messages and not client_tool_calls:
                    new_messages = [
                        (
                            message.model_dump(exclude_none=True)
                            if hasattr(message, "model_dump")
                            else message
                        )
                        for message in request.messages
                    ]

                    filtered_tool_calls = [
                        tc
                        for tc in assistant_tool_calls_for_message
                        if any(
                            tc.function.name == mcp_tc.function_name
                            for mcp_tc in mcp_tool_calls
                        )
                    ]

                    assistant_message = AssistantMessage(
                        role="assistant",
                        content=None,
                        tool_calls=filtered_tool_calls,
                    )
                    new_messages.append(assistant_message.model_dump(exclude_none=True))

                    new_messages.extend(
                        (
                            simplify_tool_message_for_llm(msg)
                            if hasattr(msg, "model_dump")
                            else msg
                        )
                        for msg in tool_result_messages
                    )

                    logger.debug(
                        f"Follow-up messages: {json.dumps(new_messages[-3:], indent=2)}"
                    )

                    if prompts:
                        for prompt in prompts.prompts:
                            new_messages = prompt.augment(
                                new_messages,
                                model_config=model_config,
                                tools=original_client_tools,
                            )

                    params = backend.build_params(
                        request.stream, original_client_tools, new_messages
                    )

                    if request.model.startswith("near-"):
                        params["api_key"] = external_service_settings.near_api_key

                    logger.info("Making follow-up call to model with tool results")
                    followup_response = await backend.converse(
                        new_messages, request.stream, params
                    )

                    followup_request = OpenAIRequest.model_validate(
                        request.model_dump(exclude_none=True)
                        | {"messages": new_messages}
                    )

                    # Recursively process the follow-up response to handle
                    # additional tool calls
                    async for chunk_str in process_streaming_response(
                        followup_response,
                        followup_request,
                        original_client_tools,
                        mcp_executor,
                        backend,
                        model_config,
                        prompts,
                        None,  # No compaction metadata for follow-up calls
                        requested_model=requested_model,
                        metrics_state=metrics_state,
                    ):
                        yield chunk_str
                    return
                elif client_tool_calls:
                    logger.info(
                        f"Executed {len(mcp_tool_calls)} server-side tools, but "
                        f"{len(client_tool_calls)} client-side tools present. "
                        "Stopping and waiting for client to execute client-side tools."
                    )
            elif client_tool_calls:
                logger.info(
                    f"All {len(client_tool_calls)} tool calls are client-side, "
                    "stopping and waiting for client"
                )

        pi_result_event = create_openai_response(await pi_emitter.await_result())
        if pi_result_event:
            yield chunk_emitter.emit(pi_result_event)

        # Return all remaining inline search chunks.
        for inline_search in await inline_search_helper.get_inline_search_chunks():
            yield chunk_emitter.emit(create_openai_response(inline_search))

        if not deep_research_triggered and has_capability(
            request.brave_capability, Capability.deep_research
        ):
            DEEP_RESEARCH_CAPABILITY_TOTAL.labels(
                request.model, "False", str(len(request.messages))
            ).inc()

        if assistant_content_accumulator:
            GENERATED_LENGTH.labels(request.model).observe(
                len(assistant_content_accumulator)
            )

        truncation_action = plan_stream_truncation_recovery(
            assistant_text=assistant_content_accumulator,
            tool_calls_in_response=bool(tool_calls_in_response),
            last_finish_reason=last_finish_reason,
            model_config=model_config,
            backend=backend,
            metrics_state=metrics_state,
        )
        if truncation_action == "continue":
            record_stream_truncation_recovery(
                model=request.model,
                action=truncation_action,
                depth=metrics_state.get("truncation_continuation_depth", 0),
            )
            async for chunk_str in stream_truncation_continuation(
                request=request,
                partial_assistant_text=assistant_content_accumulator,
                client_tools=original_client_tools,
                mcp_executor=mcp_executor,
                backend=backend,
                model_config=model_config,
                prompts_obj=prompts,
                requested_model=requested_model,
                metrics_state=metrics_state,
                process_streaming_response=process_streaming_response,
            ):
                yield chunk_str
            return
        if truncation_action == "retry":
            record_stream_truncation_recovery(
                model=request.model,
                action=truncation_action,
                depth=metrics_state.get("truncation_continuation_depth", 0),
            )

        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.exception("Exception /chat/completions stream processing")
        if _is_mid_stream_fallback_error(e):
            MID_STREAM_ERROR_TOTAL.labels(request.model).inc()
            error_response = handle_litellm_error(
                e, is_streaming=False, model=request.model
            )
            error_chunk = ChatCompletionChunk(
                id=conversation_log_id or "error",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=requested_model or actual_model or request.model,
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(
                            content=error_response["content"],
                            role="assistant",
                        ),
                        finish_reason="stop",
                    )
                ],
            )
            chunk_json = json.dumps(
                error_chunk.model_dump(exclude_none=True), separators=(",", ":")
            )
            yield chunk_emitter.emit(f"data: {chunk_json}\n\n")
        yield "data: [DONE]\n\n"
    finally:
        await inline_search_client.aclose()
        if hasattr(response, "aclose"):
            await response.aclose()


def get_last_user_message_content(messages: list[Message]) -> str | None:
    """Extract the content from the last user message in the conversation."""
    for message in reversed(messages):
        if message.role == "user" and message.content:
            if isinstance(message.content, str):
                return message.content
            elif isinstance(message.content, list):
                text_parts = []
                for part in message.content:
                    if hasattr(part, "text") and part.text is not None:
                        text_parts.append(part.text)
                    elif hasattr(part, "content") and isinstance(part.content, str):
                        text_parts.append(part.content)
                return " ".join(text_parts) if text_parts else None
    return None


def detect_media_content(messages: list[Message]) -> str | None:
    """Detect media content in messages at the API layer."""
    for message in reversed(messages):
        if not message.content:
            continue

        if isinstance(message.content, list):
            for part in message.content:
                # Check in order of decreasing frequency for performance optimization:
                # Images are most common, followed by files, then video, then audio
                if isinstance(part, ImageContentPart):
                    return "vision"
                elif isinstance(part, (FileContentPart, FileUrlContentPart)):
                    return "file"
                elif isinstance(part, VideoContentPart):
                    return "video"
                elif isinstance(part, InputAudioContentPart):
                    return "audio"

    return None


def create_openai_response(response_data) -> str | None:
    """
    Translate protocol-agnostic responses to OpenAI format by adding the 'object' field.

    Args:
        response_data: A protocol-agnostic response from aichat.responses

    Returns:
        SSE-formatted string for OpenAI protocol, or None if not translatable
    """
    response_type = type(response_data)
    object_type = OPENAI_RESPONSE_OBJECTS.get(response_type)

    if object_type is None:
        return None

    response_dict = response_data.model_dump(exclude_none=True)
    response_dict["object"] = object_type

    response_json = json.dumps(response_dict, separators=(",", ":"))
    return f"data: {response_json}\n\n"
