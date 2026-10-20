import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from openai.types.chat import CompletionCreateParams
from pydantic import TypeAdapter
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.routing import Match
from starlette.types import Scope

from aichat.protocol.open_ai_protocol import MessageUnion
from aichat.serve import internal_client
from aichat.serve.api_key_chat_settings import api_key_chat_settings
from aichat.serve.backend.litellm import apply_claude_upstream_sampling_params
from aichat.serve.common_api import extract_bearer_token
from aichat.serve.internal_settings import internal_settings
from aichat.serve.open_ai_api import detect_media_content, get_last_user_message_content
from aichat.serve.server_settings import server_settings
from aichat.serve.services import api_key_usage
from aichat.serve.services.backend import get_backend
from aichat.serve.services.data_url_gate import remove_unprocessed_data_url_parts
from aichat.serve.services.dynamic_leo.signals import run_dynamic_leo
from aichat.serve.services.model_selection import select_model_for_request
from aichat.serve.services.pdf import process_messages_for_pdf_limits
from aichat.serve.utils import get_real_ip

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

_request_adapter = TypeAdapter(CompletionCreateParams)
_message_adapter = TypeAdapter(MessageUnion)
_PASSTHROUGH_EXCLUDE = {"model", "messages", "stream"}


def openai_error_response(
    status_code: int, message: str, error_type: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": error_type}},
    )


def is_valid_api_key(api_key: str) -> bool:
    return any(
        api_key.startswith(prefix) for prefix in api_key_chat_settings.api_key_prefixes
    )


def api_key_from_authorization(authorization: str | None) -> str | None:
    token = extract_bearer_token(authorization)
    if token is None or not is_valid_api_key(token):
        return None
    return token


def is_api_key_host(headers: Headers) -> bool:
    api_key_host = api_key_chat_settings.api_key_host
    if not api_key_host:
        return False
    return api_key_host in (headers.get("host"), headers.get("x-forwarded-host"))


def is_api_key_request(headers: Headers) -> bool:
    if not server_settings.api_key_chat_enabled:
        return False
    if api_key_chat_settings.api_key_host:
        return is_api_key_host(headers)
    return api_key_from_authorization(headers.get("authorization")) is not None


def check_api_key_host(request: Request) -> JSONResponse | None:
    if (
        not server_settings.api_key_chat_enabled
        or not api_key_chat_settings.api_key_host
    ):
        return None
    if is_api_key_host(request.headers):
        path = request.url.path
        if path.startswith("/v1/") and path != CHAT_COMPLETIONS_PATH:
            return openai_error_response(
                403, "This endpoint is not available.", "permission_denied"
            )
        return None
    if api_key_from_authorization(request.headers.get("authorization")) is not None:
        return openai_error_response(
            401, "API keys are not accepted on this host.", "invalid_api_key"
        )
    return None


class ApiKeyRoute(APIRoute):
    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        if scope["type"] == "http" and not is_api_key_request(Headers(scope=scope)):
            return Match.NONE, {}
        return super().matches(scope)


v1_router = APIRouter(route_class=ApiKeyRoute)


async def authorize_api_key(request: Request, api_key: str) -> JSONResponse | None:
    if not internal_settings.internal_api_enabled:
        return None
    verdict = await internal_client.api_key_verify(request.state.httpx_client, api_key)
    if verdict is None:
        return openai_error_response(503, "Service unavailable.", "service_unavailable")
    if not verdict.get("allowed", False):
        return openai_error_response(401, "Invalid API key.", "invalid_api_key")
    return None


# Streams the completion chunks of the response to the client.
# If usage is requested, it will be settled in the background after the
# response is delivered. If usage is not requested, it will be settled in the
# background after the response is delivered.
async def _stream_completion_chunks(
    response: AsyncIterator,
    model: str,
    usage_ctx: api_key_usage.UsageContext | None = None,
    include_usage: bool = True,
) -> AsyncIterator[str]:
    chunks: asyncio.Queue = asyncio.Queue(maxsize=api_key_usage.STREAM_BUFFER_SIZE)
    disconnected = asyncio.Event()
    reader = api_key_usage.spawn(
        api_key_usage.read_stream_and_settle(response, chunks, disconnected, usage_ctx)
    )
    try:
        while True:
            chunk = await chunks.get()
            if chunk is api_key_usage.STREAM_END:
                break
            if isinstance(chunk, Exception):
                raise chunk
            if getattr(chunk, "usage", None) is not None and not include_usage:
                if not chunk.choices:
                    continue
                chunk.usage = None
            try:
                chunk.model = model
                yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as e:
                logger.warning(f"Failed to serialize chunk: {e}")
        yield "data: [DONE]\n\n"
    finally:
        disconnected.set()
        while not chunks.empty():
            chunks.get_nowait()
        if usage_ctx is None:
            reader.cancel()


@v1_router.post("/chat/completions", response_model=None)
async def chat_completions(request: Request):
    api_key = api_key_from_authorization(request.headers.get("authorization"))
    if api_key is None:
        return openai_error_response(401, "Invalid API key.", "invalid_api_key")
    error = await authorize_api_key(request, api_key)
    if error is not None:
        return error
    return await handle_chat_completions(request, api_key)


async def handle_chat_completions(request: Request, api_key: str | None = None):
    logger.debug("api_key_chat_api.py /chat/completions")
    try:
        body = await request.json()
        chat_request = _request_adapter.validate_python(body)
    except Exception as e:
        return openai_error_response(400, str(e), "invalid_request_error")

    enable_prompt_caching = body.get("prompt_caching")
    if enable_prompt_caching is not None and not isinstance(
        enable_prompt_caching, bool
    ):
        return openai_error_response(
            400, "prompt_caching must be a boolean", "invalid_request_error"
        )

    try:
        model = str(chat_request["model"])
        # Validate the RAW body messages once: outer CompletionCreateParams
        # validation exhausts the lazy content iterators, so messages taken
        # from chat_request re-validate/dump to empty content. Raw dicts
        # validate cleanly and model_dump strips extra keys (file_id/format).
        protocol_messages = [
            _message_adapter.validate_python(message) for message in body["messages"]
        ]
        messages = [m.model_dump(exclude_none=True) for m in protocol_messages]
        messages = await remove_unprocessed_data_url_parts(messages)
        messages = await process_messages_for_pdf_limits(messages, None)
        stream = bool(chat_request.get("stream"))
        extra_params = {
            key: value
            for key, value in chat_request.items()
            if key not in _PASSTHROUGH_EXCLUDE and value is not None
        }
        tools = list(extra_params.pop("tools", None) or [])
        dynamic_leo_prefetch = await run_dynamic_leo(protocol_messages)
        model = await select_model_for_request(
            model=model,
            messages=protocol_messages,
            is_premium=False,
            last_user_message_content=get_last_user_message_content(protocol_messages),
            media_type=detect_media_content(protocol_messages),
            rate_key=get_real_ip(request.headers.get("x-forwarded-for")),
            httpx_client=getattr(request.state, "httpx_client", None),
            androcles_prefetch=dynamic_leo_prefetch,
        )

        backend = get_backend(model)

        params = backend.build_params(
            stream=stream,
            tools=tools,
            messages=messages,
            enable_prompt_caching=enable_prompt_caching,
        )
        for key in ("temperature", "top_p", "stop"):
            if key not in extra_params:
                params.pop(key, None)
        params.update(extra_params)
        include_usage = bool(
            (extra_params.get("stream_options") or {}).get("include_usage")
        )
        if stream:
            params["stream_options"] = {
                **(params.get("stream_options") or {}),
                "include_usage": True,
            }
        apply_claude_upstream_sampling_params(backend.config.upstream_model, params)

        usage_ctx = None
        if api_key is not None and internal_settings.internal_api_enabled:
            usage_ctx = api_key_usage.new_usage_context(
                request.state.httpx_client, api_key, model, backend.config
            )

        converse = backend.converse(messages, stream=stream, params=params)
        if stream:
            response = await converse
        else:
            response = await asyncio.shield(
                api_key_usage.spawn(
                    api_key_usage.converse_and_settle(converse, usage_ctx)
                )
            )
    except Exception:
        logger.exception("API-key chat completion failed")
        return openai_error_response(500, "Internal server error.", "internal_error")

    if isinstance(response, dict) and response.get("type") == "error":
        return openai_error_response(
            int(response.get("code", 50000) / 100),
            response.get("content"),
            "internal_error",
        )

    if stream:
        return StreamingResponse(
            _stream_completion_chunks(response, model, usage_ctx, include_usage),
            media_type="text/event-stream",
        )
    response.model = model
    return JSONResponse(content=response.model_dump())
