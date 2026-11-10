from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic_settings import BaseSettings
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.routing import Match
from starlette.types import Scope

from aichat.serve import internal_client
from aichat.serve.common_api import extract_bearer_token
from aichat.serve.internal_settings import internal_settings
from aichat.serve.server_settings import server_settings
from aichat.serve.services.api_key_chat import (
    DEFAULT_API_KEY_CAPABILITIES,
    handle_chat_completions,
    openai_error_response,
)

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


class ApiKeyChatSettings(BaseSettings):
    api_key_prefixes: list[str] = ["test_key_"]
    api_key_host: str = ""
    default_key_capabilities: list[str] = ["chat"]


api_key_chat_settings = ApiKeyChatSettings()


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


_DENY_RESPONSES = {
    "insufficient_credit": (402, "Insufficient credit.", "insufficient_credit"),
    "limit_reached": (429, "API key limit reached.", "key_limit_reached"),
    "unavailable": (503, "Service unavailable.", "service_unavailable"),
    "model_not_allowed": (400, "The model does not exist.", "model_not_found"),
}


async def requested_model(request: Request) -> str | None:
    try:
        body = await request.json()
    except Exception:
        return None
    model = body.get("model") if isinstance(body, dict) else None
    return model if isinstance(model, str) else None


async def authorize_api_key(
    request: Request, api_key: str, model: str | None = None
) -> tuple[JSONResponse | None, str | None, frozenset[str]]:
    if not internal_settings.internal_api_enabled:
        return (
            None,
            None,
            frozenset(
                api_key_chat_settings.default_key_capabilities
                or DEFAULT_API_KEY_CAPABILITIES
            ),
        )
    verdict = await internal_client.api_key_verify(
        request.state.httpx_client, api_key, model
    )
    if verdict is None:
        return (
            openai_error_response(503, "Service unavailable.", "service_unavailable"),
            None,
            frozenset(),
        )
    if not verdict.get("allowed", False):
        status, message, error_type = _DENY_RESPONSES.get(
            verdict.get("reason"), (401, "Invalid API key.", "invalid_api_key")
        )
        return openai_error_response(status, message, error_type), None, frozenset()
    return (
        None,
        verdict.get("model"),
        frozenset(verdict.get("key_capabilities") or DEFAULT_API_KEY_CAPABILITIES),
    )


class ApiKeyRoute(APIRoute):
    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        if scope["type"] == "http" and not is_api_key_request(Headers(scope=scope)):
            return Match.NONE, {}
        return super().matches(scope)


v1_router = APIRouter(route_class=ApiKeyRoute)


@v1_router.post("/chat/completions", response_model=None)
async def chat_completions(request: Request):
    api_key = api_key_from_authorization(request.headers.get("authorization"))
    if api_key is None:
        return openai_error_response(401, "Invalid API key.", "invalid_api_key")
    error, model, key_capabilities = await authorize_api_key(
        request, api_key, await requested_model(request)
    )
    if error is not None:
        return error
    return await handle_chat_completions(request, api_key, model, key_capabilities)
