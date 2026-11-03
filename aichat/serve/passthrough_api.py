from fastapi import APIRouter
from starlette.requests import Request

from aichat.serve.common_api import require_internal_models_api_key
from aichat.serve.services.passthrough import handle_passthrough

v1_router = APIRouter()


@v1_router.post("/passthrough", response_model=None)
async def v1_passthrough(request: Request):
    """
    Passthrough endpoint that forwards the request directly to the litellm backend
    and streams the raw OpenAI-compatible chunks back to the caller.
    Supports both streaming and non-streaming responses.
    """
    if auth_err := require_internal_models_api_key(
        request.headers.get("Authorization")
    ):
        return auth_err
    return await handle_passthrough(request)
