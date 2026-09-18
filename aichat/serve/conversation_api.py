import fastapi
from fastapi.responses import JSONResponse

from aichat.protocol.leo_api_protocol import (
    ConversationRequest,
    ConversationResponseEvent,
)
from aichat.serve.constants import CONVERSATION_API_DEPRECATION_MESSAGES
from aichat.serve.utils import SSEResponse

router = fastapi.APIRouter()


def get_conversation_deprecation_message(system_language: str | None) -> str:
    """
    Builds the "please upgrade" notice for the deprecated /conversation endpoint.
    Localized when we recognize the client's system language; English is always
    included so the notice stays understandable regardless.
    """
    english_message = CONVERSATION_API_DEPRECATION_MESSAGES["en"]
    language = (system_language or "").split("-")[0].split("_")[0].lower()
    localized_message = CONVERSATION_API_DEPRECATION_MESSAGES.get(language)

    if not localized_message or language == "en":
        return english_message

    return f"{localized_message}\n\n{english_message}"


async def _deprecation_notice_stream(model: str, message: str):
    yield ConversationResponseEvent(
        type="completion",
        model=model,
        completion=message,
        stop_reason="stop_sequence",
    ).json(exclude_none=True)


@router.post("/conversation")
async def create_conversation_completion(request: ConversationRequest):
    """
    The /conversation endpoint is deprecated in favor of /chat/completions.
    It's kept only so older clients get a clear upgrade message instead of a
    broken response; no auth, rate limiting, or LLM calls happen here.
    """
    message = get_conversation_deprecation_message(request.system_language)

    if not request.stream:
        return JSONResponse(
            content=ConversationResponseEvent(
                type="completion",
                model=request.model,
                completion=message,
                stop_reason="stop_sequence",
            ).model_dump(exclude_none=True),
            status_code=200,
        )

    return SSEResponse(_deprecation_notice_stream(request.model, message))
