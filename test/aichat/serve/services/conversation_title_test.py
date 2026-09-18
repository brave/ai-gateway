from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from aichat.prompts.conversation_title import TYPE
from aichat.protocol.open_ai_protocol import (
    ConversationTitleContentPart,
    UserMessage,
)
from aichat.protocol.open_ai_protocol import (
    Request as OpenAIRequest,
)
from aichat.serve.services.conversation_title import (
    _title_part_has_no_text,
    complete_conversation_title_chat,
)


def _metric_value() -> float:
    return REGISTRY.get_sample_value("conversation_title_invalid_request_total") or 0.0


def test_title_part_has_no_text_true_when_title_part_empty():
    from aichat.protocol.open_ai_protocol import (
        ConversationTitleContentPart,
        UserMessage,
    )

    assert _title_part_has_no_text(
        [UserMessage(content=[ConversationTitleContentPart(type=TYPE, text="")])]
    )


def test_title_part_has_no_text_false_when_title_part_has_text():
    from aichat.protocol.open_ai_protocol import (
        ConversationTitleContentPart,
        UserMessage,
    )

    assert not _title_part_has_no_text(
        [UserMessage(content=[ConversationTitleContentPart(type=TYPE, text="hello")])]
    )


@pytest.mark.asyncio
async def test_complete_conversation_title_chat_returns_error_when_no_title_text():
    request = OpenAIRequest(
        model="caracal",
        messages=[
            UserMessage(
                content=[
                    ConversationTitleContentPart(type=TYPE, text=""),
                ],
            ),
        ],
        stream=True,
    )
    before = _metric_value()

    with patch(
        "aichat.serve.services.conversation_title.get_backend"
    ) as mock_get_backend:
        response = await complete_conversation_title_chat(
            request=request,
            prompts=MagicMock(),
            process_streaming_response=AsyncMock(),
        )

    assert response.status_code == 400
    assert "missing required text content" in response.body.decode()
    assert _metric_value() == before + 1
    mock_get_backend.assert_not_called()
