from unittest.mock import AsyncMock, patch

import pytest

from aichat.protocol.open_ai_protocol import Message
from aichat.serve.services.dynamic_leo import signals


@pytest.mark.asyncio
async def test_run_dynamic_leo_logs_keyword_matches():
    messages = [Message(role="user", content="write a python function")]

    mock_androcles = AsyncMock(return_value=[0.0] * 21)

    with (
        patch.object(
            signals.dynamic_leo_settings,
            "enable_dynamic_leo",
            True,
        ),
        patch.object(
            signals.dynamic_leo_settings,
            "dynamic_leo_config",
            {
                "coding": {
                    "keywords": ["python", "function"],
                }
            },
        ),
        patch.object(signals, "androcles_inference", mock_androcles),
        patch.object(signals.logger, "info") as mock_log,
    ):
        prefetch = await signals.run_dynamic_leo(messages)

    assert prefetch is not None
    assert prefetch.matched_categories == frozenset({"coding"})
    mock_log.assert_any_call(
        "Dynamic Leo categories: %s",
        {"coding": frozenset({"keywords_last"})},
    )


@pytest.mark.asyncio
async def test_run_dynamic_leo_continues_when_embedding_tasks_fail():
    messages = [Message(role="user", content="write a python script")]

    mock_androcles = AsyncMock(return_value=[0.0] * 21)

    with (
        patch.object(
            signals.dynamic_leo_settings,
            "enable_dynamic_leo",
            True,
        ),
        patch.object(
            signals.dynamic_leo_settings,
            "dynamic_leo_config",
            {
                "coding": {
                    "similar": ["write code"],
                    "labels": {"Coding": 0.8},
                }
            },
        ),
        patch.object(
            signals,
            "phrase_vectors_for_request",
            AsyncMock(return_value={"write code": [1.0, 0.0]}),
        ),
        patch.object(
            signals,
            "embedding_match_reasons_for_text",
            AsyncMock(side_effect=RuntimeError("embedding down")),
        ),
        patch.object(signals, "androcles_inference", mock_androcles),
    ):
        prefetch = await signals.run_dynamic_leo(messages)

    assert prefetch is not None
    mock_androcles.assert_awaited()
