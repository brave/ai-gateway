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
            "embedding_match_reasons_for_texts",
            AsyncMock(side_effect=RuntimeError("embedding down")),
        ),
        patch.object(signals, "androcles_inference", mock_androcles),
    ):
        prefetch = await signals.run_dynamic_leo(messages)

    assert prefetch is not None
    mock_androcles.assert_awaited()


@pytest.mark.asyncio
async def test_batched_makes_single_androcles_and_single_embedding_call():
    messages = [
        Message(role="user", content="write some code"),
        Message(role="assistant", content="sure"),
        Message(role="user", content="now add tests"),
    ]

    mock_androcles_batch = AsyncMock(return_value=[[0.1] * 21, [0.2] * 21])
    mock_embed_batch = AsyncMock(return_value={})

    with (
        patch.object(signals.dynamic_leo_settings, "enable_dynamic_leo", True),
        patch.object(signals.dynamic_leo_settings, "dynamic_leo_batch_inference", True),
        patch.object(
            signals.dynamic_leo_settings,
            "dynamic_leo_config",
            {"coding": {"similar": ["write code"], "labels": {"Coding": 0.8}}},
        ),
        patch.object(
            signals,
            "phrase_vectors_for_request",
            AsyncMock(return_value={"write code": [1.0, 0.0]}),
        ),
        patch.object(signals, "_embedding_model_id", return_value="embedding_gemma"),
        patch.object(signals, "embedding_match_reasons_for_texts", mock_embed_batch),
        patch.object(signals, "androcles_inference_batch", mock_androcles_batch),
    ):
        prefetch = await signals.run_dynamic_leo(messages)

    assert prefetch is not None
    mock_androcles_batch.assert_awaited_once()
    mock_embed_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_batched_recovers_category_from_last_text_embedding():
    messages = [Message(role="user", content="write some code")]

    with (
        patch.object(signals.dynamic_leo_settings, "enable_dynamic_leo", True),
        patch.object(signals.dynamic_leo_settings, "dynamic_leo_batch_inference", True),
        patch.object(
            signals.dynamic_leo_settings,
            "dynamic_leo_embedding_similarity_threshold",
            0.7,
        ),
        patch.object(
            signals.dynamic_leo_settings,
            "dynamic_leo_config",
            {"coding": {"similar": ["write code"]}},
        ),
        patch.object(
            signals,
            "phrase_vectors_for_request",
            AsyncMock(return_value={"write code": [1.0, 0.0]}),
        ),
        patch.object(signals, "_embedding_model_id", return_value="embedding_gemma"),
        patch.object(
            signals,
            "androcles_inference_batch",
            AsyncMock(return_value=[None, None]),
        ),
        patch(
            "aichat.serve.services.dynamic_leo.embedding_gemma.generate_embeddings",
            AsyncMock(
                return_value={
                    "data": [
                        {"embedding": [1.0, 0.0]},
                    ]
                }
            ),
        ),
    ):
        prefetch = await signals.run_dynamic_leo(messages)

    assert prefetch is not None
    assert prefetch.matched_categories == frozenset({"coding"})


@pytest.mark.asyncio
async def test_batched_path_used_when_flag_on_legacy_not_called():
    messages = [Message(role="user", content="write a python function")]

    mock_androcles = AsyncMock(return_value=[0.0] * 21)
    mock_androcles_batch = AsyncMock(return_value=[[0.0] * 21, None])

    with (
        patch.object(signals.dynamic_leo_settings, "enable_dynamic_leo", True),
        patch.object(signals.dynamic_leo_settings, "dynamic_leo_batch_inference", True),
        patch.object(
            signals.dynamic_leo_settings,
            "dynamic_leo_config",
            {"coding": {"keywords": ["python", "function"]}},
        ),
        patch.object(signals, "androcles_inference", mock_androcles),
        patch.object(signals, "androcles_inference_batch", mock_androcles_batch),
    ):
        prefetch = await signals.run_dynamic_leo(messages)

    assert prefetch is not None
    assert prefetch.matched_categories == frozenset({"coding"})
    mock_androcles.assert_not_awaited()
    mock_androcles_batch.assert_awaited_once()
