import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from aichat.serve.androcles import (
    ANDROCLES_TRIAGE_INDICES,
    androcles_inference,
    task_type_from_androcles_probabilities,
)


def _triage_prob_vector(base: float = 0.02) -> list[float]:
    return [base] * (max(ANDROCLES_TRIAGE_INDICES.values()) + 1)


def test_task_from_single_label_coding_index():
    v = _triage_prob_vector()
    v[3] = 0.95
    assert task_type_from_androcles_probabilities(v) == "coding"


def test_task_from_single_label_language_index():
    v = _triage_prob_vector()
    v[10] = 0.95
    assert task_type_from_androcles_probabilities(v) == "language"


def test_task_from_single_label_vision_index():
    v = _triage_prob_vector()
    v[8] = 0.95
    assert task_type_from_androcles_probabilities(v) == "vision"


def test_task_from_short_vector_returns_none():
    assert task_type_from_androcles_probabilities([0.1, 0.1, 0.95, 0.1]) is None


@pytest.mark.asyncio
async def test_androcles_inference_returns_none_on_timeout():
    async def slow_call(**kwargs):
        await asyncio.sleep(5)
        return AsyncMock(json=lambda: {"outputs": [{"data": [0.1]}]})()

    with patch(
        "aichat.serve.androcles.get_global_router",
        return_value=AsyncMock(allm_passthrough_route=slow_call),
    ):
        result = await androcles_inference("hello", timeout_seconds=0.05)

    assert result is None
