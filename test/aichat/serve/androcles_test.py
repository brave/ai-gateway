import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aichat.serve.androcles import (
    ANDROCLES_TRIAGE_INDICES,
    _androcles_batch_probs_rows,
    androcles_inference,
    androcles_inference_batch,
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


@pytest.mark.asyncio
async def test_androcles_inference_batch_uses_probabilities_output_by_name():
    async def passthrough(**kwargs):
        resp = MagicMock()
        resp.json = MagicMock(
            return_value={
                "outputs": [
                    {"name": "predicted_label", "data": ["Multilingualism", "Coding"]},
                    {"name": "probabilities", "data": [[0.1, 0.2], [0.3, 0.4]]},
                ]
            }
        )
        return resp

    with patch(
        "aichat.serve.androcles.get_global_router",
        return_value=AsyncMock(allm_passthrough_route=passthrough),
    ):
        rows = await androcles_inference_batch(["a", "b"])

    assert rows == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_androcles_inference_batch_sends_single_request_and_splits_rows():
    captured = {}

    async def passthrough(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.json = MagicMock(
            return_value={"outputs": [{"data": [[0.1, 0.2], [0.3, 0.4]]}]}
        )
        return resp

    with patch(
        "aichat.serve.androcles.get_global_router",
        return_value=AsyncMock(allm_passthrough_route=passthrough),
    ):
        rows = await androcles_inference_batch(["a", "b"])

    assert rows == [[0.1, 0.2], [0.3, 0.4]]
    inp = captured["json"]["inputs"][0]
    assert inp["shape"] == [2, 1]
    assert inp["data"] == [["a", "b"]]


@pytest.mark.asyncio
async def test_androcles_inference_batch_returns_empty_for_no_text():
    async def passthrough(**kwargs):  # pragma: no cover - must not be called
        raise AssertionError("should not call router for empty input")

    with patch(
        "aichat.serve.androcles.get_global_router",
        return_value=AsyncMock(allm_passthrough_route=passthrough),
    ):
        rows = await androcles_inference_batch(["", "   "])

    assert rows == [None, None]


def test_androcles_batch_probs_rows_splits_flat_triton_output():
    flat = [0.1, 0.2, 0.3, 0.4]
    assert _androcles_batch_probs_rows(flat, 2) == [[0.1, 0.2], [0.3, 0.4]]
    nested = [[0.1, 0.2], [0.3, 0.4]]
    assert _androcles_batch_probs_rows(nested, 2) == nested


@pytest.mark.asyncio
async def test_androcles_inference_batch_none_on_row_count_mismatch():
    async def passthrough(**kwargs):
        resp = MagicMock()
        resp.json = MagicMock(return_value={"outputs": [{"data": [[0.1, 0.2]]}]})
        return resp

    with patch(
        "aichat.serve.androcles.get_global_router",
        return_value=AsyncMock(allm_passthrough_route=passthrough),
    ):
        rows = await androcles_inference_batch(["a", "b"])

    assert rows == [None, None]


@pytest.mark.asyncio
async def test_androcles_inference_batch_preserves_none_for_empty_slot():
    captured = {}

    async def passthrough(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.json = MagicMock(return_value={"outputs": [{"data": [[0.5, 0.6]]}]})
        return resp

    with patch(
        "aichat.serve.androcles.get_global_router",
        return_value=AsyncMock(allm_passthrough_route=passthrough),
    ):
        rows = await androcles_inference_batch(["", "only prior"])

    assert rows == [None, [0.5, 0.6]]
    assert captured["json"]["inputs"][0]["data"] == [["only prior"]]
