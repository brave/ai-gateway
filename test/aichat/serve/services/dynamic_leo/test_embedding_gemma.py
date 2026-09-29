import asyncio
from unittest.mock import patch

import pytest

from aichat.serve.services.dynamic_leo import embedding_gemma
from aichat.serve.services.dynamic_leo.config import DynamicLeoCategoryConfig
from aichat.serve.services.dynamic_leo.embedding_gemma import (
    EMBEDDING_GEMMA_STS_PREFIX,
    format_embeddinggemma_input,
)


def _cfg(**kwargs) -> DynamicLeoCategoryConfig:
    return DynamicLeoCategoryConfig(**kwargs)


def test_sts_prefix_matches_google_model_card_sentence_similarity():
    out = format_embeddinggemma_input("hello world")
    assert out.startswith(EMBEDDING_GEMMA_STS_PREFIX)
    assert out.endswith("hello world")


@pytest.mark.asyncio
async def test_generate_embeddings_with_timeout_returns_empty_on_timeout():
    async def slow_generate(*args, **kwargs):
        await asyncio.sleep(5)
        return {"data": []}

    with (
        patch.object(
            embedding_gemma.dynamic_leo_settings,
            "dynamic_leo_embedding_timeout_seconds",
            0.05,
        ),
        patch.object(
            embedding_gemma,
            "generate_embeddings",
            side_effect=slow_generate,
        ),
    ):
        result = await embedding_gemma.fetch_vectors_for_unique_phrases(
            "embedding_gemma",
            ["hello"],
        )

    assert result == {}


@pytest.mark.asyncio
async def test_embedding_match_reasons_for_texts_single_batched_call():
    calls = []

    async def fake_generate(model, input_data, **kwargs):
        calls.append(input_data)
        n = len(input_data) if isinstance(input_data, list) else 1
        return {"data": [{"embedding": [1.0, 0.0]} for _ in range(n)]}

    categories = {"coding": _cfg(similar=["write code"])}
    phrase_vectors = {"write code": [1.0, 0.0]}

    with patch.object(
        embedding_gemma, "generate_embeddings", side_effect=fake_generate
    ):
        reasons = await embedding_gemma.embedding_match_reasons_for_texts(
            [
                ("write code please", "embedding_last"),
                ("earlier turn", "embedding_prior"),
            ],
            "embedding_gemma",
            categories,
            phrase_vectors,
            0.7,
        )

    assert len(calls) == 1
    assert isinstance(calls[0], list) and len(calls[0]) == 2
    assert reasons == {"coding": frozenset({"embedding_last", "embedding_prior"})}


@pytest.mark.asyncio
async def test_embedding_match_reasons_for_texts_empty_when_no_model():
    reasons = await embedding_gemma.embedding_match_reasons_for_texts(
        [("hello", "embedding_last")],
        "",
        {"coding": _cfg(similar=["x"])},
        {"x": [1.0, 0.0]},
        0.7,
    )
    assert reasons == {}
