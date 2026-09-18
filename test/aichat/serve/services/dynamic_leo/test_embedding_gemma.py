import asyncio
from unittest.mock import patch

import pytest

from aichat.serve.services.dynamic_leo import embedding_gemma
from aichat.serve.services.dynamic_leo.embedding_gemma import (
    EMBEDDING_GEMMA_STS_PREFIX,
    format_embeddinggemma_input,
)


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
