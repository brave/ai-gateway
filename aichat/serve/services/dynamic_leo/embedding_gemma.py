from __future__ import annotations

import asyncio
import logging
import math

from aichat.serve.services.dynamic_leo.config import DynamicLeoCategoryConfig
from aichat.serve.services.dynamic_leo.settings import dynamic_leo_settings
from aichat.serve.services.embeddings.generator import generate_embeddings

logger = logging.getLogger(__name__)

EMBEDDING_GEMMA_STS_PREFIX = "task: sentence similarity | query: "

_embedding_cache_sig: tuple[str, tuple[str, ...]] | None = None
_embedding_vectors: dict[str, list[float]] = {}
_embedding_lock = asyncio.Lock()


def format_embeddinggemma_input(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    return f"{EMBEDDING_GEMMA_STS_PREFIX}{stripped}"


def categories_have_similar_phrases(
    categories: dict[str, DynamicLeoCategoryConfig],
) -> bool:
    return any(phrase.strip() for cfg in categories.values() for phrase in cfg.similar)


def unique_similar_phrases_sorted(
    categories: dict[str, DynamicLeoCategoryConfig],
) -> list[str]:
    return sorted(
        {
            phrase
            for cfg in categories.values()
            for phrase in cfg.similar
            if phrase.strip()
        }
    )


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(x * y for x, y in zip(vec_a, vec_b, strict=False))
    na = math.sqrt(sum(x * x for x in vec_a))
    nb = math.sqrt(sum(y * y for y in vec_b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


async def _generate_embeddings_with_timeout(
    embedding_model_id: str,
    input_data: str | list[str],
) -> dict:
    timeout = dynamic_leo_settings.dynamic_leo_embedding_timeout_seconds
    coro = generate_embeddings(embedding_model_id, input_data)
    if timeout > 0:
        return await asyncio.wait_for(coro, timeout=timeout)
    return await coro


async def fetch_vectors_for_unique_phrases(
    embedding_model_id: str,
    unique_phrases: list[str],
) -> dict[str, list[float]]:
    if not unique_phrases:
        return {}

    vectors: dict[str, list[float]] = {}
    batch_size = 32
    for start in range(0, len(unique_phrases), batch_size):
        batch = unique_phrases[start : start + batch_size]
        batch_submit = [format_embeddinggemma_input(phrase) for phrase in batch]
        try:
            response = await _generate_embeddings_with_timeout(
                embedding_model_id, batch_submit
            )
        except TimeoutError:
            logger.warning(
                "Dynamic Leo phrase embedding batch timed out after %.3fs",
                dynamic_leo_settings.dynamic_leo_embedding_timeout_seconds,
            )
            continue
        except Exception as exc:
            logger.warning("Dynamic Leo phrase embedding batch failed: %s", exc)
            continue

        rows = response.get("data") or []
        if len(rows) != len(batch):
            logger.warning(
                "Unexpected embedding batch size: got %d expected %d",
                len(rows),
                len(batch),
            )
            continue

        for phrase, row in zip(batch, rows, strict=False):
            emb = row.get("embedding")
            if isinstance(emb, list) and emb:
                vectors[phrase] = emb

    return vectors


async def phrase_vectors_for_request(
    embedding_model_id: str,
    unique_sorted: list[str],
) -> dict[str, list[float]]:
    if not unique_sorted:
        return {}

    global _embedding_cache_sig, _embedding_vectors
    async with _embedding_lock:
        sig = (embedding_model_id, tuple(unique_sorted))
        if _embedding_cache_sig != sig:
            _embedding_vectors = await fetch_vectors_for_unique_phrases(
                embedding_model_id, unique_sorted
            )
            _embedding_cache_sig = sig
        return _embedding_vectors


def embedding_reasons_for_query_vec(
    query_vec: list[float],
    categories: dict[str, DynamicLeoCategoryConfig],
    phrase_vectors: dict[str, list[float]],
    threshold: float,
    reason_tag: str,
) -> dict[str, frozenset[str]]:
    outcome: dict[str, frozenset[str]] = {}
    for cat_id, cfg in categories.items():
        best = 0.0
        for phrase in cfg.similar:
            stripped = phrase.strip()
            vec = phrase_vectors.get(stripped)
            if vec is None:
                continue
            best = max(best, cosine_similarity(query_vec, vec))
        if best >= threshold:
            outcome[cat_id] = frozenset({reason_tag})
    return outcome


async def embedding_match_reasons_for_text(
    fused_text: str,
    embedding_model_id: str,
    categories: dict[str, DynamicLeoCategoryConfig],
    phrase_vectors: dict[str, list[float]],
    threshold: float,
    *,
    reason_tag: str,
) -> dict[str, frozenset[str]]:
    outcome: dict[str, frozenset[str]] = {}
    if not embedding_model_id or not fused_text.strip() or not phrase_vectors:
        return outcome
    try:
        query_body = format_embeddinggemma_input(fused_text)
        query_resp = await _generate_embeddings_with_timeout(
            embedding_model_id, query_body
        )
        query_rows = query_resp.get("data") or []
        if not query_rows:
            return outcome
        query_vec = query_rows[0].get("embedding")
        if not isinstance(query_vec, list) or not query_vec:
            return outcome
    except TimeoutError:
        logger.warning(
            "Dynamic Leo query embedding timed out after %.3fs",
            dynamic_leo_settings.dynamic_leo_embedding_timeout_seconds,
        )
        return outcome
    except Exception as exc:
        logger.warning("Dynamic Leo query embedding failed: %s", exc)
        return outcome

    return embedding_reasons_for_query_vec(
        query_vec, categories, phrase_vectors, threshold, reason_tag
    )
