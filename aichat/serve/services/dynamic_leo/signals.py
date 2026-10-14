from __future__ import annotations

import asyncio
import json
import logging

from aichat.protocol.open_ai_protocol import Message
from aichat.serve.androcles import (
    androcles_inference,
    androcles_inference_batch,
    task_type_from_androcles_probabilities,
)
from aichat.serve.services.androcles_prefetch import AndroclesPrefetch
from aichat.serve.services.dynamic_leo.androcles_labels import (
    label_reasons,
)
from aichat.serve.services.dynamic_leo.config import (
    DynamicLeoCategoryConfig,
    parse_dynamic_leo_categories,
)
from aichat.serve.services.dynamic_leo.embedding_gemma import (
    categories_have_similar_phrases,
    embedding_match_reasons_for_texts,
    phrase_vectors_for_request,
    unique_similar_phrases_sorted,
)
from aichat.serve.services.dynamic_leo.keywords import (
    automaton_for_categories,
    categories_have_keywords,
    keyword_match_reasons_for_text,
)
from aichat.serve.services.dynamic_leo.settings import dynamic_leo_settings
from aichat.serve.services.dynamic_leo.user_text import (
    extract_last_n_user_plain_text_turns,
    fuse_turns_for_classifier,
    last_message_and_prior_fused,
)
from aichat.serve.services.model_settings import model_settings
from aichat.serve.user_message_text import extract_last_user_message_text

logger = logging.getLogger(__name__)

_categories_sig: str | None = None
_categories_cache: dict[str, DynamicLeoCategoryConfig] | None = None


def _categories_from_config(
    raw_config: dict,
) -> dict[str, DynamicLeoCategoryConfig] | None:
    global _categories_sig, _categories_cache
    sig = json.dumps(raw_config, sort_keys=True)
    if sig == _categories_sig and _categories_cache is not None:
        return _categories_cache
    categories = parse_dynamic_leo_categories(raw_config)
    if not categories:
        return None
    _categories_sig = sig
    _categories_cache = categories
    return categories


def _merge_probs_for_triage(
    probs_last: list[float] | None,
    probs_prior: list[float] | None,
) -> list[float] | None:
    if probs_last is None:
        return probs_prior
    elif probs_prior is None or len(probs_last) != len(probs_prior):
        return probs_last
    else:
        return [max(a, b) for a, b in zip(probs_last, probs_prior, strict=False)]


def _merge_reason_dicts(
    a: dict[str, frozenset[str]],
    b: dict[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    out: dict[str, set[str]] = {}
    for d in (a, b):
        for k, rs in d.items():
            out.setdefault(k, set()).update(rs)
    return {k: frozenset(v) for k, v in out.items()}


def _gather_result_or_default(result, default):
    if isinstance(result, BaseException):
        logger.warning("Dynamic Leo task failed: %s", result)
        return default
    return result


def _androcles_probs_from_batch(
    probs: list | tuple | None,
) -> tuple[list[float] | None, list[float] | None]:
    if isinstance(probs, (list, tuple)) and len(probs) >= 2:
        return probs[0], probs[1]
    return None, None


def _embedding_model_id() -> str | None:
    explicit = dynamic_leo_settings.dynamic_leo_embedding_model.strip()
    if explicit:
        info = model_settings.models.get(explicit)
        if info and info.get("type") == "embedding":
            return explicit
        logger.warning(
            "Dynamic Leo embedding model %r missing or not type=embedding; skipping similarity",
            explicit,
        )

    for name, info in model_settings.models.items():
        if info.get("type") == "embedding":
            return name
    return None


async def run_dynamic_leo(messages: list[Message]) -> AndroclesPrefetch | None:
    if not dynamic_leo_settings.enable_dynamic_leo:
        return None

    last_n = dynamic_leo_settings.dynamic_leo_last_n_user_turns
    turns = extract_last_n_user_plain_text_turns(messages, last_n)
    fused = fuse_turns_for_classifier(turns)
    if not fused:
        return None

    last_text, prior_text = last_message_and_prior_fused(turns)

    categories = _categories_from_config(dynamic_leo_settings.dynamic_leo_config)

    if categories and categories_have_keywords(categories):
        automaton = automaton_for_categories(categories)
        kw_reasons = _merge_reason_dicts(
            keyword_match_reasons_for_text(
                last_text, automaton, reason_tag="keywords_last"
            ),
            keyword_match_reasons_for_text(
                prior_text, automaton, reason_tag="keywords_prior"
            ),
        )
    else:
        kw_reasons = {}

    thresh = dynamic_leo_settings.dynamic_leo_embedding_similarity_threshold
    need_embed = bool(categories) and categories_have_similar_phrases(categories)

    phrase_vectors: dict[str, list[float]] = {}
    embedding_model_id: str | None = None
    if need_embed:
        unique_sorted = unique_similar_phrases_sorted(categories)
        embedding_model_id = _embedding_model_id()
        if unique_sorted and embedding_model_id:
            try:
                phrase_vectors = await phrase_vectors_for_request(
                    embedding_model_id, unique_sorted
                )
            except Exception as exc:
                logger.warning("Dynamic Leo phrase embedding prefetch failed: %s", exc)
                phrase_vectors = {}

    do_embed = need_embed and bool(phrase_vectors)
    probs_last, probs_prior, embed_reasons = await _gather_androcles_and_embeddings(
        last_text,
        prior_text,
        batch_androcles=dynamic_leo_settings.dynamic_leo_batch_inference,
        do_embed=do_embed,
        embedding_model_id=embedding_model_id,
        categories=categories or {},
        phrase_vectors=phrase_vectors,
        thresh=thresh,
    )

    if categories:
        lbl_reasons = label_reasons(categories, probs_last, probs_prior)
    else:
        lbl_reasons = {}

    merged_reasons = _merge_reason_dicts(kw_reasons, lbl_reasons)
    merged_reasons = _merge_reason_dicts(merged_reasons, embed_reasons)

    if merged_reasons:
        logger.info("Dynamic Leo categories: %s", merged_reasons)

    merged_probs = _merge_probs_for_triage(probs_last, probs_prior)
    analytics_text = extract_last_user_message_text(messages)
    # Androcles runs on last_text; analytics POST uses analytics_text (may differ
    # for multi-part user content). Only passthrough when they match exactly.
    analytics_probs = (
        probs_last if analytics_text and analytics_text == last_text else None
    )
    return AndroclesPrefetch(
        task_type=task_type_from_androcles_probabilities(merged_probs),
        matched_categories=frozenset(merged_reasons.keys()),
        last_user_androcles_probabilities=analytics_probs,
    )


async def _maybe_androcles(text: str) -> list | None:
    if not text.strip():
        return None
    return await androcles_inference(
        text,
        timeout_seconds=dynamic_leo_settings.dynamic_leo_androcles_timeout_seconds,
    )


async def _gather_androcles_and_embeddings(
    last_text: str,
    prior_text: str,
    *,
    batch_androcles: bool,
    do_embed: bool,
    embedding_model_id: str | None,
    categories: dict[str, DynamicLeoCategoryConfig],
    phrase_vectors: dict[str, list[float]],
    thresh: float,
) -> tuple[list[float] | None, list[float] | None, dict[str, frozenset[str]]]:
    em = embedding_model_id or ""
    timeout = dynamic_leo_settings.dynamic_leo_androcles_timeout_seconds
    tasks: list = []
    if batch_androcles:
        tasks.append(
            androcles_inference_batch([last_text, prior_text], timeout_seconds=timeout)
        )
    else:
        tasks.extend([_maybe_androcles(last_text), _maybe_androcles(prior_text)])

    embed_index: int | None = None
    if do_embed:
        embed_index = len(tasks)
        tasks.append(
            embedding_match_reasons_for_texts(
                [(last_text, "embedding_last"), (prior_text, "embedding_prior")],
                em,
                categories,
                phrase_vectors,
                thresh,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    if batch_androcles:
        probs_last, probs_prior = _androcles_probs_from_batch(
            _gather_result_or_default(results[0], None)
        )
    else:
        probs_last = _gather_result_or_default(results[0], None)
        probs_prior = _gather_result_or_default(results[1], None)

    embed_reasons = (
        _gather_result_or_default(results[embed_index], {})
        if embed_index is not None
        else {}
    )
    return probs_last, probs_prior, embed_reasons
