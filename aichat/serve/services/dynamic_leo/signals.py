from __future__ import annotations

import asyncio
import json
import logging

from aichat.protocol.open_ai_protocol import Message
from aichat.serve.androcles import (
    androcles_inference,
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
    embedding_match_reasons_for_text,
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

    async def _androcles_if_nonempty(text: str) -> list[float] | None:
        if not text.strip():
            return None
        return await androcles_inference(
            text,
            timeout_seconds=dynamic_leo_settings.dynamic_leo_androcles_timeout_seconds,
        )

    coros: list = [
        _androcles_if_nonempty(last_text),
        _androcles_if_nonempty(prior_text),
    ]
    if need_embed:
        em = embedding_model_id or ""
        coros.extend(
            [
                embedding_match_reasons_for_text(
                    last_text,
                    em,
                    categories,
                    phrase_vectors,
                    thresh,
                    reason_tag="embedding_last",
                ),
                embedding_match_reasons_for_text(
                    prior_text,
                    em,
                    categories,
                    phrase_vectors,
                    thresh,
                    reason_tag="embedding_prior",
                ),
            ]
        )
    results = await asyncio.gather(*coros, return_exceptions=True)
    probs_last = _gather_result_or_default(results[0], None)
    probs_prior = _gather_result_or_default(results[1], None)
    if need_embed:
        embed_last = _gather_result_or_default(results[2], {})
        embed_prior = _gather_result_or_default(results[3], {})
        embed_reasons = _merge_reason_dicts(embed_last, embed_prior)
    else:
        embed_reasons = {}

    if categories:
        lbl_reasons = label_reasons(categories, probs_last, probs_prior)
    else:
        lbl_reasons = {}

    merged_reasons = _merge_reason_dicts(kw_reasons, lbl_reasons)
    merged_reasons = _merge_reason_dicts(merged_reasons, embed_reasons)

    if merged_reasons:
        logger.info("Dynamic Leo categories: %s", merged_reasons)

    merged_probs = _merge_probs_for_triage(probs_last, probs_prior)
    return AndroclesPrefetch(
        task_type=task_type_from_androcles_probabilities(merged_probs),
        matched_categories=frozenset(merged_reasons.keys()),
    )
