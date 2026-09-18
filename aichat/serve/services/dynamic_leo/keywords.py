from __future__ import annotations

import json

import ahocorasick

from aichat.serve.services.dynamic_leo.config import DynamicLeoCategoryConfig

_automaton_cache_sig: str | None = None
_automaton_cache: ahocorasick.Automaton | None = None


def categories_have_keywords(
    categories: dict[str, DynamicLeoCategoryConfig],
) -> bool:
    return any(kw.strip() for cfg in categories.values() for kw in cfg.keywords)


def _keywords_cache_sig(
    categories: dict[str, DynamicLeoCategoryConfig],
) -> str:
    return json.dumps(
        {
            cat_id: sorted(kw.strip().lower() for kw in cfg.keywords if kw.strip())
            for cat_id, cfg in sorted(categories.items())
        },
        sort_keys=True,
    )


def build_automaton(
    categories: dict[str, DynamicLeoCategoryConfig],
) -> ahocorasick.Automaton:
    automaton = ahocorasick.Automaton()
    for cat_id, cfg in categories.items():
        for keyword in cfg.keywords:
            stripped = keyword.strip()
            if stripped:
                lowered = stripped.lower()
                automaton.add_word(lowered, (cat_id, lowered))
    automaton.make_automaton()
    return automaton


def automaton_for_categories(
    categories: dict[str, DynamicLeoCategoryConfig],
) -> ahocorasick.Automaton:
    global _automaton_cache_sig, _automaton_cache
    sig = _keywords_cache_sig(categories)
    if sig == _automaton_cache_sig and _automaton_cache is not None:
        return _automaton_cache
    _automaton_cache_sig = sig
    _automaton_cache = build_automaton(categories)
    return _automaton_cache


def _has_word_boundaries(text: str, end_index: int, keyword_len: int) -> bool:
    start_index = end_index - keyword_len + 1
    if start_index > 0 and text[start_index - 1].isalnum():
        return False
    return not (end_index + 1 < len(text) and text[end_index + 1].isalnum())


def keyword_match_reasons_for_text(
    text: str,
    automaton: ahocorasick.Automaton,
    *,
    reason_tag: str,
) -> dict[str, frozenset[str]]:
    if not text.strip():
        return {}

    outcome: dict[str, frozenset[str]] = {}
    lowered = text.lower()
    for end_index, (cat_id, keyword) in automaton.iter(lowered):
        if not _has_word_boundaries(lowered, end_index, len(keyword)):
            continue
        outcome[cat_id] = frozenset({reason_tag})
    return outcome
