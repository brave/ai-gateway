from __future__ import annotations

from aichat.serve.services.dynamic_leo.config import DynamicLeoCategoryConfig

_LABEL_BY_INDEX: dict[int, str] = {
    0: "Brave",
    1: "Browser",
    2: "Choice",
    3: "Coding",
    4: "Data Analysis",
    5: "Diagrams",
    6: "Fact Checking",
    7: "Finance",
    8: "Image Generation",
    9: "Math / Calculations",
    10: "Multilingualism",
    11: "News",
    12: "Recommendation",
    13: "Sports",
    14: "Structured Writing",
    15: "Summarisation",
    16: "Thinking",
    17: "Time-critical",
    18: "Translation",
    19: "Travel Planning",
    20: "Weather",
}

ANDROCLES_NUM_LABELS = len(_LABEL_BY_INDEX)

_LABEL_LOWER_TO_INDEX: dict[str, int] = {
    " ".join(label.strip().split()).lower(): idx
    for idx, label in _LABEL_BY_INDEX.items()
}


def androcles_label_index(name: str) -> int | None:
    if not name or not str(name).strip():
        return None
    normalized = " ".join(str(name).strip().split())
    return _LABEL_LOWER_TO_INDEX.get(normalized.lower())


def category_matches_labels(
    cat_cfg: DynamicLeoCategoryConfig,
    probs: list[float] | None,
) -> bool:
    if not cat_cfg.labels or not probs:
        return False
    for raw_label, threshold in cat_cfg.labels.items():
        idx = androcles_label_index(str(raw_label))
        if idx is None or idx >= len(probs):
            continue
        if probs[idx] >= threshold:
            return True
    return False


def categories_from_label_probs(
    categories: dict[str, DynamicLeoCategoryConfig],
    probs_last: list[float] | None,
    probs_prior: list[float] | None,
) -> set[str]:
    matched: set[str] = set()
    for cat_id, cfg in categories.items():
        if category_matches_labels(cfg, probs_last) or category_matches_labels(
            cfg, probs_prior
        ):
            matched.add(cat_id)
    return matched


def label_reasons(
    categories: dict[str, DynamicLeoCategoryConfig],
    probs_last: list[float] | None,
    probs_prior: list[float] | None,
) -> dict[str, frozenset[str]]:
    return {
        cat_id: frozenset({"labels"})
        for cat_id in categories_from_label_probs(categories, probs_last, probs_prior)
    }
