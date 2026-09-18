from dataclasses import dataclass


@dataclass(frozen=True)
class AndroclesPrefetch:
    task_type: str | None
    matched_categories: frozenset[str] | None = None
