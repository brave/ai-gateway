"""Shared helpers for mapping routes to logical API version (V1 vs V2)."""


def api_version_from_path(path: str) -> str:
    """
    Infer API version from a URL path or FastAPI route template (e.g. handler string).

    V2 is OpenAI-style chat completions; V1 is the legacy conversation API.
    """
    if "/chat/completions" in path:
        return "v2"
    if "/conversation" in path:
        return "v1"
    return "unknown"
