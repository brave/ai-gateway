"""Citation formatting utilities for deep research answers.

This module handles building citation maps and formatting answers
with inline citation links and source sections.
"""

import re
from urllib.parse import urlparse


def encode_url_for_markdown(url: str) -> str:
    """Encode URL for use in markdown links (handle spaces and special chars)."""
    return url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def build_citation_map(citations: list[dict]) -> dict[int, dict]:
    """Build a map of citation number -> {url, encoded_url, hostname}."""
    citation_map = {}
    for c in citations:
        num = c.get("number")
        url = c.get("url", "")
        if num is not None and url:
            try:
                parsed = urlparse(url)
                hostname = parsed.hostname or parsed.netloc or url
            except Exception:
                hostname = url
            citation_map[num] = {
                "url": url,
                "encoded_url": encode_url_for_markdown(url),
                "hostname": hostname,
            }
    return citation_map


def format_answer_with_citations(answer: str, citations: list[dict]) -> str:
    """Format a deep research answer preserving [N] citation markers.

    Keeps [N] inline markers as-is so the browser can resolve them against
    the webSources data (which is emitted before this answer chunk).
    Adds a space before any citation marker that is directly attached to a word.
    """
    # Add space before citation markers (e.g., "text[1]" -> "text [1]")
    formatted = re.sub(r"(\w|\S)\[(\d+)\]", r"\1 [\2]", answer)

    # Add separator before the report
    return f"---\n\n{formatted}"


def format_citations_plaintext(answer: str, citations: list[dict]) -> str:
    """Format an answer with plaintext citation references.

    Used as a fallback when the deep research handler is not available.
    """
    citations_text = ""
    if citations:
        citations_text = "\n\nSources:\n"
        citation_map = build_citation_map(citations)
        for c in citations:
            num = c.get("number", "")
            info = citation_map.get(num, {})
            url = info.get("url", c.get("url", ""))
            if url:
                citations_text += f"[{num}] {url}\n"
    return f"{answer}{citations_text}"
