"""Extract plain text from the last user message (OpenAI-style messages)."""

from __future__ import annotations

from typing import Any


def extract_text_from_content(content: str | list | None) -> str:
    # Messages come in not as plain text but as a list of dictionaries, e.g.:
    #
    #   [
    #     {'text': 'hello', 'type': 'text'},
    #     {'type': 'text', 'text': 'This is the text of a web page: <page>some page context</page>.'},
    #   ]
    #
    # We need to normalise them to just the text combined, e.g.:
    #
    #   hello\n\nThis is the text of a web page: <page>some page context</page>.
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n\n".join(parts)

    return ""


def extract_last_user_message_text(messages: list[dict] | list[Any]) -> str:
    for message in reversed(messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role != "user":
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        return extract_text_from_content(content)
    return ""
