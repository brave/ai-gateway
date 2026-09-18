from aichat.protocol.open_ai_protocol import Message


def _plain_text_parts_from_content(content: object) -> list[str]:
    if content is None:
        return []

    if isinstance(content, str):
        s = content.strip()
        return [s] if s else []

    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            ptype = getattr(part, "type", None)
            if (
                ptype == "text"
                or str(ptype).lower() == "text"
                or (hasattr(part, "type") and not ptype)
            ):
                t = getattr(part, "text", None)
                if isinstance(t, str) and t.strip():
                    chunks.append(t.strip())
        return chunks

    return []


def extract_last_n_user_plain_text_turns(messages: list[Message], n: int) -> list[str]:
    if n < 1:
        return []

    user_turns: list[str] = []
    for msg in reversed(messages):
        if getattr(msg, "role", None) != "user":
            continue
        parts = _plain_text_parts_from_content(getattr(msg, "content", None))
        if not parts:
            continue
        user_turns.append("\n".join(parts))
        if len(user_turns) >= n:
            break

    user_turns.reverse()
    return user_turns


def fuse_turns_for_classifier(turns: list[str]) -> str:
    return "\n\n".join(t.strip() for t in turns if t.strip()).strip()


def last_message_and_prior_fused(turns: list[str]) -> tuple[str, str]:
    """Latest user turn (trimmed) and prior turns fused with ``fuse_turns_for_classifier``."""
    if not turns:
        return "", ""
    last = turns[-1].strip()
    prior = fuse_turns_for_classifier(turns[:-1]) if len(turns) > 1 else ""
    return last, prior
