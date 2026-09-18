"""Prompt templates for conversation compaction summaries."""

JAGUAR_COMPACTION_SUMMARY_PROMPT = """compact
target tokens: {max_tokens}
style: {style}
preserve: {preserve}
focus: {focus}
recency: {recency}
language: {language}
"""

COMPACTION_FALLBACK_SUMMARY_PROMPT = """Summarise the following conversation so that an LLM can continue a conversation with a user.
Note reference images are not included in the conversation below.
Output Format:

- Current Goal: [One sentence summary of the overarching request from the user]
- Key Actions Taken By Assistant: [List of any actions taken so far]
- Key Actions of Questions asked by User: [List of any supplementary actions, requests or questions from the user]
- Pending Actions: [List of any actions that must happen next]

Constraints:
- Preserve exact values for names, dates, code snippets, numbers, or technical constraints
- Be concise and maximise information density; ensure that your summary is < {max_tokens} words. Remove all conversational filler.
- Always distinguish between what the user asked and the assistant responded.

Conversation:
<conversation_start>
{content}
<conversation_end>

Summary:
"""


def build_jaguar_compaction_prompt(
    max_tokens: int, style: str, preserve: str, focus: str, recency: str, language: str
) -> str:
    """Build the Jaguar compaction model prompt."""
    return JAGUAR_COMPACTION_SUMMARY_PROMPT.format(
        max_tokens=max_tokens,
        style=style,
        preserve=preserve,
        focus=focus,
        recency=recency,
        language=language,
    )


def build_fallback_compaction_prompt(content: str, max_tokens: int) -> str:
    """Build the standard summarization prompt used by the fallback model."""
    return COMPACTION_FALLBACK_SUMMARY_PROMPT.format(
        content=content, max_tokens=max_tokens
    )
