from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from aichat.protocol.open_ai_protocol import (
    Capability,
    CapabilityOptions,
    has_capability,
)
from aichat.serve.server_settings import server_settings

from .prompt import Prompt

_SYSTEM_PROMPT_BODY_PATH = Path(__file__).parent / "leo_system_body.md"

# Included only when the client does NOT advertise the `math_ml` capability.
# MathML rendering landed in brave-core 1.96; older clients would show raw
# markup, so those clients are told to emit plain text math instead.
DISABLE_MATH_ML = """- **Math**: Do not emit MathML (`<math>`, `<mrow>`, etc.) in answers. Use plain text and Unicode (×, π, x², √(x)) or ```text blocks for steps. MathML source is allowed only inside a code fence when the user asks about MathML itself.

"""

QWEN_SEARCH_DIRECTIVE = """
**Qwen:** Apply the search rules above strictly—run a search tool before the final answer whenever they apply; do not answer time-sensitive facts from memory alone.
"""

CONTENT_AGENT_INTRO = (
    " When this request includes the **content_agent** capability, you may also perform "
    "user-requested agentic browsing (supervised browser automation)—see below."
)

CONTENT_AGENT_SCOPE = (
    " With **content_agent**, user-requested agentic browsing is in scope."
)

CONTENT_AGENT_DIRECTIVE = """
**Agentic browsing (content_agent only):** This mode is active only when the client set the `content_agent` capability and supplied browser automation tools. Act only on tasks the user clearly requested (multi-step research, forms, checkout, comparisons). Prefer search tools for plain factual lookups unless on-page action is required. Confirm before sensitive or irreversible steps; use `user_choice_tool` when offered. Do not start agentic browsing without a user request; in ordinary chat Leo, do not claim to click, navigate, or fill forms.

"""

DEEP_RESEARCH_DIRECTIVE = """
**Deep research (deep_research capability only):** Call `deep_research` only when this request includes that tool and the user explicitly wants a slow, in-depth cited report—not for questions one search can answer.

"""


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# West → east anchors. At most two extras vs UTC: the westernmost anchor still on
# the previous calendar day, and the westernmost anchor already on the next calendar day.
# Order is important here: must be from WEST -> EAST
_DATE_ANCHOR_ZONES_WEST_TO_EAST: tuple[tuple[str, str], ...] = (
    ("America/Los_Angeles", "Los Angeles"),
    ("America/New_York", "New York"),
    ("Europe/Warsaw", "Warsaw"),
    ("Asia/Kolkata", "Kolkata"),
    ("Asia/Tokyo", "Tokyo"),
)


def _instant_as_utc(dt: datetime) -> datetime:
    """Normalize to UTC. Naive datetimes are treated as UTC (stable for tests and UTC servers)."""
    if dt.tzinfo is not None:
        return dt.astimezone(UTC)
    return dt.replace(tzinfo=UTC)


# (UTC year, month, day, hour, half-hour index 0|1) -> formatted prompt date line.
# Half-hour buckets: India is UTC+5:30 and flips local date on :30 UTC, so a pure hourly
# cache can show the wrong eastern anchor for part of an hour.
_format_date_multi_city_cache: tuple[tuple[int, int, int, int, int], str] | None = None


def clear_format_date_multi_city_cache() -> None:
    """Reset the date-line cache (intended for tests)."""
    global _format_date_multi_city_cache
    _format_date_multi_city_cache = None


def _format_local_date_line(local: datetime, city: str) -> str:
    abbr = local.tzname() or ""
    return (
        f"{local.strftime('%A')} {local.strftime('%B')} {_ordinal(local.day)} "
        f"{local.strftime('%Y')} {abbr} {city}"
    )


def _format_date_multi_city_compute(utc: datetime) -> str:
    utc_date = (utc.year, utc.month, utc.day)
    baseline = (
        f"{utc.strftime('%A')} {utc.strftime('%B')} {_ordinal(utc.day)} "
        f"{utc.strftime('%Y')} UTC"
    )
    previous_line: str | None = None
    next_line: str | None = None
    for zone_id, city in _DATE_ANCHOR_ZONES_WEST_TO_EAST:
        local = utc.astimezone(ZoneInfo(zone_id))
        local_date = (local.year, local.month, local.day)
        if local_date < utc_date and previous_line is None:
            previous_line = _format_local_date_line(local, city)
        elif local_date > utc_date and next_line is None:
            next_line = _format_local_date_line(local, city)
    extras: list[str] = []
    if previous_line is not None:
        extras.append(previous_line)
    if next_line is not None:
        extras.append(next_line)
    if not extras:
        return baseline
    return f"{baseline}, {', '.join(extras)}"


def _format_date_multi_city(dt: datetime) -> str:
    """UTC baseline; up to two city lines. Result is cached per 30-minute UTC slot."""
    utc = _instant_as_utc(dt)
    bucket = (utc.year, utc.month, utc.day, utc.hour, utc.minute // 30)
    global _format_date_multi_city_cache
    cached = _format_date_multi_city_cache
    if cached is not None and cached[0] == bucket:
        return cached[1]
    result = _format_date_multi_city_compute(utc)
    _format_date_multi_city_cache = (bucket, result)
    return result


def _format_training_cutoff(cutoff_str: str) -> str:
    """Format YYYY-MM-DD as 'Jan 2024' style."""
    try:
        dt = datetime.strptime(cutoff_str, "%Y-%m-%d")
        return dt.strftime("%b %Y")
    except (ValueError, TypeError):
        return cutoff_str


def _months_since(cutoff_str: str, now: datetime) -> int:
    try:
        cutoff = datetime.strptime(cutoff_str, "%Y-%m-%d")
        return max(0, int((now - cutoff).days / 30))
    except (ValueError, TypeError):
        return 0


class LeoSystem(Prompt):
    def __init__(self) -> None:
        super().__init__()
        self.priority = 0
        self.system_prompt_body = None

    def _get_system_prompt_body(self) -> str:
        """
        Returns the system prompt body. If it hasn't been loaded yet,
        or the environment is local, it will read the system prompt
        body from the disk.
        """
        if self.system_prompt_body is None or server_settings.env == "local":
            self.system_prompt_body = _SYSTEM_PROMPT_BODY_PATH.read_text()
        return self.system_prompt_body

    def augment(self, messages: list[dict], **kwargs: dict[str, Any]) -> list[dict]:
        has_system_prompt = any(
            message.get("role") == "system" or message.get("role") == "developer"
            for message in messages
        )

        if not has_system_prompt:
            now = kwargs.get("now", datetime.now(UTC))
            model_config = kwargs.get("model_config")
            friendly_name = model_config.get("friendly_name") if model_config else None
            if "qwen" in (friendly_name or "").lower():
                qwen_search_directive = QWEN_SEARCH_DIRECTIVE
            else:
                qwen_search_directive = ""

            model_suffix = f" (powered by {friendly_name})" if friendly_name else ""

            # Clients that can render MathML opt in via the `math_ml`
            # capability; everyone else gets the plain text math directive.
            brave_capability = cast(CapabilityOptions, kwargs.get("brave_capability"))
            if has_capability(brave_capability, Capability.math_ml):
                disable_math_ml = ""
            else:
                disable_math_ml = DISABLE_MATH_ML

            if has_capability(brave_capability, Capability.content_agent):
                content_agent_intro = CONTENT_AGENT_INTRO
                content_agent_scope = CONTENT_AGENT_SCOPE
                content_agent_directive = CONTENT_AGENT_DIRECTIVE
            else:
                content_agent_intro = ""
                content_agent_scope = ""
                content_agent_directive = ""

            if has_capability(brave_capability, Capability.deep_research):
                deep_research_directive = DEEP_RESEARCH_DIRECTIVE
            else:
                deep_research_directive = ""

            training_cutoff_raw = (
                model_config.get("training_cutoff") if model_config else None
            )
            if training_cutoff_raw:
                line = (
                    f"Your training data ends in {_format_training_cutoff(training_cutoff_raw)}. "
                    f"For events after that, use search tools to fetch current information.\n"
                )
            else:
                line = ""
            content = (
                self._get_system_prompt_body()
                .replace("{{date}}", _format_date_multi_city(now))
                .replace("{{model_suffix}}", model_suffix)
                .replace("{{training_cutoff_line}}", line)
                .replace("{{qwen_search_directive}}", qwen_search_directive)
                .replace("{{disable_math_ml}}", disable_math_ml)
                .replace("{{content_agent_intro}}", content_agent_intro)
                .replace("{{content_agent_scope}}", content_agent_scope)
                .replace("{{content_agent_directive}}", content_agent_directive)
                .replace("{{deep_research_directive}}", deep_research_directive)
            )

            messages = [
                {"role": "system", "content": content},
            ] + messages

        return messages


leo_system = LeoSystem()
