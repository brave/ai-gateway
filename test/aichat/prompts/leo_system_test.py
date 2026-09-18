from datetime import UTC, datetime

import pytest

from aichat.prompts.leo_system import (
    _SYSTEM_PROMPT_BODY_PATH,
    CONTENT_AGENT_DIRECTIVE,
    CONTENT_AGENT_INTRO,
    CONTENT_AGENT_SCOPE,
    DEEP_RESEARCH_DIRECTIVE,
    DISABLE_MATH_ML,
    QWEN_SEARCH_DIRECTIVE,
    _format_date_multi_city,
    _format_training_cutoff,
    clear_format_date_multi_city_cache,
    leo_system,
)
from aichat.protocol.open_ai_protocol import Capability

TEST_DATETIME = datetime(2025, 6, 15, 12, 0, 0)


@pytest.fixture(autouse=True)
def reset_format_date_multi_city_cache():
    clear_format_date_multi_city_cache()
    yield


def _expected_content(
    now,
    model_suffix="",
    training_cutoff_raw=None,
    qwen_search_directive="",
    disable_math_ml=DISABLE_MATH_ML,
    content_agent_intro="",
    content_agent_scope="",
    content_agent_directive="",
    deep_research_directive="",
):
    if training_cutoff_raw:
        line = (
            f"Your training data ends in {_format_training_cutoff(training_cutoff_raw)}. "
            f"For events after that, use search tools to fetch current information.\n"
        )
    else:
        line = ""
    body = _SYSTEM_PROMPT_BODY_PATH.read_text()
    return (
        body.replace("{{date}}", _format_date_multi_city(now))
        .replace("{{model_suffix}}", model_suffix)
        .replace("{{training_cutoff_line}}", line)
        .replace("{{qwen_search_directive}}", qwen_search_directive)
        .replace("{{disable_math_ml}}", disable_math_ml)
        .replace("{{content_agent_intro}}", content_agent_intro)
        .replace("{{content_agent_scope}}", content_agent_scope)
        .replace("{{content_agent_directive}}", content_agent_directive)
        .replace("{{deep_research_directive}}", deep_research_directive)
    )


def test_format_date_multi_city_cached_within_same_utc_half_hour():
    assert _format_date_multi_city(
        datetime(2025, 6, 15, 12, 0, 0)
    ) == _format_date_multi_city(datetime(2025, 6, 15, 12, 29, 0))


def test_format_date_multi_city_half_hour_bucket_near_ist_midnight():
    early = _format_date_multi_city(datetime(2025, 6, 15, 18, 0, 0, tzinfo=UTC))
    late = _format_date_multi_city(datetime(2025, 6, 15, 18, 45, 0, tzinfo=UTC))
    assert early != late


def test_format_date_multi_city_utc_only_when_same_calendar_day():
    dt = datetime(2025, 6, 15, 12, 0, 0)
    assert _format_date_multi_city(dt) == "Sunday June 15th 2025 UTC"


def test_format_date_multi_city_appends_westernmost_previous_day_anchor():
    dt = datetime(2025, 6, 16, 0, 0, 0, tzinfo=UTC)
    assert (
        _format_date_multi_city(dt)
        == "Monday June 16th 2025 UTC, Sunday June 15th 2025 PDT Los Angeles"
    )


def test_format_date_multi_city_appends_westernmost_next_day_anchor():
    # Warsaw is the first anchor eastward already on the next local day.
    dt = datetime(2025, 6, 15, 22, 0, 0, tzinfo=UTC)
    assert (
        _format_date_multi_city(dt)
        == "Sunday June 15th 2025 UTC, Monday June 16th 2025 CEST Warsaw"
    )


def test_format_date_multi_city_next_day_skips_eastern_anchors_until_westernmost_flips():
    # Same calendar Sunday in UTC through Warsaw; Kolkata is the first anchor already Monday.
    dt = datetime(2025, 6, 15, 19, 0, 0, tzinfo=UTC)
    assert (
        _format_date_multi_city(dt)
        == "Sunday June 15th 2025 UTC, Monday June 16th 2025 IST Kolkata"
    )


def test_format_date_multi_city_tokyo_first_next_when_kolkata_still_same_day():
    # Kolkata still Sunday; Tokyo is the first anchor on Monday.
    dt = datetime(2025, 6, 15, 18, 0, 0, tzinfo=UTC)
    assert (
        _format_date_multi_city(dt)
        == "Sunday June 15th 2025 UTC, Monday June 16th 2025 JST Tokyo"
    )


def test_existing_system_prompt():
    messages = [{"role": "system"}]
    actual_messages = leo_system.augment(messages)
    assert actual_messages == messages


def test_existing_developer_prompt():
    messages = [{"role": "developer"}]
    actual_messages = leo_system.augment(messages)
    assert actual_messages == messages


def test_leo_system_no_model():
    messages = []
    expected_messages = [
        {"role": "system", "content": _expected_content(TEST_DATETIME)},
    ]
    actual_messages = leo_system.augment(messages, now=TEST_DATETIME)
    assert actual_messages == expected_messages


def test_leo_system_with_model():
    messages = []
    expected_messages = [
        {
            "role": "system",
            "content": _expected_content(
                TEST_DATETIME,
                model_suffix=" (powered by Test Model)",
            ),
        },
    ]
    actual_messages = leo_system.augment(
        messages,
        model_config={"friendly_name": "Test Model"},
        now=TEST_DATETIME,
    )
    assert actual_messages == expected_messages


def test_leo_system_empty_model_config_no_training_line():
    messages = []
    expected_messages = [
        {"role": "system", "content": _expected_content(TEST_DATETIME)},
    ]
    actual_messages = leo_system.augment(messages, model_config={}, now=TEST_DATETIME)
    assert actual_messages == expected_messages


def test_leo_system_with_training_cutoff():
    messages = []
    expected_messages = [
        {
            "role": "system",
            "content": _expected_content(
                TEST_DATETIME,
                model_suffix=" (powered by Test)",
                training_cutoff_raw="2025-01-01",
            ),
        },
    ]
    actual_messages = leo_system.augment(
        messages,
        model_config={
            "friendly_name": "Test",
            "training_cutoff": "2025-01-01",
        },
        now=TEST_DATETIME,
    )
    assert actual_messages == expected_messages


def test_leo_system_qwen_model_includes_search_directive():
    messages = []
    expected_messages = [
        {
            "role": "system",
            "content": _expected_content(
                TEST_DATETIME,
                model_suffix=" (powered by Qwen2.5)",
                qwen_search_directive=QWEN_SEARCH_DIRECTIVE,
            ),
        },
    ]
    actual_messages = leo_system.augment(
        messages,
        model_config={"friendly_name": "Qwen2.5"},
        now=TEST_DATETIME,
    )
    assert actual_messages == expected_messages


def test_leo_system_without_math_ml_capability_includes_directive():
    messages = []
    actual_messages = leo_system.augment(messages, now=TEST_DATETIME)
    content = actual_messages[0]["content"]
    assert DISABLE_MATH_ML in content
    assert "{{disable_math_ml}}" not in content


def test_leo_system_with_math_ml_capability_omits_directive():
    messages = []
    expected_messages = [
        {
            "role": "system",
            "content": _expected_content(TEST_DATETIME, disable_math_ml=""),
        },
    ]
    actual_messages = leo_system.augment(
        messages,
        now=TEST_DATETIME,
        brave_capability=Capability.math_ml,
    )
    assert actual_messages == expected_messages
    assert "MathML" not in actual_messages[0]["content"]


def test_leo_system_math_ml_capability_in_list_omits_directive():
    messages = []
    actual_messages = leo_system.augment(
        messages,
        now=TEST_DATETIME,
        brave_capability=[Capability.chat, Capability.math_ml],
    )
    assert "MathML" not in actual_messages[0]["content"]


def test_leo_system_other_capability_still_includes_directive():
    messages = []
    actual_messages = leo_system.augment(
        messages,
        now=TEST_DATETIME,
        brave_capability=Capability.chat,
    )
    assert DISABLE_MATH_ML in actual_messages[0]["content"]


def test_leo_system_content_agent_capability_includes_agentic_directive():
    messages = []
    expected_messages = [
        {
            "role": "system",
            "content": _expected_content(
                TEST_DATETIME,
                content_agent_intro=CONTENT_AGENT_INTRO,
                content_agent_scope=CONTENT_AGENT_SCOPE,
                content_agent_directive=CONTENT_AGENT_DIRECTIVE,
            ),
        },
    ]
    actual_messages = leo_system.augment(
        messages,
        now=TEST_DATETIME,
        brave_capability=Capability.content_agent,
    )
    assert actual_messages == expected_messages
    assert "Agentic browsing" in actual_messages[0]["content"]


def test_leo_system_without_content_agent_omits_agentic_directive():
    messages = []
    actual_messages = leo_system.augment(messages, now=TEST_DATETIME)
    content = actual_messages[0]["content"]
    assert "Agentic browsing" not in content
    assert "{{content_agent_directive}}" not in content


def test_leo_system_deep_research_capability_includes_directive():
    messages = []
    actual_messages = leo_system.augment(
        messages,
        now=TEST_DATETIME,
        brave_capability=Capability.deep_research,
    )
    assert DEEP_RESEARCH_DIRECTIVE.strip() in actual_messages[0]["content"]


def test_leo_system_without_deep_research_omits_directive():
    messages = []
    actual_messages = leo_system.augment(messages, now=TEST_DATETIME)
    content = actual_messages[0]["content"]
    assert "Deep research" not in content
    assert "{{deep_research_directive}}" not in content
