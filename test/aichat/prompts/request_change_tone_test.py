import pytest

from aichat.prompts.request_change_tone import TYPE, request_change_tone

TONE = "casual"


def test_request_change_tone():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                    "tone": TONE,
                },
            ],
        }
    ]
    expected_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": request_change_tone.TEXT.format(tone=TONE)}
            ],
        }
    ]
    actual_messages = request_change_tone.augment(messages)
    assert actual_messages == expected_messages


def test_request_change_tone_invalid():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                    "tone": "invalid tone",
                },
            ],
        }
    ]
    with pytest.raises(ValueError):
        request_change_tone.augment(messages)
