from aichat.prompts.conversation_title import (
    MAX_CONVERSATION_CHARACTERS_FOR_TITLE,
    TYPE,
    USER_LEAD_IN,
    conversation_title,
)

TEXT = "Hello, Leo"


def _expected_title_user_content(transcript: str) -> str:
    return (
        USER_LEAD_IN
        + "\n\n<user_question>"
        + transcript
        + "</user_question>\n\n<assistant_response></assistant_response>"
    )


def test_conversation_title():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                    "text": TEXT,
                },
            ],
        }
    ]
    expected_messages = [
        {"role": "user", "content": _expected_title_user_content(TEXT)},
    ]
    actual_messages = conversation_title.augment(messages)
    assert actual_messages == expected_messages


def test_conversation_title_no_system_message():
    """Title augment is a single user turn; LeoSystem may still prepend separately."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                    "text": TEXT,
                },
            ],
        }
    ]
    after_title = conversation_title.augment(messages)
    assert len(after_title) == 1
    assert after_title[0]["role"] == "user"
    assert after_title[0]["content"] == _expected_title_user_content(TEXT)
    assert not any(
        message.get("role") in ("system", "developer") for message in after_title
    )


def test_conversation_title_empty_text_returns_unchanged():
    messages = [
        {
            "role": "user",
            "content": [{"type": TYPE, "text": ""}],
        }
    ]
    assert conversation_title.augment(messages) == messages


def test_conversation_title_truncates_long_transcript():
    long_text = "a" * (MAX_CONVERSATION_CHARACTERS_FOR_TITLE + 100)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                    "text": long_text,
                },
            ],
        }
    ]
    expected_transcript = long_text[:MAX_CONVERSATION_CHARACTERS_FOR_TITLE]
    actual_messages = conversation_title.augment(messages)
    assert actual_messages == [
        {"role": "user", "content": _expected_title_user_content(expected_transcript)},
    ]
