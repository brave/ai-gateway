from aichat.prompts.request_improve_excerpt_language import (
    TYPE,
    request_improve_excerpt_language,
)


def test_request_improve_excerpt_language():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                },
            ],
        }
    ]
    expected_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": request_improve_excerpt_language.TEXT,
                }
            ],
        }
    ]
    actual_messages = request_improve_excerpt_language.augment(messages)
    assert actual_messages == expected_messages
