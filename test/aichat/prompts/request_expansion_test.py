from aichat.prompts.request_expansion import TYPE, request_expansion


def test_request_expansion():
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
                    "text": request_expansion.TEXT,
                }
            ],
        }
    ]
    actual_messages = request_expansion.augment(messages)
    assert actual_messages == expected_messages
