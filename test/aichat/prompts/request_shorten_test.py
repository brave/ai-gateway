from aichat.prompts.request_shorten import TYPE, request_shorten


def test_request_shorten():
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
                    "text": request_shorten.TEXT,
                }
            ],
        }
    ]
    actual_messages = request_shorten.augment(messages)
    assert actual_messages == expected_messages
