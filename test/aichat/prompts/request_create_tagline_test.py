from aichat.prompts.request_create_tagline import TYPE, request_create_tagline


def test_request_create_tagline():
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
                    "text": request_create_tagline.TEXT,
                }
            ],
        }
    ]
    actual_messages = request_create_tagline.augment(messages)
    assert actual_messages == expected_messages
