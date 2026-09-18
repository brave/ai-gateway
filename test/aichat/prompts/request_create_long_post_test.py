from aichat.prompts.request_create_long_post import TYPE, request_create_long_post


def test_request_create_long_post():
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
                    "text": request_create_long_post.TEXT,
                }
            ],
        }
    ]
    actual_messages = request_create_long_post.augment(messages)
    assert actual_messages == expected_messages
