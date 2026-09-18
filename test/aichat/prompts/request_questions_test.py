from aichat.prompts.request_questions import TYPE, request_questions


def test_request_questions():
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
                    "text": request_questions.TEXT,
                }
            ],
        }
    ]
    actual_messages = request_questions.augment(messages)
    assert actual_messages == expected_messages
