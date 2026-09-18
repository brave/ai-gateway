from aichat.prompts.request_paraphrase_excerpt import TYPE, request_paraphrase_excerpt


def test_request_paraphrase_excerpt():
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
                    "text": request_paraphrase_excerpt.TEXT,
                }
            ],
        }
    ]
    actual_messages = request_paraphrase_excerpt.augment(messages)
    assert actual_messages == expected_messages
