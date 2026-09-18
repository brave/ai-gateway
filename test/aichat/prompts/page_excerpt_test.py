from aichat.prompts.page_excerpt import TYPE, page_excerpt

TEXT = "Hello, Leo"


def test_page_excerpt():
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
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": page_excerpt.TEXT.format(excerpt=TEXT),
                }
            ],
        }
    ]
    actual_messages = page_excerpt.augment(messages)
    assert actual_messages == expected_messages
