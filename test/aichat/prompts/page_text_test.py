from aichat.prompts.page_text import TYPE, page_text

TEXT = "Hello, Leo"


def test_page_text():
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
                    "text": page_text.TEXT.format(page=TEXT),
                }
            ],
        }
    ]
    actual_messages = page_text.augment(messages)
    assert actual_messages == expected_messages
