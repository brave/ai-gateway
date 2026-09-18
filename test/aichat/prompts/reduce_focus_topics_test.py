from aichat.prompts.reduce_focus_topics import TYPE, reduce_focus_topics

TEXT = "topic1; topic2; topic3"


def test_reduce_focus_topics():
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
                {"type": "text", "text": reduce_focus_topics.TEXT.format(topics=TEXT)}
            ],
        }
    ]
    actual_messages = reduce_focus_topics.augment(messages)
    assert actual_messages == expected_messages
