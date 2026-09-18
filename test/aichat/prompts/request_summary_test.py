from aichat.prompts.request_summary import TYPE, RequestSummary, request_summary


def test_request_summary():
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
                    "text": request_summary.TEXT,
                }
            ],
        }
    ]
    actual_messages = request_summary.augment(messages)
    assert actual_messages == expected_messages


def test_request_summary_brave_model_uses_alternate_text():
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
                    "text": RequestSummary.BRAVE_SUMMARY_TEXT,
                }
            ],
        }
    ]
    actual_messages = request_summary.augment(
        messages, model_config={"model_id": "brave-summary"}
    )
    assert actual_messages == expected_messages
