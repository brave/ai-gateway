from aichat.prompts.search_results import TYPE, search_results

TEXT = "search results"


def test_search_results_without_citations():
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
                    "text": search_results.TEXT.format(results=TEXT),
                }
            ],
        }
    ]
    actual_messages = search_results.augment(messages)
    assert actual_messages == expected_messages


def test_search_results_with_citations():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                    "text": TEXT,
                    "use_citations": True,
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
                    "text": search_results.TEXT.format(results=TEXT)
                    + search_results.CITATIONS,
                }
            ],
        }
    ]
    actual_messages = search_results.augment(messages)
    assert actual_messages == expected_messages
