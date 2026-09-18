from aichat.prompts.video_transcript import TYPE, video_transcript


def test_video_transcript():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": TYPE,
                    "text": "Hello, Leo",
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
                    "text": video_transcript.TEXT.format(transcript="Hello, Leo"),
                }
            ],
        }
    ]
    actual_messages = video_transcript.augment(messages)
    assert actual_messages == expected_messages
