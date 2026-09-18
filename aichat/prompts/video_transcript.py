from typing import Any, Literal

from aichat.services.security_utils import sanitize_untrusted_content

from .prompt import ContentPart, Prompt

TYPE = "brave-video-transcript"


class VideoTranscriptContentPart(ContentPart):
    type: Literal["brave-video-transcript"]
    ALIGNMENT_TRACE_TEXT: str = (
        "Video transcript provided to the LLM. The actual content is omitted for security reasons"
    )


class VideoTranscript(Prompt):
    TEXT = "This is a video transcript: <transcript>{transcript}</transcript>."

    def __init__(self) -> None:
        super().__init__()

    def augment(self, messages: list[dict], **kwargs: dict[str, Any]) -> list[dict]:
        augmented_messages: list[dict] = []

        for message in messages:
            content = message.get("content")

            if message.get("role") == "user" and type(content) is list:
                augmented_contents: list[dict] = []

                for c in content:
                    if c.get("type") == TYPE:
                        c = {
                            "type": "text",
                            "text": (VideoTranscript.TEXT).format(
                                transcript=sanitize_untrusted_content(c.get("text"))
                            ),
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages


video_transcript = VideoTranscript()
