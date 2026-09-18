from typing import Any, Literal

from .prompt import ContentPart, Prompt

TYPE = "brave-request-change-tone"


class RequestChangeToneContentPart(ContentPart):
    type: Literal["brave-request-change-tone"]
    tone: Literal["academic", "persuasive", "professional", "funny", "casual"]
    ALIGNMENT_TRACE_TEXT: str = "Change the tone of the attached excerpt"


class RequestChangeTone(Prompt):
    TEXT = "Rewrite the excerpt in a {tone} tone. Include only the rewritten text in your response without code block formatting."
    TONES = {
        "academic": ("academic"),
        "persuasive": ("persuasive"),
        "professional": ("professional"),
        "funny": ("funny"),
        "casual": ("casual"),
    }

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
                        tone = c.get("tone")

                        if tone not in RequestChangeTone.TONES:
                            raise ValueError(f"Invalid tone: {tone}")

                        c = {
                            "type": "text",
                            "text": (RequestChangeTone.TEXT).format(
                                tone=(RequestChangeTone.TONES[tone])
                            ),
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages


request_change_tone = RequestChangeTone()
