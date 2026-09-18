from typing import Any, Literal

from .prompt import ContentPart, Prompt

TYPE = "brave-request-create-short-post"


class RequestCreateShortPostContentPart(ContentPart):
    type: Literal["brave-request-create-short-post"]
    ALIGNMENT_TRACE_TEXT: str = (
        "Use the attached excerpt to generate a post that could be used on social platforms like Twitter."
    )


class RequestCreateShortPost(Prompt):
    TEXT = "Use the excerpt to generate a post that could be used on social platforms like Twitter."

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
                            "text": (RequestCreateShortPost.TEXT),
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages


request_create_short_post = RequestCreateShortPost()
