from typing import Any, Literal

from .prompt import ContentPart, Prompt

TYPE = "brave-request-create-long-post"


class RequestCreateLongPostContentPart(ContentPart):
    type: Literal["brave-request-create-long-post"]
    ALIGNMENT_TRACE_TEXT: str = (
        "Use the attached excerpt to generate a post that could be used on social platforms like Reddit."
    )


class RequestCreateLongPost(Prompt):
    TEXT = "Use the excerpt to generate a post that could be used on social platforms like Reddit."

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
                            "text": (RequestCreateLongPost.TEXT),
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages


request_create_long_post = RequestCreateLongPost()
