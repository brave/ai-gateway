from typing import Any, Literal

from .prompt import ContentPart, Prompt

TYPE = "brave-request-paraphrase"


class RequestParaphraseExcerptContentPart(ContentPart):
    type: Literal["brave-request-paraphrase"]
    ALIGNMENT_TRACE_TEXT: str = "Paraphrase the attached excerpt"


class RequestParaphraseExcerpt(Prompt):
    TEXT = "Rewrite the excerpt by paraphrasing. Include only the rewritten text in your response without code block formatting."

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
                            "text": (RequestParaphraseExcerpt.TEXT),
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages


request_paraphrase_excerpt = RequestParaphraseExcerpt()
