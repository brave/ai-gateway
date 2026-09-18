from typing import Any, Literal

from aichat.services.security_utils import sanitize_untrusted_content

from .prompt import ContentPart, Prompt

TYPE = "brave-page-text"


class PageTextContentPart(ContentPart):
    type: Literal["brave-page-text"]
    ALIGNMENT_TRACE_TEXT: str = (
        "Webpage content provided to the LLM. The actual content is omitted for security reasons"
    )


class PageText(Prompt):
    TEXT = "This is the text of a web page: <page>{page}</page>."

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
                            "text": (PageText.TEXT).format(
                                page=sanitize_untrusted_content(c.get("text"))
                            ),
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages


page_text = PageText()
