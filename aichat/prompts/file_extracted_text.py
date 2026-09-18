from typing import Any, Literal

from aichat.services.security_utils import sanitize_untrusted_content

from .prompt import ContentPart, Prompt

TYPE = "brave-file-extracted-text"


class FileExtractedTextContentPart(ContentPart):
    type: Literal["brave-file-extracted-text"]
    content: str | list[dict] = ""
    ALIGNMENT_TRACE_TEXT: str = (
        "Extracted file text provided to the LLM. The actual content is omitted for security reasons"
    )


class FileExtractedText(Prompt):
    TEXT_PREFIX = "The following is extracted text from a file:\n"

    def __init__(self) -> None:
        super().__init__()

    def _extract_text(self, content: str | list[dict]) -> str:
        if isinstance(content, str):
            return sanitize_untrusted_content(content)
        return sanitize_untrusted_content(
            "\n".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        )

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
                            "text": FileExtractedText.TEXT_PREFIX
                            + self._extract_text(c.get("content", "")),
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages


file_extracted_text = FileExtractedText()
