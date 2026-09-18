from typing import Any, Literal

from aichat.services.security_utils import sanitize_untrusted_content

from .prompt import ContentPart, Prompt

TYPE = "brave-pdf-text"


class PDFTextContentPart(ContentPart):
    type: Literal["brave-pdf-text"]
    ALIGNMENT_TRACE_TEXT: str = (
        "Extracted PDF page text provided to the LLM. The actual content is omitted for security reasons"
    )


class PdfTextContent(Prompt):
    TEXT_PREFIX = (
        "The following is extracted text from PDF pages that exceeded the upload page limit. "
        "Page numbers are relative to the original document:\n"
    )

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
                            "text": PdfTextContent.TEXT_PREFIX
                            + sanitize_untrusted_content(c.get("text")),
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages


pdf_text_content = PdfTextContent()
