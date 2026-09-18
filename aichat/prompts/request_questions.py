from typing import Any, Literal

from .prompt import ContentPart, Prompt

TYPE = "brave-request-questions"


class RequestQuestionsContentPart(ContentPart):
    type: Literal["brave-request-questions"]
    ALIGNMENT_TRACE_TEXT: str = "Suggest three questions to ask about this page"


class RequestQuestions(Prompt):
    TEXT = "Propose 3 very short questions, around 10 words, that a reader may ask about the this content. Consider intriguing or unusual elements of the content, or structurally important. Please output the questions in the same line separated by a vertical bar |. No numbers."

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
                            "text": (RequestQuestions.TEXT),
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages


request_questions = RequestQuestions()
