from typing import Any, Literal

from .prompt import ContentPart, Prompt

TYPE = "brave-reduce-focus-topics"


class ReduceFocusTopicsContentPart(ContentPart):
    type: Literal["brave-reduce-focus-topics"]


class ReduceFocusTopics(Prompt):
    TEXT = """
        Here is a list of browser topics: {topics}.
        Please return an array, delimited by "[" and "]" containing the 5 most unusual or interesting topics.
        Prepend each topic string with a single relevant emoji.
        Try to choose unrelated topics.
        Make sure to return only an array of strings enclosed in quotes.
    """

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
                            "text": (ReduceFocusTopics.TEXT).format(
                                topics=c.get("text")
                            ),
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages


reduce_focus_topics = ReduceFocusTopics()
