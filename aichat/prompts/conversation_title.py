from typing import Any, Literal

from .prompt import ContentPart, Prompt

TYPE = "brave-conversation-title"


class ConversationTitleContentPart(ContentPart):
    type: Literal["brave-conversation-title"]


USER_LEAD_IN = "Generate a title for this conversation:"
DEFAULT_CONVERSATION_TITLE = "Conversation with Leo"

# Rough estimate for 8192 tokens which is enough to use to generate a title
MAX_CONVERSATION_CHARACTERS_FOR_TITLE = 24576


class ConversationTitle(Prompt):
    def __init__(self) -> None:
        super().__init__()
        self.priority = -1

    def augment(self, messages: list[dict], **kwargs: dict[str, Any]) -> list[dict]:
        title_texts: list[str] = []
        for message in messages:
            content = message.get("content")
            if message.get("role") == "user" and type(content) is list:
                for c in content:
                    if c.get("type") == TYPE:
                        t = c.get("text")
                        if t:
                            title_texts.append(t)

        if not title_texts:
            return messages

        transcript = "\n\n".join(title_texts)
        # Browser only sends a single message either user or assistant to the title model.
        # The model is trained on the prompt structure below and should be able to generate a title
        # regardless of which message is passed as the <user_question>.

        if len(transcript) > MAX_CONVERSATION_CHARACTERS_FOR_TITLE:
            transcript = transcript[:MAX_CONVERSATION_CHARACTERS_FOR_TITLE]

        return [
            {
                "role": "user",
                "content": USER_LEAD_IN
                + "\n\n<user_question>"
                + transcript
                + "</user_question>\n\n<assistant_response></assistant_response>",
            },
        ]


conversation_title = ConversationTitle()
