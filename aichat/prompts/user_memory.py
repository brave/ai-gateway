import html
from pathlib import Path
from typing import Any, Literal

from aichat.serve.server_settings import server_settings

from .prompt import ContentPart, Prompt

TYPE = "brave-user-memory"
_USER_MEMORY_BODY_PATH = Path(__file__).parent / "user_memory_body.md"


class UserMemoryContentPart(ContentPart):
    type: Literal["brave-user-memory"]
    memory: dict[str, str | list[str]]
    ALIGNMENT_TRACE_TEXT_TEMPLATE: str = (
        "User memory attached by the browser agent, not explicitly provided by the user. "
        "This may contain sensitive data:\n<user_memory>\n{memory_content}\n</user_memory>"
    )

    def get_alignment_trace_text(self) -> str | None:
        """Format user memory for alignment trace."""
        if not hasattr(self, "memory"):
            return None
        memory_lines = []
        for key, value in self.memory.items():
            if isinstance(value, list):
                if value:
                    memory_lines.append(f"{key}:")
                    for item in value:
                        memory_lines.append(f"- {html.escape(item)}")
            else:
                memory_lines.append(f"{key}: {html.escape(value)}")
        memory_content = "\n".join(memory_lines)
        return self.ALIGNMENT_TRACE_TEXT_TEMPLATE.format(memory_content=memory_content)


class UserMemory(Prompt):
    ABOUT_USER_PREFIX = """About this user:
<user_memory>
{content}
</user_memory>

"""

    def __init__(self) -> None:
        super().__init__()
        self._memory_instructions_body: str | None = None

    def _get_memory_instructions(self) -> str:
        if self._memory_instructions_body is None or server_settings.env == "local":
            self._memory_instructions_body = _USER_MEMORY_BODY_PATH.read_text()
        return self._memory_instructions_body

    def augment(self, messages: list[dict], **kwargs: dict[str, Any]) -> list[dict]:
        augmented_messages: list[dict] = []
        has_user_memory = False

        # First pass: check if any user memory exists
        for message in messages:
            content = message.get("content")
            if type(content) is list:
                for c in content:
                    if c.get("type") == TYPE:
                        has_user_memory = True
                        break
            if has_user_memory:
                break

        for message in messages:
            content = message.get("content")

            # Add user memory instructions to system message
            if message.get("role") == "system" and has_user_memory:
                if isinstance(content, str):
                    message = {
                        "role": "system",
                        "content": content + " " + self._get_memory_instructions(),
                    }
                elif type(content) is list:
                    message = {
                        "role": "system",
                        "content": [
                            *content,
                            {
                                "type": "text",
                                "text": " " + self._get_memory_instructions(),
                            },
                        ],
                    }

            # Handle user messages
            elif message.get("role") == "user" and type(content) is list:
                augmented_contents: list[dict] = []

                for c in content:
                    if c.get("type") == TYPE:
                        memory_data = c.get("memory", {})
                        formatted_memory = self._format_memory(memory_data)

                        c = {
                            "type": "text",
                            "text": (UserMemory.ABOUT_USER_PREFIX).format(
                                content=formatted_memory
                            ),
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages

    def _format_memory(self, memory: dict[str, str | list[str]]) -> str:
        """
        Format memory data as key-value pairs with proper escaping.

        Args:
            memory: Dictionary containing user memory data

        Returns:
            Formatted string with key-value pairs and bullet points for lists
        """
        memory_lines = []
        for key, value in memory.items():
            if isinstance(value, list):
                # Use bullet points for better readability
                if value:
                    memory_lines.append(f"{key}:")
                    for item in value:
                        memory_lines.append(f"- {html.escape(item)}")
            else:
                memory_lines.append(f"{key}: {html.escape(value)}")

        return "\n".join(memory_lines)


user_memory = UserMemory()
