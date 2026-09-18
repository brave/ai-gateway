from typing import Any

from aichat.serve.mcp_tool_execution import _format_web_sources
from aichat.services.security_utils import tag_tool_output

from .prompt import Prompt


class ToolResult(Prompt):
    """
    Prompt augmenter that sanitizes tool result messages to prevent prompt injections.
    """

    def __init__(self) -> None:
        super().__init__()

    def augment(self, messages: list[dict], **kwargs: dict[str, Any]) -> list[dict]:
        augmented_messages: list[dict] = []

        for message in messages:
            if message.get("role") == "tool" and type(message.get("content")) is list:
                content = message.get("content")
                tool_call_id = message.get("tool_call_id")
                function_name = ""
                for msg in messages:
                    if msg.get("tool_calls"):
                        for tc in msg.get("tool_calls"):
                            if tc.get("id") == tool_call_id:
                                function_name = tc.get("function", {}).get("name", "")
                                break
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type")
                        if part_type == "text":
                            text_parts.append(part.get("text") or "")
                        elif part_type == "brave-chat.webSources":
                            # Format websources as text so LLM can read them
                            sources = part.get("sources", [])
                            query = part.get("query")
                            if sources:
                                text_parts.append(_format_web_sources(sources, query))
                            elif part.get("text"):
                                text_parts.append(part.get("text"))

                combined_text = "\n\n".join(text_parts) if text_parts else ""

                message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": [
                        {
                            "type": "text",
                            "text": tag_tool_output(combined_text, function_name),
                        }
                    ],
                }

            augmented_messages.append(message)

        return augmented_messages


tool_result = ToolResult()
