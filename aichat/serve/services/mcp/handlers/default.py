import json
import logging
from typing import Any

from aichat.serve.services.mcp.registry import (
    AugmentedToolConfig,
    MCPServerHandler,
)

logger = logging.getLogger(__name__)


class DefaultMCPServerHandler(MCPServerHandler):
    """Default handler for generic MCP servers."""

    def validate_result(self, result: Any) -> bool:
        """Generic validation - accept any non-None result.

        Args:
            result: The result to validate

        Returns:
            True if result is not None
        """
        return result is not None

    def format_result(self, tool_name: str, result: Any) -> dict:
        """Generic formatting for MCP results.

        Args:
            tool_name: Name of the tool
            result: Raw result to format

        Returns:
            Formatted result dictionary
        """
        texts = self._parse_mcp_content(result)
        content_text = self._format_extracted_texts(texts)

        return {
            "type": "brave-mcp-result",
            "tool_name": tool_name,
            "content": content_text,
            "raw_result": result,
        }

    def _format_extracted_texts(self, texts: list[str]) -> str:
        """Format extracted text content for human readability.

        Args:
            texts: List of text strings extracted from MCP result

        Returns:
            Formatted content as string
        """
        if not texts:
            return ""

        formatted = []
        for text_content in texts:
            try:
                parsed = json.loads(text_content)
                if isinstance(parsed, dict):
                    if "title" in parsed and "url" in parsed:
                        title = parsed.get("title", "")
                        url = parsed.get("url", "")
                        snippet = parsed.get("snippet", "")
                        formatted.append(f"{title}\n{url}\n{snippet}")
                    else:
                        formatted.append(json.dumps(parsed, indent=2))
                else:
                    formatted.append(str(parsed))
            except (json.JSONDecodeError, TypeError):
                formatted.append(text_content)

        return "\n\n".join(formatted) if formatted else ""

    @property
    def server_name(self) -> str:
        """Return the default server name.

        Returns:
            The string "default"
        """
        return "default"

    def get_tool_guidance(self) -> dict[str, str]:
        """No specific guidance for default handler.

        Returns:
            Empty dictionary
        """
        return {}

    def get_augmented_tools(self) -> list[AugmentedToolConfig]:
        """No augmented tools for default handler.

        Returns:
            Empty list
        """
        return []
