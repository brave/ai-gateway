import logging
from typing import Any

from aichat.protocol.open_ai_protocol import TextContentPart
from aichat.serve.services.mcp.handlers.utils import (
    build_web_sources_content_part,
    build_web_sources_output_part,
)
from aichat.serve.services.mcp.registry import (
    AugmentedToolConfig,
    MCPServerHandler,
)

logger = logging.getLogger(__name__)


class DeepResearchServerHandler(MCPServerHandler):
    """Handler for deep research MCP server results."""

    @property
    def server_name(self) -> str:
        """Return the deep research server name."""
        return "deep_research"

    def validate_result(self, result: Any) -> bool:
        """Validate deep research result has expected structure.

        Deep research results come as a final answer event with:
        - event: "answer"
        - final: true
        - answer: string
        - citations: array (optional)
        """
        if not isinstance(result, dict):
            return False
        # Accept final answer events
        if result.get("event") == "answer" and result.get("final"):
            return "answer" in result
        # Also accept aggregated results with content
        return "content" in result or "answer" in result

    def format_result(self, tool_name: str, result: Any) -> dict:
        """Format deep research results for consumption by the model.

        Args:
            tool_name: Name of the deep research tool
            result: Raw result with research data

        Returns:
            Formatted result with research content and citations
        """
        logger.debug(f"Formatting deep research result for {tool_name}: {result}")

        # Extract answer text
        answer = result.get("answer") or result.get("content", "")

        # Extract citations if present
        citations = result.get("citations", [])

        # Format citations for display
        sources = []
        citations_text = ""
        if citations:
            citations_text = "\n\nSources:\n"
            for i, citation in enumerate(citations):
                url = citation.get("url", "")
                snippet = citation.get("snippet", "")
                number = citation.get("number", i + 1)

                sources.append(
                    {
                        "title": citation.get("title")
                        or (snippet[:100] if snippet else None)
                        or f"Source {number}",
                        "url": url,
                        "favicon": citation.get("favicon"),
                    }
                )

                citations_text += f"[{number}] {url}\n"
                if snippet:
                    citations_text += (
                        f"    {snippet[:200]}...\n"
                        if len(snippet) > 200
                        else f"    {snippet}\n"
                    )

        # Build full content for model
        full_content = f"{answer}{citations_text}"

        return {
            "type": "brave-deep-research-result",
            "sources": sources,
            "tool_name": tool_name,
            "content": full_content,
            "answer": answer,
            "citations": citations,
        }

    def get_tool_guidance(self) -> dict[str, str]:
        """Return tool guidance for deep research tool."""
        return {
            "deep_research": """Use `deep_research` only when the user explicitly requests research-style output using phrases like: 'deep research', 'deep dive', 'research report', 'comprehensive analysis', 'investigate', 'thorough comparison', 'literature review', 'write a report on', or 'analyze in depth'. This tool is SLOW and EXPENSIVE (takes minutes), so only invoke it when clearly warranted.

This tool:
- Performs many iterative web searches to gather information
- Analyzes multiple sources for accuracy and relevance
- Synthesizes findings into a comprehensive cited report

Examples of prompts that warrant deep research:
- "Do a deep dive on the latest developments in quantum computing and their potential applications"
- "Write a research report comparing economic policies on renewable energy across countries"
- "Give me a comprehensive analysis of the pros and cons of [technology/policy/approach]"

Do NOT use for:
- 'Tell me about X', 'Who is X', 'What is X', 'Explain X' style prompts (answer directly or use `brave_web_search`)
- Single-fact lookups, definitions, recent news, scores, prices, or summaries (use `brave_web_search`)
- Conversational or general-knowledge questions the assistant can answer from its own knowledge
- Any question a single web search could resolve

When the user's intent is ambiguous, prefer `brave_web_search` or a direct answer. Do not upgrade a casual question to deep research on the user's behalf.""",
        }

    def get_tool_message_content(
        self, formatted_result: dict, tool_call: Any
    ) -> str | list:
        """Get rich content for ToolMessage with web sources."""
        sources = formatted_result.get("sources", [])
        content = formatted_result.get("content", "")

        if sources:
            content_parts = []

            text_summary = (
                content.strip() if content.strip() else "Deep research completed"
            )
            content_parts.append(TextContentPart(type="text", text=text_summary))

            web_sources_part = build_web_sources_content_part(sources, tool_call)
            content_parts.append(web_sources_part)

            return content_parts

        if content.strip():
            return content.strip()
        else:
            return "Deep research completed"

    def get_output_content_parts(
        self, formatted_result: dict, tool_call: Any
    ) -> list[dict]:
        """Get output content parts for streaming (web sources from citations)."""
        output_content_parts = []
        sources = formatted_result.get("sources", [])

        if sources:
            output_content_parts.append(
                build_web_sources_output_part(sources, tool_call)
            )

        return output_content_parts

    def get_augmented_tools(self) -> list[AugmentedToolConfig]:
        """No augmented tools for deep research handler."""
        return []
