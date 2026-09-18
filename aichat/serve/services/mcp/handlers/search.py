import json
import logging
from typing import Any

from aichat.prompts.search_results import SearchResults
from aichat.protocol.open_ai_protocol import (
    TextContentPart,
)
from aichat.serve.services.mcp.handlers.utils import (
    build_web_sources_content_part,
    build_web_sources_output_part,
)
from aichat.serve.services.mcp.registry import (
    AugmentedToolConfig,
    MCPServerHandler,
)

logger = logging.getLogger(__name__)


class SearchServerHandler(MCPServerHandler):
    """Handler for search MCP server results."""

    @property
    def server_name(self) -> str:
        """Return the search server name."""
        return "brave_search"

    def validate_result(self, result: Any) -> bool:
        """Validate search result has expected structure."""
        if not isinstance(result, dict):
            return False
        return "content" in result and isinstance(result["content"], list)

    @staticmethod
    def _extract_rich_weather_data(search_data: dict) -> bool:
        """Fill title, url, page_content for Brave weather rich/infobox. Returns False if payload is unusable."""
        provider = search_data.get("provider")
        if not isinstance(provider, dict):
            return False
        url = provider.get("url")
        if not isinstance(url, str) or not url.strip():
            return False
        if "weather" not in search_data:
            return False
        name = provider.get("name", "")
        if not isinstance(name, str):
            name = str(name)
        search_data["title"] = f"Weather JSON Data from {name}"
        search_data["url"] = url
        search_data["page_content"] = (
            f"Here is some weather data in JSON format: {search_data['weather']}"
        )
        return True

    @staticmethod
    def _extract_rich_currency_data(search_data: dict) -> bool:
        """Fill title, url, page_content for Brave currency rich/infobox. Returns False if payload is unusable."""
        provider = search_data.get("provider")
        if not isinstance(provider, dict):
            return False
        url = provider.get("url")
        if not isinstance(url, str) or not url.strip():
            return False
        currency = search_data.get("currency")
        if not isinstance(currency, dict):
            return False
        conversion = currency.get("conversion")
        if not isinstance(conversion, dict):
            return False
        query = conversion.get("query", "")
        info = conversion.get("info", "")
        timeseries = currency.get("timeseries", "")
        name = provider.get("name", "")
        if not isinstance(name, str):
            name = str(name)
        search_data["title"] = f"Currency Conversion JSON Data from {name}"
        search_data["url"] = url
        search_data["page_content"] = (
            "Here is some Currency Conversion data in JSON format:\n"
            f" Conversion Query: {query}\n"
            f" Conversion Result: {info}\n"
            f" Currency Data Timeseries: {timeseries}"
        )
        return True

    def format_result(self, tool_name: str, result: dict) -> dict:
        """Format search results for consumption by the model.

        Args:
            tool_name: Name of the search tool
            result: Raw MCP result with search data

        Returns:
            Formatted result with search content
        """
        sources = []
        search_results_content = []
        rich_results = []

        logger.debug(f"Raw MCP result for {tool_name}: {result}")
        texts = self._parse_mcp_content(result)
        logger.debug(f"Extracted texts: {texts}")

        for i, text_content in enumerate(texts):
            if not text_content or not text_content.strip():
                continue

            parsed_objects = []
            try:
                parsed_data = json.loads(text_content)
                if isinstance(parsed_data, list):
                    parsed_objects.extend(parsed_data)
                elif isinstance(parsed_data, dict) and "results" in parsed_data:
                    results = parsed_data.get("results", [])
                    if isinstance(results, list):
                        parsed_objects.extend(results)
                    else:
                        parsed_objects.append(parsed_data)
                else:
                    parsed_objects.append(parsed_data)
            except json.JSONDecodeError:
                for line in text_content.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        line_data = json.loads(line)

                        if isinstance(line_data, list):
                            parsed_objects.extend(line_data)
                        elif isinstance(line_data, dict) and "results" in line_data:
                            results = line_data.get("results", [])
                            if isinstance(results, list):
                                parsed_objects.extend(results)
                            else:
                                parsed_objects.append(line_data)
                        else:
                            parsed_objects.append(line_data)
                    except json.JSONDecodeError:
                        continue
            logger.debug(f"Parsed {len(parsed_objects)} objects from text")
            for search_data in parsed_objects:
                try:
                    logger.debug(f"Processing search_data: {search_data}")

                    if isinstance(search_data, dict) and search_data.get("type") in (
                        "rich",
                    ):
                        rich_results.append(search_data)
                        subtype = search_data.get("subtype")
                        if subtype == "weather":
                            if not self._extract_rich_weather_data(search_data):
                                continue
                        elif subtype == "currency":
                            if not self._extract_rich_currency_data(search_data):
                                continue
                        else:
                            continue

                    is_valid = isinstance(search_data, dict) and bool(
                        search_data.get("url")
                    )
                    logger.debug(f"is_valid: {is_valid}, type: {type(search_data)}")
                    if is_valid:
                        title = search_data.get("title", "")
                        url = search_data.get("url", "")

                        snippet = search_data.get("snippet") or search_data.get(
                            "description", ""
                        )

                        favicon = None
                        if "meta_url" in search_data:
                            meta_url = search_data.get("meta_url", {})
                            if isinstance(meta_url, dict) and "favicon" in meta_url:
                                favicon = meta_url.get("favicon")

                        if not favicon and "favicon" in search_data:
                            favicon = search_data.get("favicon")

                        page_content = search_data.get("page_content")
                        extra_snippets = search_data.get("extra_snippets")
                        if extra_snippets is not None and not isinstance(
                            extra_snippets, list
                        ):
                            logger.warning(
                                f"extra_snippets is not a list: "
                                f"{type(extra_snippets)}"
                            )
                            extra_snippets = None

                        source = {
                            "title": title,
                            "url": url,
                            "favicon": favicon,
                            "page_content": page_content,
                            "extra_snippets": extra_snippets,
                        }
                        sources.append(source)

                        content_parts = [
                            f"<search_result citation_number={i+1}>"
                            f"**{title}**\n{url}\n{snippet}"
                        ]

                        if extra_snippets:
                            content_parts.append(
                                "\n\nThese are some snippets from the page:"
                            )
                            for snippet_text in extra_snippets:
                                if isinstance(snippet_text, str):
                                    content_parts.append(f"- {snippet_text}")

                        if page_content:
                            content_parts.append(
                                f"\n\nThis is a summary of the page and some key quotes:\n{page_content}"
                            )

                        content_parts.append("</search_result>")
                        content_item = "".join(content_parts)
                        search_results_content.append(content_item)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Failed to process search result: {e}")
                    continue

        content_summary = f"Found {len(sources)} search results"
        if sources:
            titles = [source["title"] for source in sources[:3]]
            content_summary += f": {', '.join(titles)}"
            if len(sources) > 3:
                content_summary += f" and {len(sources) - 3} more"

        if search_results_content:
            full_content = SearchResults.TEXT
            full_content = full_content.format(
                results=f"{content_summary}\n\n" + "\n\n".join(search_results_content)
            )
            # add citation prompt instructions - the LLM should add inline citations itself
            full_content += f"\n\n{SearchResults.CITATIONS}"
        else:
            full_content = content_summary

        result_dict = {
            "type": "brave-web-sources",
            "sources": sources,
            "tool_name": tool_name,
            "content": full_content,
            "search_results": (
                "\n".join(search_results_content)
                if search_results_content
                else content_summary
            ),
        }
        if isinstance(result, dict):
            queries = result.get("queries")
            if queries is None:
                structured = result.get("structuredContent") or {}
                queries = (
                    structured.get("queries") if isinstance(structured, dict) else {}
                )
            if queries is not None and isinstance(queries, list):
                result_dict["queries"] = queries

        if rich_results:
            result_dict["rich_results"] = rich_results
            logger.debug(f"Included {len(rich_results)} rich results in output")

        return result_dict

    def get_tool_start_message(self, tool_name: str, tool_args: dict) -> str:
        """Get the start message for search tools.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments

        Returns:
            Start message string
        """
        if tool_name == "brave_web_search":
            query = tool_args.get("query", "")
            if query:
                return f"Searching for: {query}"
            return "Searching the web..."
        elif tool_name == "brave_news_search":
            query = tool_args.get("query", "")
            if query:
                return f"Searching news for: {query}"
            return "Searching news..."
        return f"Running {tool_name}..."

    def get_tool_guidance(self) -> dict[str, str]:
        """Return tool guidance for search tools."""
        return {
            "brave_web_search": """Use `brave_web_search` when you need:
- Up-to-date information from the web
- Current events or recent information

Examples of questions that REQUIRE using this tool:
- What is the population of Tokyo in 2025? = search for `Tokyo population 2025`
- Who is the current CEO of Apple Inc.? = search for `Apple Inc. CEO`
- Can you describe quantum computing to me? = search for `quantum computing`
- What is the capital of Norway? = search for `Norway capital`
- Who is the current president of the United States? = search for `United States president`
- What are the best hiking trails in New Zealand? = search for `New Zealand hiking trails`
- How does nuclear fusion work? = search for `nuclear fusion explanation`
- What are the most popular tourist attractions in Paris? = search for `Paris tourist attractions`

Use this often to get up-to-date information and context.""",
            "brave_news_search": """Use `brave_news_search` for:
- Latest news articles and breaking news
- Current events and trending topics
- News-specific queries requiring recent news sources
- Journalistic coverage of events

Example prompts that would require this tool:
- 'What's the latest news about AI regulation?' = search for `AI regulation`
- 'Show me recent news about climate change' = search for `climate change`
- 'What are the top tech news stories today?' = search for `technology`
- 'Any breaking news about the economy?' = search for `economy`

This tool focuses specifically on news sources and
journalism.""",
        }

    def get_augmented_tools(self) -> list[AugmentedToolConfig]:
        """No augmented tools for search handler."""
        # TODO: Add answer_brave_related_questions tool
        return []

    def get_tool_message_content(
        self, formatted_result: dict, tool_call: Any
    ) -> str | list:
        """Get rich content for ToolMessage with web sources."""
        sources = formatted_result.get("sources", [])
        rich_results = formatted_result.get("rich_results", [])
        content = formatted_result.get("content", "")

        if sources or rich_results:
            content_parts = []
            if sources:
                content_parts.append(
                    build_web_sources_content_part(
                        sources, tool_call, queries=formatted_result.get("queries")
                    )
                )
                logger.info(f"Built rich tool message with {len(sources)} sources")
            else:
                text_summary = (
                    content.strip() if content.strip() else "Found search results"
                )
                content_parts.append(TextContentPart(type="text", text=text_summary))
            return content_parts

        if content.strip():
            return content.strip()
        else:
            return "Tool execution completed"

    def get_output_content_parts(
        self, formatted_result: dict, tool_call: Any
    ) -> list[dict]:
        """Get output content parts for streaming (web sources, rich results)."""
        output_content_parts = []
        sources = formatted_result.get("sources", [])
        rich_results = formatted_result.get("rich_results", [])

        if sources or rich_results:
            output_content_parts.append(
                build_web_sources_output_part(
                    sources,
                    tool_call,
                    rich_results if rich_results else None,
                    queries=formatted_result.get("queries"),
                )
            )

        return output_content_parts

    def format_web_sources_content(
        self, sources: list[dict], query: str | None = None
    ) -> str:
        """Format web sources from client-provided data into text for LLM.

        This is used when clients send back brave-chat.webSources content
        in tool messages.

        Args:
            sources: List of source dicts with title, url, page_content, extra_snippets
            query: Optional search query

        Returns:
            Formatted text content with search results
        """
        if not sources:
            return "Found 0 search results"

        content_summary = f"Found {len(sources)} search results"
        titles = [source.get("title", "") for source in sources[:3]]
        content_summary += f": {', '.join(titles)}"
        if len(sources) > 3:
            content_summary += f" and {len(sources) - 3} more"

        search_results_content = []
        for i, source in enumerate(sources):
            title = source.get("title", "")
            url = source.get("url", "")
            snippet = source.get("snippet") or source.get("description", "")
            page_content = source.get("page_content")
            extra_snippets = source.get("extra_snippets")

            content_parts = [
                f"<search_result citation_number={i+1}>" f"**{title}**\n{url}"
            ]

            if snippet:
                content_parts.append(f"\n{snippet}")

            if extra_snippets and isinstance(extra_snippets, list):
                content_parts.append("\n\nAdditional snippets:")
                for snippet_text in extra_snippets:
                    if isinstance(snippet_text, str):
                        content_parts.append(f"- {snippet_text}")

            if page_content:
                content_parts.append(f"\n\nFull page content:\n{page_content}")

            content_parts.append("</search_result>")
            content_item = "".join(content_parts)
            search_results_content.append(content_item)

        if search_results_content:
            full_content = f"{content_summary}\n\n" + "\n\n".join(
                search_results_content
            )
            full_content += f"\n\n{SearchResults.CITATIONS}"
        else:
            full_content = content_summary

        return full_content
