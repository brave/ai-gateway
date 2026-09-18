"""
Protocol-agnostic response models that services can yield.

These response models are independent of any specific API protocol (OpenAI, Leo, etc).
Protocol-specific API layers translate these to their own format.
"""

from pydantic import BaseModel


class SearchQueries(BaseModel):
    queries: list[str]


class WebSource(BaseModel):
    title: str
    url: str
    favicon: str | None = None
    page_content: str | None = None
    extra_snippets: list[str] | None = None


class WebSources(BaseModel):
    sources: list[WebSource]
    rich_results: list[dict] | None = None


class ContentReceipt(BaseModel):
    total_tokens: int
    trimmed_tokens: int


class CompactionMetadata(BaseModel):
    """
    Compaction metadata embedded in the first streaming chunk.

    The browser should:
    1. Store the summary for use in subsequent requests
    2. Tag conversation entries at compacted_message_indices as "summarized"
    3. Optionally display token savings to user
    """

    summary: str  # LLM-generated summary of compacted messages
    compacted_message_indices: list[
        int
    ]  # 0-based indices of messages that were compacted
    tokens_before: int  # Token count before compaction
    tokens_after: int  # Token count after compaction


class ToolStart(BaseModel):
    tool_call_id: str
    tool_name: str
    message: str | None = None


class ToolError(BaseModel):
    tool_call_id: str
    tool_name: str
    error: str


class SecurityScan(BaseModel):
    type: str
    tool_name: str | None = None


class PromptInjectionScanResult(BaseModel):
    probability: int  # 1 (benign) to 5 (clear injection)
    reasoning: str


class InlineSearchResultThumbnail(BaseModel):
    src: str


class InlineSearchResultMeta(BaseModel):
    netloc: str
    path: str
    favicon: str


class InlineSearchResult(BaseModel):
    type: str
    title: str
    url: str
    description: str
    meta_url: InlineSearchResultMeta
    thumbnail: InlineSearchResultThumbnail | None = None
    age: str | None = None


class InlineSearch(BaseModel):
    query: str
    results: list[InlineSearchResult]


class ToolEnd(BaseModel):
    tool_call_id: str
    tool_name: str
