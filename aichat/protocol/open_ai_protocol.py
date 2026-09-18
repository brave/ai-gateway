from enum import Enum, IntEnum
from typing import Literal

from pydantic import BaseModel, field_validator
from pydantic_core import PydanticCustomError

from aichat import responses
from aichat.prompts.conversation_title import ConversationTitleContentPart
from aichat.prompts.file_extracted_text import FileExtractedTextContentPart
from aichat.prompts.filter_tabs import FilterTabsContentPart
from aichat.prompts.page_excerpt import PageExcerptContentPart
from aichat.prompts.page_text import PageTextContentPart
from aichat.prompts.pdf_text_content import PDFTextContentPart
from aichat.prompts.prompt import ContentPart
from aichat.prompts.reduce_focus_topics import ReduceFocusTopicsContentPart
from aichat.prompts.request_change_tone import RequestChangeToneContentPart
from aichat.prompts.request_create_long_post import RequestCreateLongPostContentPart
from aichat.prompts.request_create_short_post import RequestCreateShortPostContentPart
from aichat.prompts.request_create_tagline import RequestCreateTaglineContentPart
from aichat.prompts.request_expansion import RequestExpansionContentPart
from aichat.prompts.request_improve_excerpt_language import (
    RequestImproveExcerptLanguageContentPart,
)
from aichat.prompts.request_paraphrase_excerpt import (
    RequestParaphraseExcerptContentPart,
)
from aichat.prompts.request_questions import RequestQuestionsContentPart
from aichat.prompts.request_shorten import RequestShortenContentPart
from aichat.prompts.request_summary import RequestSummaryContentPart
from aichat.prompts.search_results import SearchResultsContentPart
from aichat.prompts.suggest_focus_topics import SuggestFocusTopicsContentPart
from aichat.prompts.suggest_focus_topics_with_emoji import (
    SuggestFocusTopicsWithEmojiContentPart,
)
from aichat.prompts.user_memory import UserMemoryContentPart
from aichat.prompts.video_transcript import VideoTranscriptContentPart

ATTACHMENT_URL_ERROR = "attachment_url"


def _require_data_url(value: str) -> str:
    """Only allow inline attachment data, preventing downstream URL fetches."""
    if not value.startswith("data:"):
        raise PydanticCustomError(
            ATTACHMENT_URL_ERROR,
            "Message attachments must use inline data URLs",
        )
    return value


class ImageUrl(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None

    _validate_url = field_validator("url")(_require_data_url)


class ImageContentPart(ContentPart):
    image_url: ImageUrl
    type: Literal["image_url"]
    ALIGNMENT_TRACE_TEXT: str = (
        "Image uploaded by the user. Actual image content is omitted"
    )


class InputAudio(BaseModel):
    data: str
    format: Literal["wav", "mp3"]


class InputAudioContentPart(ContentPart):
    input_audio: InputAudio
    type: Literal["input_audio"] = "input_audio"
    ALIGNMENT_TRACE_TEXT: str = (
        "Audio uploaded by the user. Actual audio content is omitted"
    )


class VideoUrl(BaseModel):
    url: str

    _validate_url = field_validator("url")(_require_data_url)


class VideoContentPart(ContentPart):
    video_url: VideoUrl
    type: Literal["video_url"] = "video_url"
    ALIGNMENT_TRACE_TEXT: str = (
        "Video uploaded by the user. Actual video content is omitted"
    )


class FileUrl(BaseModel):
    url: str

    _validate_url = field_validator("url")(_require_data_url)


class FileUrlContentPart(ContentPart):
    file_url: FileUrl
    type: Literal["file_url"] = "file_url"
    ALIGNMENT_TRACE_TEXT: str = (
        "File uploaded by the user. Actual file content is omitted"
    )


class File(BaseModel):
    filename: str
    file_data: str

    _validate_file_data = field_validator("file_data")(_require_data_url)


class FileContentPart(ContentPart):
    file: File
    type: Literal["file"] = "file"
    ALIGNMENT_TRACE_TEXT: str = (
        "File uploaded by the user. Actual file content is omitted"
    )


class RefusalContentPart(BaseModel):
    refusal: str
    type: Literal["refusal"]


class TextContentPart(ContentPart):
    type: Literal["text"] = "text"


class WebSourcesContentPart(ContentPart):
    type: Literal["brave-chat.webSources"] = "brave-chat.webSources"
    tool_call_id: str | None = None
    sources: list[responses.WebSource]
    query: list[str] | str | None = None


RequestContentPartUnion = (
    RequestChangeToneContentPart
    | RequestCreateLongPostContentPart
    | RequestCreateShortPostContentPart
    | RequestCreateTaglineContentPart
    | RequestExpansionContentPart
    | RequestImproveExcerptLanguageContentPart
    | RequestParaphraseExcerptContentPart
    | RequestQuestionsContentPart
    | RequestShortenContentPart
    | RequestSummaryContentPart
)

AttachmentContentPartUnion = (
    FileContentPart
    | FileExtractedTextContentPart
    | FileUrlContentPart
    | FilterTabsContentPart
    | ImageContentPart
    | InputAudioContentPart
    | PDFTextContentPart
    | PageExcerptContentPart
    | PageTextContentPart
    | SuggestFocusTopicsContentPart
    | SuggestFocusTopicsWithEmojiContentPart
    | VideoContentPart
    | VideoTranscriptContentPart
)

ContentPartUnion = (
    ConversationTitleContentPart
    | FileContentPart
    | FileExtractedTextContentPart
    | FileUrlContentPart
    | FilterTabsContentPart
    | ImageContentPart
    | InputAudioContentPart
    | PDFTextContentPart
    | PageExcerptContentPart
    | PageTextContentPart
    | ReduceFocusTopicsContentPart
    | RefusalContentPart
    | RequestChangeToneContentPart
    | RequestCreateLongPostContentPart
    | RequestCreateShortPostContentPart
    | RequestCreateTaglineContentPart
    | RequestExpansionContentPart
    | RequestImproveExcerptLanguageContentPart
    | RequestParaphraseExcerptContentPart
    | RequestQuestionsContentPart
    | RequestShortenContentPart
    | RequestSummaryContentPart
    | SearchResultsContentPart
    | SuggestFocusTopicsContentPart
    | SuggestFocusTopicsWithEmojiContentPart
    | TextContentPart
    | VideoContentPart
    | UserMemoryContentPart
    | VideoTranscriptContentPart
    | WebSourcesContentPart
)

# Messages


class Message(BaseModel):
    content: str | list[ContentPartUnion] | None = None
    role: Literal["assistant", "developer", "system", "tool", "user"]


class AssistantAudio(BaseModel):
    id: str


class ToolCallFunction(BaseModel):
    arguments: str
    name: str | None = None


class AlignmentCheck(BaseModel):
    allowed: bool
    reasoning: str | None = None


class ToolCall(BaseModel):
    function: ToolCallFunction
    id: str | None = None
    type: Literal["function"]
    index: int | None = None
    alignment_check: AlignmentCheck | None = None


class AssistantMessage(Message):
    audio: AssistantAudio | None = None
    content: str | list[TextContentPart | RefusalContentPart] | None = None
    name: str | None = None
    refusal: str | None = None
    role: Literal["assistant"] = "assistant"
    tool_calls: list[ToolCall] | None = None


class DeveloperMessage(Message):
    content: str | list[TextContentPart]
    name: str | None = None
    role: Literal["developer"] = "developer"


class SystemMessage(Message):
    content: str | list[TextContentPart]
    name: str | None = None
    role: Literal["system"] = "system"


class ToolMessage(Message):
    content: str | list[TextContentPart | WebSourcesContentPart]
    role: Literal["tool"] = "tool"
    tool_call_id: str | None = None


class UserMessage(Message):
    content: str | list[ContentPartUnion]
    role: Literal["user"] = "user"
    name: str | None = None


MessageUnion = (
    AssistantMessage | DeveloperMessage | SystemMessage | ToolMessage | UserMessage
)

# Request


class RequestAudio(BaseModel):
    format: Literal["wav", "mp3", "flac", "opus", "pcm16"]
    voice: Literal[
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
    ]


class ToolChoiceFunction(BaseModel):
    name: str | None = None


class ToolChoice(BaseModel):
    function: ToolChoiceFunction
    type: Literal["function"]


class ToolFunction(BaseModel):
    description: str | None = None
    name: str | None = None
    parameters: dict | None = None
    strict: bool | None = False


class FunctionCallContentPart(ContentPart):
    type: str = "function_call"
    function: str
    args: dict | None = None


class Tool(BaseModel):
    function: ToolFunction
    type: Literal["function"]


class Capability(str, Enum):
    chat = "chat"
    content_agent = "content_agent"
    deep_research = "deep_research"
    # Client can render MathML (brave-core 1.96+). When absent, the system
    # prompt tells the model to avoid MathML and use plain text math instead.
    math_ml = "math_ml"


CapabilityOptions = list[str] | str | None


def has_capability(
    brave_capability: CapabilityOptions,
    target: str | Capability,
) -> bool:
    if brave_capability is None:
        return False
    if isinstance(brave_capability, list):
        return target in brave_capability
    return brave_capability == target


class BraveCapability(BaseModel):
    brave_capability: Capability


class Request(BaseModel):
    audio: RequestAudio | None = None
    messages: list[MessageUnion]
    model: str
    system_language: str | None = "en"
    parallel_tool_calls: bool | None = True
    tool_choice: Literal["none", "auto", "required"] | ToolChoice | None = None
    tools: list[Tool] | None = None
    stream: bool | None = None
    brave_mcp_tools_include: list[str] | None = None
    brave_mcp_tools_exclude: list[str] | None = None
    brave_capability: CapabilityOptions = None


OPENAI_RESPONSE_OBJECTS = {
    responses.SearchQueries: "brave-chat.searchQueries",
    responses.WebSources: "brave-chat.webSources",
    responses.ContentReceipt: "brave-chat.contentReceipt",
    responses.ToolStart: "brave-chat.toolStart",
    responses.ToolError: "brave-chat.toolError",
    responses.SecurityScan: "brave-chat.securityScan",
    responses.PromptInjectionScanResult: "brave-chat.promptInjectionScanResult",
    responses.InlineSearch: "brave-chat.inlineSearch",
    responses.ToolEnd: "brave-chat.toolEnd",
}


class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes
    """

    VALIDATION_TYPE_ERROR = 40001
    BAD_REQUEST_ERROR = 40002
    API_TIMEOUT_ERROR = 40003
    API_CONNECTION_ERROR = 40004
    INVALID_REQUEST = 40005

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103
    INVALID_SKU_CREDENTIAL = 40104

    INVALID_MODEL = 40301
    CONTEXT_OVERFLOW = 40303

    MODEL_NOT_FOUND = 40401

    EXCEEDED_CONTEXT_LENGTH = 41301

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    BAD_GATEWAY = 50201

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006
