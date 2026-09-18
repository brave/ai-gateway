from typing import Literal

import shortuuid
from pydantic import BaseModel, Field


class MetadataDict(BaseModel):
    user_id: str | None = None


class CompletionRequest(BaseModel):
    prompt: str
    model: str
    max_tokens_to_sample: int
    stop_sequences: str | list[str] | None = []
    stream: bool | None = False
    temperature: float | None = 1.0
    top_k: int | None = -1
    top_p: float | None = -1.0
    metadata: MetadataDict | None = None


class CompletionResponse(BaseModel):
    completion: str
    stop_reason: Literal["stop_sequence", "max_tokens"] | None = None
    truncated: bool = False  # Not officially supported
    stop: str | None = None
    model: str
    log_id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    exception: str | None = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: int | None = None
    finish_reason: Literal["stop", "length"] | None = None


class CompletionStreamResponse(BaseModel):
    completion: str
    stop_reason: Literal["stop_sequence", "max_tokens"] | None = None
    truncated: bool = False  # Not supported
    stop: str | None = None
    model: str
    log_id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    exception: str | None = None


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: list[Message]
    system: str | None = ""
    stop_sequences: str | list[str] | None = []
    stream: bool | None = False
    temperature: float | None = 1.0
    top_k: int | None = -1
    top_p: float | None = -1.0
    metadata: MetadataDict | None = None


class MessagesUsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class MessagesResponseContent(BaseModel):
    type: Literal["text"]
    text: str


class MessagesResponse(BaseModel):
    id: str
    type: Literal["message"]
    role: Literal["user", "assistant"]
    content: list[MessagesResponseContent]
    stop_reason: Literal["stop_sequence", "max_tokens", "end_turn"]
    stop_sequence: str | None = None
    truncated: bool = False  # Not officially supported
    model: str
    log_id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    exception: str | None = None
    usage: MessagesUsageInfo


class ErrorMessage(BaseModel):
    type: str
    message: str


class ErrorResponse(BaseModel):
    type: str
    error: ErrorMessage


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
