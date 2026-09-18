#!/usr/bin/env python
from enum import Enum, unique
from typing import Any, Literal

import shortuuid
from pydantic import BaseModel, Field


@unique
class Capability(str, Enum):
    chat = "chat"
    content_agent = "content_agent"


class BaseEvent(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[str]


class ConversationRequest(BaseModel):
    model: str
    system_language: str | None = "en"
    stream: bool | None = False
    events: list
    tools: list | None = None
    parallel_tool_calls: bool | None = True
    use_citations: bool | None = False
    capability: Capability | None = None


@unique
class ResponseEventType(str, Enum):
    completion = "completion"
    conversationTitle = "conversationTitle"
    selectedLanguage = "selectedLanguage"
    contentReceipt = "contentReceipt"
    isSearching = "isSearching"
    searchQueries = "searchQueries"
    webSources = "webSources"
    securityScan = "securityScan"


class BaseResponseEvent(BaseModel):
    type: ResponseEventType
    model: str
    log_id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    exception: str | None = None


class ConversationResponseEvent(BaseResponseEvent):
    type: Any
    completion: str | None = None
    stop_reason: str | None = None
    stop_sequence: str | None = None
    tool_calls: list[dict] | None = None
    alignment_check: Any | None = None
