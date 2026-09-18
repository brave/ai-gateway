from typing import Any

from pydantic import BaseModel


class OpenAIChatParams(BaseModel):
    """
    OpenAI Chat API params, as defined in the OpenAI API docs.
    """

    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stop: list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    # Note: "stream" is omitted intentionally, because
    # that param is decided depending on if generate
    # or generate_iterator is called.
