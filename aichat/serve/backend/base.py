from collections.abc import AsyncIterator

from aichat.protocol.open_ai_protocol import Tool
from aichat.serve.services.models import ModelConfig


class Backend:
    def __init__(self, model_config: ModelConfig) -> None:
        self.config: ModelConfig = model_config

    async def converse(
        self, messages: list[dict], stream: bool, params: dict
    ) -> AsyncIterator | dict:
        pass

    def build_params(
        self,
        stream: bool,
        tools: list[Tool],
        messages: list[dict],
        context_window_override: int | None = None,
        enable_prompt_caching: bool | None = None,
    ) -> dict:
        pass
