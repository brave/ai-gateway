from abc import ABC, abstractmethod
from sys import maxsize
from typing import Any

from pydantic import BaseModel


class Prompt(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._priority = maxsize

    @abstractmethod
    def augment(self, messages: list[dict], **kwargs: dict[str, Any]) -> list[dict]:
        raise NotImplementedError()

    @property
    def priority(self) -> int:
        return self._priority

    @priority.setter
    def priority(self, new_priority: int) -> None:
        self._priority = new_priority


class ContentPart(BaseModel):
    text: str | None = None
