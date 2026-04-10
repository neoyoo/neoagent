from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from neoagent.core.types import ContentBlock, TextBlock, ToolUseBlock

@dataclass
class Response:
    content: list[ContentBlock]
    stop_reason: str
    input_tokens: int
    output_tokens: int

    @property
    def tool_use_blocks(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def text_content(self) -> str:
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))

class Provider(ABC):
    @abstractmethod
    async def create(self, system: str, messages: list, tools: list) -> Response: ...

    @abstractmethod
    def get_context_window(self) -> int: ...
