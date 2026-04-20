from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel
from neoagent.core.types import ToolResult


class BaseTool(ABC):
    name: str
    description: str
    input_model: type[BaseModel]
    permission: Literal["auto", "ask", "deny"] = "ask"
    is_concurrent_safe: bool = False

    @abstractmethod
    async def execute(self, input: BaseModel) -> ToolResult: ...

    def preview(self, output: str) -> str:
        """80-char preview for freed placeholders. Override in multimedia tools."""
        snippet = output.replace("\n", " ").strip()
        return snippet[:80] + "…" if len(snippet) > 80 else snippet

    def get_schema(self) -> dict:
        raw_schema = self.input_model.model_json_schema()
        raw_schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": raw_schema,
        }
