from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult


class WriteInput(BaseModel):
    file_path: str
    content: str


class WriteTool(BaseTool):
    name: str = "write"
    description: str = "Write content to a file, creating parent directories if needed"
    input_model: type[BaseModel] = WriteInput
    permission: str = "ask"
    is_concurrent_safe: bool = False

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, WriteInput)
        try:
            path = Path(input.file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(input.content)
            return ToolResult(call_id="", output=f"Written to {input.file_path}")
        except Exception as e:
            return ToolResult(call_id="", output=str(e), is_error=True)
