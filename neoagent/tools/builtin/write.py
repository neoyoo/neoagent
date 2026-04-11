from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult
from neoagent.tools.pathguard import validate_path


class WriteInput(BaseModel):
    file_path: str
    content: str


class WriteTool(BaseTool):
    name: str = "write"
    description: str = "Write content to a file, creating parent directories if needed"
    input_model: type[BaseModel] = WriteInput
    permission: str = "ask"
    is_concurrent_safe: bool = False

    def __init__(self, allowed_directories: list[Path] | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._allowed = allowed_directories or [Path.cwd()]

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, WriteInput)
        try:
            path = validate_path(input.file_path, self._allowed)
        except ValueError as e:
            return ToolResult(call_id="", output=str(e), is_error=True)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(input.content)
            return ToolResult(call_id="", output=f"Written to {path}")
        except Exception as e:
            return ToolResult(call_id="", output=str(e), is_error=True)
