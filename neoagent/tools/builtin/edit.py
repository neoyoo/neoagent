from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult


class EditInput(BaseModel):
    file_path: str
    old_string: str
    new_string: str


class EditTool(BaseTool):
    name: str = "edit"
    description: str = "Replace a unique string in a file"
    input_model: type[BaseModel] = EditInput
    permission: str = "ask"
    is_concurrent_safe: bool = False

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, EditInput)
        try:
            path = Path(input.file_path)
            if not path.exists():
                return ToolResult(
                    call_id="", output=f"File not found: {input.file_path}", is_error=True
                )
            content = path.read_text()
            count = content.count(input.old_string)
            if count == 0:
                return ToolResult(
                    call_id="",
                    output=f"String not found in {input.file_path}",
                    is_error=True,
                )
            if count > 1:
                return ToolResult(
                    call_id="",
                    output=f"String not unique in {input.file_path} (found {count} times)",
                    is_error=True,
                )
            new_content = content.replace(input.old_string, input.new_string, 1)
            path.write_text(new_content)
            return ToolResult(call_id="", output=f"Edited {input.file_path}")
        except Exception as e:
            return ToolResult(call_id="", output=str(e), is_error=True)
