from __future__ import annotations
import itertools
from pathlib import Path
from pydantic import BaseModel
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult
from neoagent.tools.pathguard import validate_path


class ReadInput(BaseModel):
    file_path: str
    offset: int = 0
    limit: int = 2000


class ReadTool(BaseTool):
    name: str = "read"
    description: str = "Read file contents with line numbers"
    input_model: type[BaseModel] = ReadInput
    permission: str = "auto"
    is_concurrent_safe: bool = True

    def __init__(self, allowed_directories: list[Path] | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._allowed = allowed_directories or [Path.cwd()]

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, ReadInput)
        try:
            path = validate_path(input.file_path, self._allowed)
        except ValueError as e:
            return ToolResult(call_id="", output=str(e), is_error=True)
        try:
            with open(path, "r") as f:
                # Skip offset lines without loading them into memory
                if input.offset:
                    for _ in itertools.islice(f, input.offset):
                        pass
                # Read only limit lines
                selected = list(itertools.islice(f, input.limit))
            numbered = []
            for i, line in enumerate(selected, start=input.offset + 1):
                numbered.append(f"{i}\t{line.rstrip()}")
            return ToolResult(call_id="", output="\n".join(numbered))
        except FileNotFoundError:
            return ToolResult(
                call_id="", output=f"File not found: {input.file_path}", is_error=True
            )
        except Exception as e:
            return ToolResult(call_id="", output=str(e), is_error=True)
