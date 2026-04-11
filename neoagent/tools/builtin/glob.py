from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult
from neoagent.tools.pathguard import validate_path


class GlobInput(BaseModel):
    pattern: str
    path: str = "."


class GlobTool(BaseTool):
    name: str = "glob"
    description: str = "Find files matching a glob pattern"
    input_model: type[BaseModel] = GlobInput
    permission: str = "auto"
    is_concurrent_safe: bool = True

    def __init__(self, allowed_directories: list[Path] | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._allowed = allowed_directories or [Path.cwd()]

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, GlobInput)
        try:
            validate_path(input.path, self._allowed)
        except ValueError as e:
            return ToolResult(call_id="", output=str(e), is_error=True)
        try:
            base = Path(input.path)
            if not base.exists():
                return ToolResult(call_id="", output=f"Path not found: {input.path}", is_error=True)
            matches = sorted(str(p) for p in base.glob(input.pattern) if p.is_file())
            return ToolResult(call_id="", output="\n".join(matches) if matches else "No matches found")
        except Exception as e:
            return ToolResult(call_id="", output=str(e), is_error=True)
