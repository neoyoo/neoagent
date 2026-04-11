from __future__ import annotations
import asyncio
from pathlib import Path
from pydantic import BaseModel
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult
from neoagent.tools.pathguard import validate_path


class GrepInput(BaseModel):
    pattern: str
    path: str = "."
    glob_filter: str | None = None


class GrepTool(BaseTool):
    name: str = "grep"
    description: str = "Search file contents for a regex pattern"
    input_model: type[BaseModel] = GrepInput
    permission: str = "auto"
    is_concurrent_safe: bool = True

    def __init__(self, allowed_directories: list[Path] | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._allowed = allowed_directories or [Path.cwd()]

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, GrepInput)
        try:
            path = validate_path(input.path, self._allowed)
        except ValueError as e:
            return ToolResult(call_id="", output=str(e), is_error=True)
        try:
            cmd = ["grep", "-r", "-n", "-E", input.pattern, str(path)]
            if input.glob_filter:
                cmd.extend(["--include", input.glob_filter])
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            output = stdout.decode("utf-8", errors="replace")
            if proc.returncode == 1:  # no matches
                return ToolResult(call_id="", output="No matches found")
            if proc.returncode >= 2:
                return ToolResult(call_id="", output=stderr.decode("utf-8", errors="replace"), is_error=True)
            return ToolResult(call_id="", output=output)
        except Exception as e:
            return ToolResult(call_id="", output=str(e), is_error=True)
