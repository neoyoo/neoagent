from __future__ import annotations
import asyncio
import re
from pydantic import BaseModel
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult

_DEFAULT_BLOCKED_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*f|-[a-zA-Z]*r|--force|--recursive)\b",  # rm -rf, rm -f
    r"\bchmod\s+777\b",
    r"\bmkfs\b",
    r"\bdd\s+.*of=/dev/",
    r">\s*/dev/sd",
    r"\b(curl|wget)\b.*\|\s*(ba)?sh",  # curl | sh
]


class BashInput(BaseModel):
    command: str
    timeout: int = 120

class BashTool(BaseTool):
    name: str = "bash"
    description: str = "Execute a shell command and return output"
    input_model: type[BaseModel] = BashInput
    permission: str = "ask"
    is_concurrent_safe: bool = False

    def __init__(self, blocked_patterns: list[str] | None = None):
        self._blocked_patterns: list[str] = (
            blocked_patterns if blocked_patterns is not None else _DEFAULT_BLOCKED_PATTERNS
        )

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, BashInput)
        for pattern in self._blocked_patterns:
            if re.search(pattern, input.command):
                return ToolResult(
                    call_id="",
                    output=f"Command blocked by safety filter: matches pattern {pattern!r}",
                    is_error=True,
                )
        try:
            proc = await asyncio.create_subprocess_shell(
                input.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=input.timeout)
                output = stdout.decode("utf-8", errors="replace") if stdout else ""
                if proc.returncode != 0:
                    return ToolResult(call_id="", output=output or f"Exit code: {proc.returncode}", is_error=True)
                return ToolResult(call_id="", output=output)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(call_id="", output=f"Timed out after {input.timeout}s", is_error=True)
        except Exception as e:
            return ToolResult(call_id="", output=str(e), is_error=True)
