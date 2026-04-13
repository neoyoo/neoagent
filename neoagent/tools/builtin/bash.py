from __future__ import annotations
import asyncio
import os
import re
import signal
from pathlib import Path
from pydantic import BaseModel, Field
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult

_DEFAULT_BLOCKED_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*f|-[a-zA-Z]*r|--force|--recursive)\b",  # rm -rf, rm -f
    r"\bchmod\s+777\b",
    r"\bmkfs\b",
    r"\bdd\s+.*of=/dev/",
    r">\s*/dev/sd",
    r"\b(curl|wget)\b.*\|\s*(ba)?sh",  # curl | sh
    r"\bpython[23]?(\.\d+)?\s+-(c|m)\b",  # python -c / python3.12 -c / python -m
    r"\bperl\s+-e\b",                       # perl -e
    r"\bruby\s+-e\b",                       # ruby -e
    r"\bnode\s+-e\b",                       # node -e
    r"\b(ba|da|z|k|tc|c|fi)?sh\s+-c\b",    # sh/bash/dash/zsh/ksh/tcsh/csh/fish -c
    r"\beval\s+",                            # eval anything
    r"\|\s*(zsh|ksh|fish|tcsh|csh|dash)\b",  # pipe into alternative shells
]

# Environment variable whitelist: only these keys are inherited from the parent process
_DEFAULT_ENV_WHITELIST = {
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM",
    "TMPDIR", "TZ", "SHELL",
}


class BashInput(BaseModel):
    command: str
    timeout: int = Field(default=30, ge=1, le=300)  # default 30s, max 5min


class BashTool(BaseTool):
    name: str = "bash"
    description: str = "Execute a shell command and return output. Working directory is fixed."
    input_model: type[BaseModel] = BashInput
    permission: str = "ask"
    is_concurrent_safe: bool = False

    def __init__(
        self,
        blocked_patterns: list[str] | None = None,
        cwd: Path | None = None,
        allowed_env: list[str] | None = None,
    ):
        self._blocked_patterns: list[str] = (
            blocked_patterns if blocked_patterns is not None else _DEFAULT_BLOCKED_PATTERNS
        )
        self._cwd = cwd
        # allowed_env=None → use default whitelist; allowed_env=[] → empty env (strictest)
        self._allowed_env: set[str] | None = (
            set(allowed_env) if allowed_env is not None else None
        )

    def _safe_env(self) -> dict[str, str]:
        """Build a whitelist-filtered environment variable dict."""
        whitelist = self._allowed_env if self._allowed_env is not None else _DEFAULT_ENV_WHITELIST
        return {k: v for k, v in os.environ.items() if k in whitelist}

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, BashInput)

        # First line of defense: regex blocklist
        for pattern in self._blocked_patterns:
            if re.search(pattern, input.command):
                return ToolResult(
                    call_id="",
                    output=f"Command blocked by safety filter: matches pattern {pattern!r}",
                    is_error=True,
                )

        safe_env = self._safe_env()
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "--restricted", "-c", input.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._cwd) if self._cwd else None,
                env=safe_env,
                start_new_session=True,  # create process group so we can kill the whole tree
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=input.timeout)
                output = stdout.decode("utf-8", errors="replace") if stdout else ""
                if proc.returncode != 0:
                    return ToolResult(call_id="", output=output or f"Exit code: {proc.returncode}", is_error=True)
                return ToolResult(call_id="", output=output)
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)  # Kill entire process group
                except (ProcessLookupError, PermissionError):
                    proc.kill()  # Fallback to killing just the top-level process
                await proc.wait()
                return ToolResult(call_id="", output=f"Timed out after {input.timeout}s", is_error=True)
        except Exception as e:
            return ToolResult(call_id="", output=str(e), is_error=True)
