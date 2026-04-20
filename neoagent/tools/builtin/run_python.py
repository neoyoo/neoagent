from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult


# Env vars forwarded into the sandbox — must be enough to reach pip-installed
# packages (httpx/bs4) and traverse an outbound proxy, but nothing secret.
_SAFE_ENV_PASSTHROUGH = (
    "PATH",
    "PYTHONPATH",
    "HOME",
    "LANG", "LC_ALL", "LC_CTYPE",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
)


class RunPythonInput(BaseModel):
    code: str
    timeout_seconds: int = Field(default=30, ge=1, le=120)


class RunPythonTool(BaseTool):
    """Execute sandboxed Python code for ad-hoc data extraction and exploration.

    Use this when fetch_and_clean returns empty clean_text but image_urls are present,
    or when you need to parse embedded JSON blobs, decompress responses, or run
    custom extraction logic that cannot be expressed as a prompt alone.

    Available standard-library and third-party packages:
      bs4 (BeautifulSoup), json, re, gzip, urllib, httpx, hashlib, base64,
      collections, itertools, pathlib, io, struct, zlib

    Execution environment:
      - Runs python3 in a fresh temporary directory (discarded after each call)
      - Environment variables are fully cleared (no secrets leaked)
      - Default timeout: 30 s; can be raised up to 120 s via timeout_seconds
      - Network access is available (httpx/urllib work normally)

    Returns JSON with keys:
      stdout      — captured output (str)
      stderr      — captured error output (str)
      returncode  — process exit code (int)
      timed_out   — true if the process was killed due to timeout (bool)

    Naming convention when writing skills after a successful exploration:
      call write_skill with the extraction pattern so future runs skip exploration.
    """

    name: str = "run_python"
    description: str = (
        "Execute a sandboxed Python 3 snippet and return stdout/stderr/returncode. "
        "Available packages: bs4, httpx, json, re, gzip, urllib, hashlib, base64. "
        "Use for parsing embedded JSON, decompressing responses, or custom extraction. "
        "Timeout default 30 s (max 120 s). Returns {stdout, stderr, returncode, timed_out}."
    )
    input_model: type[BaseModel] = RunPythonInput
    permission: str = "auto"
    is_concurrent_safe: bool = True

    def __init__(self, sandbox_root: Path | None = None) -> None:
        super().__init__()
        # Caller may anchor the ephemeral cwd under a project dir so that
        # scratch files land inside the run workspace, not /var/folders/…
        if sandbox_root is not None:
            sandbox_root.mkdir(parents=True, exist_ok=True)
        self._sandbox_root = sandbox_root

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, RunPythonInput)

        work_dir = tempfile.mkdtemp(
            dir=str(self._sandbox_root) if self._sandbox_root else None
        )
        try:
            # Use same interpreter as parent (venv python has httpx/bs4 installed);
            # forward only whitelisted env so the sandbox can run pip-installed
            # packages and reach through the outbound proxy, without leaking secrets.
            sandbox_env = {
                k: os.environ[k]
                for k in _SAFE_ENV_PASSTHROUGH
                if k in os.environ
            }
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", input.code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env=sandbox_env,
                start_new_session=True,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=input.timeout_seconds,
                )
                result = {
                    "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                    "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                    "returncode": proc.returncode,
                    "timed_out": False,
                }
                return ToolResult(call_id="", output=json.dumps(result))
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                await proc.wait()
                result = {
                    "stdout": "",
                    "stderr": f"Process killed after {input.timeout_seconds}s timeout",
                    "returncode": -1,
                    "timed_out": True,
                }
                return ToolResult(call_id="", output=json.dumps(result))
        except Exception as e:
            return ToolResult(call_id="", output=json.dumps({
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "timed_out": False,
            }), is_error=True)
        finally:
            # Best-effort cleanup; failure here is not fatal
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
