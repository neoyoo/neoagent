from __future__ import annotations
import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Environment variable whitelist: mirrors BashTool._safe_env pattern.
# Only these keys are inherited from the parent process by MCP subprocesses.
_DEFAULT_ENV_WHITELIST = {
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM",
    "TMPDIR", "TZ", "SHELL",
}


class MCPTransport(ABC):
    """Abstract transport layer for MCP communication.

    Implementations: StdioTransport (subprocess stdio).
    Reserved for future: SseTransport (HTTP+SSE), not implemented.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish the transport connection."""

    @abstractmethod
    async def send(self, message: dict) -> None:
        """Send a JSON-RPC message."""

    @abstractmethod
    async def receive(self) -> dict:
        """Receive and parse the next JSON-RPC message.

        Raises:
            ConnectionError: if the stream is closed (EOF).
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the transport and release resources."""


class StdioTransport(MCPTransport):
    """MCP transport over subprocess stdin/stdout (JSON lines protocol).

    Mirrors BashTool's _safe_env whitelist to prevent API key leakage into
    child processes. Extra env keys passed via env= are always forwarded.
    """

    def __init__(self, command: list[str], env: dict[str, str] | None = None) -> None:
        self._command = command
        self._extra_env: dict[str, str] = env or {}
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None

    def _safe_env(self) -> dict[str, str]:
        """Build whitelist-filtered env dict, merged with caller-supplied extras."""
        safe = {k: v for k, v in os.environ.items() if k in _DEFAULT_ENV_WHITELIST}
        safe.update(self._extra_env)
        return safe

    async def connect(self) -> None:
        """Spawn the MCP server subprocess."""
        env = self._safe_env()
        self._proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        logger.debug("StdioTransport: connected to %s (pid=%s)", self._command, self._proc.pid)
        # Continuously drain stderr to prevent pipe-full deadlock when the
        # MCP server writes diagnostic output.
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        """Background task: read stderr line-by-line and log at DEBUG level.

        Prevents the subprocess from deadlocking when the OS pipe buffer fills
        because nobody is consuming stderr output.
        """
        try:
            while self._proc and self._proc.stderr:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                logger.debug(
                    "MCP stderr [%s]: %s",
                    self._command[0],
                    line.decode("utf-8", errors="replace").rstrip(),
                )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("StdioTransport: stderr drain ended: %s", exc)

    async def send(self, message: dict) -> None:
        """Write a JSON line to the subprocess stdin."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("StdioTransport: not connected — call connect() first")
        line = json.dumps(message) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()
        logger.debug("StdioTransport: sent %s", message.get("method", "<unknown>"))

    async def receive(self) -> dict:
        """Read one JSON line from the subprocess stdout.

        Raises:
            ConnectionError: on EOF (subprocess closed its stdout).
            json.JSONDecodeError: if the line is not valid JSON.
        """
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("StdioTransport: not connected — call connect() first")
        line_bytes = await self._proc.stdout.readline()
        if not line_bytes:
            raise ConnectionError("StdioTransport: MCP server closed stdout (EOF)")
        line = line_bytes.decode("utf-8").strip()
        logger.debug("StdioTransport: received line: %s", line[:120])
        return json.loads(line)

    async def close(self) -> None:
        """Terminate the subprocess gracefully."""
        # Cancel and await the stderr drain task first to avoid resource leaks.
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):
                pass
            self._stderr_task = None

        if self._proc is None:
            return
        try:
            self._proc.terminate()
            await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except (ProcessLookupError, asyncio.TimeoutError):
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
        finally:
            self._proc = None
        logger.debug("StdioTransport: closed")
