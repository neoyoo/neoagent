from __future__ import annotations
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from neoagent.mcp.transport import MCPTransport, StdioTransport


# ── MCPTransport is abstract ─────────────────────────────────────────────────

def test_mcptransport_is_abstract():
    """MCPTransport cannot be instantiated directly."""
    import inspect
    assert inspect.isabstract(MCPTransport)


def test_stdiotransport_is_mcptransport():
    t = StdioTransport(command=["echo", "hi"])
    assert isinstance(t, MCPTransport)


# ── StdioTransport construction ──────────────────────────────────────────────

def test_stdiotransport_stores_command():
    t = StdioTransport(command=["npx", "my-server"])
    assert t._command == ["npx", "my-server"]


def test_stdiotransport_stores_extra_env():
    t = StdioTransport(command=["node", "server.js"], env={"MY_TOKEN": "secret"})
    assert t._extra_env == {"MY_TOKEN": "secret"}


def test_stdiotransport_no_extra_env_is_empty():
    t = StdioTransport(command=["node", "server.js"])
    assert t._extra_env == {}


# ── _safe_env whitelist ──────────────────────────────────────────────────────

def test_stdiotransport_safe_env_only_whitelisted(monkeypatch):
    """_safe_env() must only pass whitelisted keys from the parent env."""
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("SECRET_API_KEY", "should_not_pass")
    monkeypatch.setenv("OPENAI_API_KEY", "also_blocked")
    t = StdioTransport(command=["echo"])
    safe = t._safe_env()
    assert "PATH" in safe
    assert "SECRET_API_KEY" not in safe
    assert "OPENAI_API_KEY" not in safe


def test_stdiotransport_safe_env_merges_extra_env(monkeypatch):
    """Extra env keys passed to constructor are merged into _safe_env()."""
    monkeypatch.setenv("PATH", "/usr/bin")
    t = StdioTransport(command=["echo"], env={"MY_TOKEN": "abc123"})
    safe = t._safe_env()
    assert safe["MY_TOKEN"] == "abc123"
    assert "PATH" in safe


def test_stdiotransport_extra_env_not_in_whitelist_still_passes():
    """Extra env keys are always included even if not in the default whitelist."""
    t = StdioTransport(command=["echo"], env={"CUSTOM_KEY": "val"})
    safe = t._safe_env()
    assert safe["CUSTOM_KEY"] == "val"


# ── StdioTransport connect/send/receive/close with mock subprocess ───────────

@pytest.mark.asyncio
async def test_stdiotransport_connect_creates_subprocess():
    """connect() must call asyncio.create_subprocess_exec with the command."""
    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = AsyncMock()
    mock_proc.returncode = None

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        t = StdioTransport(command=["node", "server.js"])
        await t.connect()
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args
        assert call_args[0][0] == "node"
        assert call_args[0][1] == "server.js"


@pytest.mark.asyncio
async def test_stdiotransport_send_writes_json_line():
    """send() must write JSON + newline to the subprocess stdin."""
    mock_proc = MagicMock()
    written = []

    async def fake_drain():
        pass

    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write = lambda data: written.append(data)
    mock_proc.stdin.drain = fake_drain
    mock_proc.stdout = AsyncMock()
    mock_proc.returncode = None

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        t = StdioTransport(command=["echo"])
        await t.connect()
        await t.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})

    assert len(written) == 1
    decoded = written[0].decode("utf-8")
    parsed = json.loads(decoded.strip())
    assert parsed["method"] == "ping"


@pytest.mark.asyncio
async def test_stdiotransport_receive_reads_json_line():
    """receive() must read a line from stdout and parse it as JSON."""
    response_line = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n"

    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = AsyncMock()
    mock_proc.stdout.readline = AsyncMock(return_value=response_line.encode())
    mock_proc.returncode = None

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        t = StdioTransport(command=["echo"])
        await t.connect()
        msg = await t.receive()

    assert msg["id"] == 1
    assert msg["result"] == {}


@pytest.mark.asyncio
async def test_stdiotransport_close_terminates_process():
    """close() must terminate the subprocess."""
    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = AsyncMock()
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        t = StdioTransport(command=["echo"])
        await t.connect()
        await t.close()

    mock_proc.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_stdiotransport_close_before_connect_no_error():
    """close() before connect() must not raise."""
    t = StdioTransport(command=["echo"])
    await t.close()  # should not raise


@pytest.mark.asyncio
async def test_stdiotransport_receive_eof_raises():
    """receive() on EOF (empty readline) must raise ConnectionError."""
    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = AsyncMock()
    mock_proc.stdout.readline = AsyncMock(return_value=b"")
    mock_proc.stderr = AsyncMock()
    mock_proc.stderr.readline = AsyncMock(return_value=b"")
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        t = StdioTransport(command=["echo"])
        await t.connect()
        with pytest.raises(ConnectionError):
            await t.receive()
        await t.close()


# ── StdioTransport stderr drain ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stdiotransport_connect_starts_stderr_drain_task():
    """connect() must create a background task to drain stderr."""
    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = AsyncMock()
    mock_proc.stderr = AsyncMock()
    # stderr.readline returns EOF immediately so drain task exits cleanly
    mock_proc.stderr.readline = AsyncMock(return_value=b"")
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        t = StdioTransport(command=["node", "server.js"])
        await t.connect()
        assert t._stderr_task is not None
        # Drain task should be an asyncio.Task
        import asyncio as _asyncio
        assert isinstance(t._stderr_task, _asyncio.Task)
        await t.close()
        assert t._stderr_task is None


@pytest.mark.asyncio
async def test_stdiotransport_close_cancels_stderr_task():
    """close() must cancel the stderr drain task."""
    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = AsyncMock()
    mock_proc.stderr = AsyncMock()
    # Keep stderr.readline blocking so the task stays alive until cancelled
    cancelled_event = asyncio.Event()

    async def blocking_readline():
        await asyncio.sleep(9999)  # stays blocked until cancelled
        return b""

    mock_proc.stderr.readline = blocking_readline
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        t = StdioTransport(command=["node", "server.js"])
        await t.connect()
        task = t._stderr_task
        assert task is not None and not task.done()

        await t.close()

    # After close(), the task must have been cancelled and cleaned up
    assert t._stderr_task is None
    assert task.done()


@pytest.mark.asyncio
async def test_stdiotransport_stderr_lines_are_logged(caplog):
    """Stderr output from the MCP server must be logged at DEBUG level."""
    import logging

    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = AsyncMock()
    mock_proc.stderr = AsyncMock()

    lines = [b"MCP server started\n", b"debug info\n", b""]
    line_iter = iter(lines)
    mock_proc.stderr.readline = AsyncMock(side_effect=lambda: next(line_iter))
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with caplog.at_level(logging.DEBUG, logger="neoagent.mcp.transport"):
            t = StdioTransport(command=["node", "server.js"])
            await t.connect()
            # Allow the drain task to process the stderr lines
            await asyncio.sleep(0.05)
            await t.close()

    logged = caplog.text
    assert "MCP server started" in logged or "debug info" in logged
