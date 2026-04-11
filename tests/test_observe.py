from __future__ import annotations
import pytest
from pathlib import Path
from neoagent.observe import Observer
from neoagent.core.types import TextBlock, ToolUseBlock


@pytest.fixture
def observer(tmp_path: Path) -> Observer:
    obs = Observer(log_dir=tmp_path, console=False)
    yield obs
    obs.close()


def test_creates_log_file(tmp_path: Path) -> None:
    obs = Observer(log_dir=tmp_path, console=False)
    assert obs.log_path is not None
    assert obs.log_path.exists()
    obs.close()


def test_writes_to_file(tmp_path: Path) -> None:
    obs = Observer(log_dir=tmp_path, console=False)
    obs.on_tool_call("bash", {"command": "echo hi"})
    obs.close()
    content = obs.log_path.read_text()
    assert "bash" in content
    assert "echo hi" in content


def test_disabled_observer_writes_nothing(tmp_path: Path) -> None:
    obs = Observer(log_dir=tmp_path, console=False)
    obs.enabled = False
    obs.on_tool_call("bash", {"command": "echo hi"})
    obs.close()
    content = obs.log_path.read_text()
    assert content == ""


def test_console_only_no_crash() -> None:
    obs = Observer(log_dir=None, console=False)
    obs.on_tool_call("bash", {"command": "echo hi"})
    obs.on_provider_request("system", [], [], turn=1)
    obs.on_provider_response([TextBlock(text="hello")], "end_turn", 10, 5, turn=1)
    obs.close()  # no file to close


def test_on_provider_response_logs_tool_calls(tmp_path: Path) -> None:
    obs = Observer(log_dir=tmp_path, console=False)
    blocks = [TextBlock(text="ok"), ToolUseBlock(id="1", name="read", input={"path": "/tmp"})]
    obs.on_provider_response(blocks, "tool_use", 100, 50, turn=1)
    obs.close()
    content = obs.log_path.read_text()
    assert "read" in content
    assert "tool_use" in content


def test_on_compress_check(tmp_path: Path) -> None:
    obs = Observer(log_dir=tmp_path, console=False)
    obs.on_compress_check(800, 200, 1500, True)
    obs.close()
    content = obs.log_path.read_text()
    assert "TRIGGER" in content


def test_on_memory_extract_done(tmp_path: Path) -> None:
    obs = Observer(log_dir=tmp_path, console=False)
    obs.on_memory_extract_done(2, ["prefs.md", "goals.md"])
    obs.close()
    content = obs.log_path.read_text()
    assert "prefs.md" in content
    assert "2 item" in content


def test_observer_context_manager(tmp_path: Path) -> None:
    with Observer(log_dir=tmp_path, console=False) as obs:
        obs.on_tool_call("test", {"x": 1})
    # File should be closed after context exit
    content = obs.log_path.read_text()
    assert "test" in content
