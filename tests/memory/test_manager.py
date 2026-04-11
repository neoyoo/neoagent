from __future__ import annotations
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from neoagent.core.types import Message, TextBlock
from neoagent.memory.store import MemoryStore
from neoagent.memory.manager import MemoryManager
from neoagent.providers.base import Response
from neoagent.session import SessionState


def _make_provider(text: str = "[]") -> MagicMock:
    resp = Response(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=10,
    )
    provider = MagicMock()
    provider.create = AsyncMock(return_value=resp)
    return provider


def _msgs(n: int = 3) -> list[Message]:
    return [Message(role="user", content=f"msg {i}") for i in range(n)]


@pytest.fixture
def manager(tmp_path: Path) -> MemoryManager:
    store = MemoryStore(tmp_path)
    return MemoryManager(store, _make_provider())


def test_build_prompt_section_empty(manager: MemoryManager) -> None:
    result = manager.build_prompt_section()
    assert result == ""


def test_record_tool_calls_accumulates(manager: MemoryManager) -> None:
    state = SessionState()
    manager.record_tool_calls(3, session_state=state)
    manager.record_tool_calls(2, session_state=state)
    assert state.memory_tool_calls == 5


@pytest.mark.asyncio
async def test_maybe_extract_first_call_sets_baseline(manager: MemoryManager) -> None:
    # First call with 0 tool_calls and 0 token_delta → sets baseline, no extraction
    state = SessionState()
    await manager.maybe_extract(_msgs(), current_tokens=1000, session_state=state)
    assert state.memory_token_baseline == 1000
    manager._extractor._provider.create.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_extract_first_call_can_trigger_if_tool_calls_high(tmp_path: Path) -> None:
    """First call should still trigger extraction if tool_calls >= threshold."""
    items = [{"filename": "g.md", "description": "goals", "content": "# G\n- x\n"}]
    provider = _make_provider(json.dumps(items))
    store = MemoryStore(tmp_path)
    manager = MemoryManager(store, provider)
    state = SessionState()
    manager.record_tool_calls(5, session_state=state)
    await manager.maybe_extract(_msgs(), current_tokens=500, session_state=state)
    assert store.read_topic("g.md") != ""


@pytest.mark.asyncio
async def test_maybe_extract_triggers_on_high_tool_calls(tmp_path: Path) -> None:
    items = [{"filename": "g.md", "description": "goals", "content": "# G\n- x\n"}]
    provider = _make_provider(json.dumps(items))
    store = MemoryStore(tmp_path)
    manager = MemoryManager(store, provider)
    state = SessionState()
    await manager.maybe_extract(_msgs(), current_tokens=500, session_state=state)
    manager.record_tool_calls(5, session_state=state)
    await manager.maybe_extract(_msgs(), current_tokens=600, session_state=state)
    assert store.read_topic("g.md") != ""


@pytest.mark.asyncio
async def test_maybe_extract_resets_token_baseline(tmp_path: Path) -> None:
    """After successful extraction, memory_token_baseline must be updated to
    current_tokens so token_delta resets to zero, preventing runaway extractions."""
    items = [{"filename": "g.md", "description": "goals", "content": "# G\n- x\n"}]
    provider = _make_provider(json.dumps(items))
    store = MemoryStore(tmp_path)
    manager = MemoryManager(store, provider)
    state = SessionState()

    # Establish baseline at 1000 tokens
    await manager.maybe_extract(_msgs(), current_tokens=1000, session_state=state)
    assert state.memory_token_baseline == 1000

    # Trigger extraction via tool_calls threshold at 5000 tokens
    manager.record_tool_calls(5, session_state=state)
    await manager.maybe_extract(_msgs(), current_tokens=5000, session_state=state)

    # Baseline must have been reset to 5000, not remain at 1000
    assert state.memory_token_baseline == 5000, (
        f"Expected baseline reset to 5000 after extraction, got {state.memory_token_baseline}"
    )
    # tool_calls must also be reset
    assert state.memory_tool_calls == 0


# --- NEW: Session integration tests ---

def test_manager_no_instance_state_vars(manager: MemoryManager) -> None:
    """MemoryManager must NOT have _tool_calls_count or _initial_token_estimate as instance attrs."""
    assert not hasattr(manager, "_tool_calls_count"), (
        "_tool_calls_count should be removed — it lives in SessionState now"
    )
    assert not hasattr(manager, "_initial_token_estimate"), (
        "_initial_token_estimate should be removed — it lives in SessionState now"
    )


def test_record_tool_calls_uses_session_state() -> None:
    """record_tool_calls() writes to session_state.memory_tool_calls, not instance var."""
    store = MemoryStore(Path("/tmp"))
    manager = MemoryManager(store, _make_provider())
    state = SessionState()
    assert state.memory_tool_calls == 0
    manager.record_tool_calls(7, session_state=state)
    assert state.memory_tool_calls == 7


@pytest.mark.asyncio
async def test_maybe_extract_sets_baseline_in_state(tmp_path: Path) -> None:
    """maybe_extract() writes token baseline to session_state.memory_token_baseline."""
    store = MemoryStore(tmp_path)
    manager = MemoryManager(store, _make_provider())
    state = SessionState()
    assert state.memory_token_baseline == 0
    await manager.maybe_extract(_msgs(), current_tokens=2500, session_state=state)
    assert state.memory_token_baseline == 2500


@pytest.mark.asyncio
async def test_maybe_extract_no_session_state_is_noop(tmp_path: Path) -> None:
    """maybe_extract() with no session_state is a safe no-op."""
    store = MemoryStore(tmp_path)
    manager = MemoryManager(store, _make_provider())
    # Should not raise
    await manager.maybe_extract(_msgs(), current_tokens=1000, session_state=None)


def test_record_tool_calls_no_session_state_is_noop() -> None:
    """record_tool_calls() with no session_state is a safe no-op."""
    store = MemoryStore(Path("/tmp"))
    manager = MemoryManager(store, _make_provider())
    # Should not raise
    manager.record_tool_calls(5, session_state=None)
