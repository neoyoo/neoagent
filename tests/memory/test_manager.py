from __future__ import annotations
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from neoagent.core.types import Message, TextBlock
from neoagent.memory.store import MemoryStore
from neoagent.memory.manager import MemoryManager
from neoagent.providers.base import Response


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
    manager.record_tool_calls(3)
    manager.record_tool_calls(2)
    assert manager._tool_calls_count == 5


@pytest.mark.asyncio
async def test_maybe_extract_first_call_sets_baseline(manager: MemoryManager) -> None:
    await manager.maybe_extract(_msgs(), current_tokens=1000)
    assert manager._initial_token_estimate == 1000
    manager._extractor._provider.create.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_extract_triggers_on_high_tool_calls(tmp_path: Path) -> None:
    items = [{"filename": "g.md", "description": "goals", "content": "# G\n- x\n"}]
    provider = _make_provider(json.dumps(items))
    store = MemoryStore(tmp_path)
    manager = MemoryManager(store, provider)
    await manager.maybe_extract(_msgs(), current_tokens=500)
    manager.record_tool_calls(5)
    await manager.maybe_extract(_msgs(), current_tokens=600)
    assert store.read_topic("g.md") != ""
