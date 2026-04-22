# tests/v2/test_memory_provider_integration.py
"""TDD tests for Task 7.3 — MemoryManager accepts Optional MemoryProvider.

Spec refs: § 12.2
Contract refs: C2 (MemoryProvider)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neoagent.config import NeoAgentConfig
from neoagent.core.types import Message
from neoagent.memory.manager import MemoryManager
from neoagent.memory.store import MemoryStore
from neoagent.session import SessionState
from neoagent.v2.abc import MemoryProvider
from neoagent.v2.schema import MemoryEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(memory_id: str = "m1") -> MemoryEntry:
    return MemoryEntry(
        user_id="default",
        memory_id=memory_id,
        type="fact",
        category=None,
        content="test memory",
        created_at=datetime.now(),
    )


def _make_provider_mock() -> MagicMock:
    mock = MagicMock(spec=MemoryProvider)
    mock.upsert = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.delete = AsyncMock()
    mock.reinforce = AsyncMock()
    return mock


def _make_llm_provider(text: str = "[]") -> MagicMock:
    from neoagent.providers.base import Response
    from neoagent.core.types import TextBlock
    resp = Response(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=10,
    )
    prov = MagicMock()
    prov.create = AsyncMock(return_value=resp)
    return prov


def _msgs(n: int = 3) -> list[Message]:
    return [Message(role="user", content=f"msg {i}") for i in range(n)]


def _make_config(**kwargs) -> NeoAgentConfig:
    return NeoAgentConfig(api_key="test-key", model="test-model", **kwargs)


def _make_agent(config: NeoAgentConfig | None = None):
    from neoagent.agent import NeoAgent
    if config is None:
        config = _make_config()
    with patch("neoagent.agent._create_provider") as mock_create:
        mock_prov = MagicMock()
        mock_prov.get_context_window.return_value = 200_000
        mock_prov.model = "test-model"
        mock_create.return_value = mock_prov
        agent = NeoAgent(config)
    return agent


# ---------------------------------------------------------------------------
# P1: MemoryManager default uses MemoryStore (back-compat)
# ---------------------------------------------------------------------------


def test_memory_manager_default_no_provider(tmp_path: Path):
    """P1: MemoryManager without memory_provider uses MemoryStore path (back-compat)."""
    store = MemoryStore(tmp_path)
    llm = _make_llm_provider()
    manager = MemoryManager(store, llm)
    # No provider — _memory_provider is None or absent
    assert getattr(manager, "_memory_provider", None) is None


@pytest.mark.asyncio
async def test_memory_manager_default_path_no_exception(tmp_path: Path):
    """P1b: MemoryManager without provider runs maybe_extract without error."""
    store = MemoryStore(tmp_path)
    llm = _make_llm_provider()
    manager = MemoryManager(store, llm)
    state = SessionState()
    # Should not raise
    result = await manager.maybe_extract(_msgs(), current_tokens=100, session_state=state)
    assert result == (False, 0)


# ---------------------------------------------------------------------------
# P2: MemoryManager with injected provider — save routes to upsert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_manager_with_provider_extraction_calls_upsert(tmp_path: Path):
    """P2: When memory_provider is set, extraction calls provider.upsert instead of MemoryStore."""
    items = [{"filename": "g.md", "description": "goals", "content": "# G\n- x\n"}]
    llm = _make_llm_provider(json.dumps(items))
    store = MemoryStore(tmp_path)

    mock_provider = _make_provider_mock()
    manager = MemoryManager(store, llm, memory_provider=mock_provider)

    state = SessionState()
    # Trigger extraction via tool_calls threshold
    manager.record_tool_calls(5, session_state=state)
    triggered, items_stored = await manager.maybe_extract(_msgs(), current_tokens=500, session_state=state)

    assert triggered is True
    mock_provider.upsert.assert_called_once()
    call_args = mock_provider.upsert.call_args[0][0]
    assert isinstance(call_args, list)
    assert len(call_args) > 0
    # Each item should be a MemoryEntry
    for entry in call_args:
        assert isinstance(entry, MemoryEntry)


@pytest.mark.asyncio
async def test_memory_manager_with_provider_no_extraction_no_upsert(tmp_path: Path):
    """P2b: When extraction does not trigger, upsert is not called."""
    llm = _make_llm_provider()
    store = MemoryStore(tmp_path)
    mock_provider = _make_provider_mock()
    manager = MemoryManager(store, llm, memory_provider=mock_provider)

    state = SessionState()
    # Not enough tool calls to trigger
    await manager.maybe_extract(_msgs(), current_tokens=100, session_state=state)

    mock_provider.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# P3: search routes to provider.search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_prompt_section_with_provider_calls_search(tmp_path: Path):
    """P3: build_prompt_section with provider calls provider.search(user_id, query, k)."""
    store = MemoryStore(tmp_path)
    llm = _make_llm_provider()
    mock_provider = _make_provider_mock()
    mock_provider.search = AsyncMock(return_value=[_make_entry()])

    manager = MemoryManager(store, llm, memory_provider=mock_provider)

    # build_prompt_section is sync in existing code — if it becomes async after provider
    # injection, the test adapts; if it stays sync, search is called synchronously via
    # asyncio.run internally or returns cached results.
    # For now, test the search pathway via a direct retrieval call.
    result = await manager.search_with_provider("default", "test query", k=3)

    mock_provider.search.assert_called_once_with("default", "test query", k=3)


# ---------------------------------------------------------------------------
# P4: NeoAgent end-to-end — enable_memory uses provider when configured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_neoagent_memory_manager_wired_with_provider(tmp_path: Path):
    """P4: NeoAgent with memory_provider wires MemoryManager with that provider."""
    mock_provider = _make_provider_mock()
    cfg = _make_config(memory_provider=mock_provider)
    agent = _make_agent(cfg)

    # Call enable_memory — should wire MemoryManager with the provider from config
    agent.enable_memory(memory_dir=tmp_path)

    mm = agent._loop._memory_manager
    assert mm is not None
    assert mm._memory_provider is mock_provider


# ---------------------------------------------------------------------------
# P5: Back-compat — enable_memory without provider still works
# ---------------------------------------------------------------------------


def test_neoagent_enable_memory_without_provider_back_compat(tmp_path: Path):
    """P5: enable_memory without memory_provider still constructs MemoryManager correctly."""
    cfg = _make_config(memory_provider=None)
    agent = _make_agent(cfg)

    agent.enable_memory(memory_dir=tmp_path)

    mm = agent._loop._memory_manager
    assert mm is not None
    assert getattr(mm, "_memory_provider", None) is None
