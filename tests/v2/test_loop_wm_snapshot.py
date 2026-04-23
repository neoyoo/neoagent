# tests/v2/test_loop_wm_snapshot.py
"""Task 4.5 — QueryLoop WM snapshot + WorkingMemoryUpdatedEvent.

Spec refs:
  § 2.3a (WorkingMemoryStore.save contract)
  § 8.3 (WorkingMemoryUpdatedEvent payload)
  § 15.2, § 15.3 (snapshot timing relative to TurnCompleteEvent)
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from neoagent.core.loop import QueryLoop
from neoagent.core.types import Message, TextBlock
from neoagent.events import EventBus, TurnCompleteEvent, WorkingMemoryUpdatedEvent
from neoagent.providers.base import Provider, Response
from neoagent.session import Session, SessionState
from neoagent.tools.registry import ToolRegistry
from neoagent.v2.abc import WorkingMemoryStore
from neoagent.v2.schema import WorkingMemory


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wm(session_id: str = "s1", version: int = 2, at_turn: int = 3) -> WorkingMemory:
    return WorkingMemory(
        session_id=session_id,
        version=version,
        at_turn=at_turn,
        goal="test goal",
        constraints_and_preferences=[],
        progress="some progress",
        key_decisions=[],
        relevant_files=[],
        next_steps=[],
        critical_context="",
        updated_by="framework_compression",
        updated_at=datetime(2026, 4, 22, 12, 0, 0),
    )


def _make_provider(text: str = "Hello!") -> Provider:
    """Return a mock Provider that returns a simple end_turn response."""
    provider = MagicMock(spec=Provider)
    response = Response(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
    )
    provider.create = AsyncMock(return_value=response)
    provider.get_context_window = MagicMock(return_value=200_000)
    return provider


def _make_loop(
    wm_store: WorkingMemoryStore | None = None,
    event_bus: EventBus | None = None,
) -> QueryLoop:
    import warnings
    from neoagent.core.prompt import PromptBuilder

    provider = _make_provider()
    registry = ToolRegistry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        builder = PromptBuilder()
    bus = event_bus or EventBus()
    return QueryLoop(
        provider=provider,
        tool_registry=registry,
        prompt_builder=builder,
        event_bus=bus,
        wm_store=wm_store,
    )


def _make_session(wm: WorkingMemory | None = None) -> Session:
    session = Session.create(session_id="s1")
    if wm is not None:
        session.state._current_wm = wm
    return session


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTask45WmSnapshot:
    """Task 4.5: loop.py WM snapshot + WorkingMemoryUpdatedEvent."""

    @pytest.mark.asyncio
    async def test_wm_store_none_no_snapshot(self):
        """wm_store=None → no snapshot side-effect."""
        wm = _make_wm()
        session = _make_session(wm=wm)
        session.messages.append(Message(role="user", content="Hi"))

        loop = _make_loop(wm_store=None)
        await loop.run(session=session)
        # No wm_store — nothing to assert, just no AttributeError

    @pytest.mark.asyncio
    async def test_current_wm_none_no_snapshot(self):
        """_current_wm=None → save() not called even when wm_store is present."""
        wm_store = AsyncMock(spec=WorkingMemoryStore)
        session = _make_session(wm=None)
        session.messages.append(Message(role="user", content="Hi"))

        loop = _make_loop(wm_store=wm_store)
        await loop.run(session=session)

        wm_store.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_normal_snapshot_called_once(self):
        """_current_wm + wm_store both present → save() called exactly once."""
        wm_store = AsyncMock(spec=WorkingMemoryStore)
        wm = _make_wm(session_id="s1", version=2)
        session = _make_session(wm=wm)
        session.messages.append(Message(role="user", content="Hi"))

        loop = _make_loop(wm_store=wm_store)
        await loop.run(session=session)

        wm_store.save.assert_awaited_once()
        call_args = wm_store.save.call_args
        assert call_args[0][0] == "s1"       # session_id
        assert call_args[0][1] is wm          # the WM object

    @pytest.mark.asyncio
    async def test_wm_updated_event_emitted(self):
        """WorkingMemoryUpdatedEvent emitted with correct payload fields.

        spec § 2.3a: the snapshot path bumps version (6 → 7) and sets
        at_turn=version, so the event carries the post-bump values.
        """
        wm_store = AsyncMock(spec=WorkingMemoryStore)
        wm = _make_wm(session_id="s1", version=6, at_turn=2)
        session = _make_session(wm=wm)
        session.messages.append(Message(role="user", content="Hi"))

        bus = EventBus()
        received: list[WorkingMemoryUpdatedEvent] = []
        bus.subscribe(WorkingMemoryUpdatedEvent, received.append)

        loop = _make_loop(wm_store=wm_store, event_bus=bus)
        await loop.run(session=session)

        assert len(received) == 1
        evt = received[0]
        assert evt.session_id == "s1"
        assert evt.version == 7          # bumped 6 → 7
        assert evt.at_turn == 7          # aligned to new version
        assert evt.updated_by == "framework_compression"
        assert isinstance(evt.wm_json, dict)
        assert evt.wm_json["version"] == 7

    @pytest.mark.asyncio
    async def test_version_and_at_turn_bumped_on_snapshot(self):
        """spec § 2.3a — SDK bumps version/at_turn on every end_turn snapshot.

        The tool update_working_memory intentionally does NOT bump version
        (see its docstring). The snapshot path in loop.py owns it: each
        end_turn snapshot increments version by 1 and sets at_turn=version.
        """
        wm_store = AsyncMock(spec=WorkingMemoryStore)
        # framework_init state: version=0, at_turn=0
        wm = _make_wm(session_id="s1", version=0, at_turn=0)
        session = _make_session(wm=wm)
        session.messages.append(Message(role="user", content="Hi"))

        bus = EventBus()
        received: list[WorkingMemoryUpdatedEvent] = []
        bus.subscribe(WorkingMemoryUpdatedEvent, received.append)

        loop = _make_loop(wm_store=wm_store, event_bus=bus)
        await loop.run(session=session)

        # After one completed turn: version=1, at_turn=1
        assert session.state._current_wm.version == 1
        assert session.state._current_wm.at_turn == 1
        assert len(received) == 1
        assert received[0].version == 1
        assert received[0].at_turn == 1

        # Second run: version=2, at_turn=2
        session.messages.append(Message(role="user", content="Again"))
        await loop.run(session=session)
        assert session.state._current_wm.version == 2
        assert session.state._current_wm.at_turn == 2
        assert received[-1].version == 2
        assert received[-1].at_turn == 2

    @pytest.mark.asyncio
    async def test_updated_at_refreshed_on_snapshot(self):
        """updated_at must be refreshed at snapshot so clients see fresh timestamps."""
        wm_store = AsyncMock(spec=WorkingMemoryStore)
        stale = datetime(2020, 1, 1, 0, 0, 0)
        wm = _make_wm(session_id="s1", version=0, at_turn=0)
        wm.updated_at = stale
        session = _make_session(wm=wm)
        session.messages.append(Message(role="user", content="Hi"))

        loop = _make_loop(wm_store=wm_store)
        await loop.run(session=session)

        assert session.state._current_wm.updated_at > stale

    @pytest.mark.asyncio
    async def test_wm_updated_before_turn_complete(self):
        """WorkingMemoryUpdatedEvent emitted BEFORE TurnCompleteEvent."""
        wm_store = AsyncMock(spec=WorkingMemoryStore)
        wm = _make_wm()
        session = _make_session(wm=wm)
        session.messages.append(Message(role="user", content="Hi"))

        bus = EventBus()
        order: list[str] = []
        bus.subscribe(WorkingMemoryUpdatedEvent, lambda e: order.append("wm_updated"))
        bus.subscribe(TurnCompleteEvent, lambda e: order.append("turn_complete"))

        loop = _make_loop(wm_store=wm_store, event_bus=bus)
        await loop.run(session=session)

        # WM snapshot must precede TurnCompleteEvent
        assert "wm_updated" in order
        assert "turn_complete" in order
        assert order.index("wm_updated") < order.index("turn_complete")
