# tests/v2/test_compressor_with_strategy.py
"""Task 4.6 — ContextCompressor with CompressionStrategy injection.

Spec refs:
  § 10.2 (CompressionStrategy / CompressionDelta)
  § 10.8 (Deterministic Executor — orphan id validation)
  § 8.3  (BatchCreatedEvent / CompressionFailedEvent)
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from neoagent.core.compress import ContextCompressor
from neoagent.core.types import Message, TextBlock
from neoagent.events import (
    BatchCreatedEvent,
    CompressionFailedEvent,
    EventBus,
    WorkingMemoryUpdatedEvent,
)
from neoagent.providers.base import Provider, Response
from neoagent.session import Session, SessionState
from neoagent.v2.abc import CompressionStrategy
from neoagent.v2.errors import CompressionError
from neoagent.v2.schema import BatchMember, CompressionDelta, WorkingMemory


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_provider(text: str = "compressed summary") -> Provider:
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


def _make_wm(session_id: str = "s1", version: int = 1) -> WorkingMemory:
    return WorkingMemory(
        session_id=session_id,
        version=version,
        at_turn=0,
        constraints_and_preferences=[],
        progress="",
        key_decisions=[],
        relevant_files=[],
        next_steps=[],
        critical_context="",
        updated_by="framework_init",
        updated_at=datetime(2026, 4, 22, 12, 0, 0),
    )


def _make_messages(n: int = 5) -> list[Message]:
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(Message(role=role, content=f"message {i}"))
    return msgs


def _make_delta(
    batch_members: list[BatchMember] | None = None,
    wm_ops: list[dict] | None = None,
) -> CompressionDelta:
    members = batch_members or [
        BatchMember(id="m0", role="user", preview="msg0"),
        BatchMember(id="m1", role="assistant", preview="msg1"),
    ]
    ops = wm_ops or []
    return CompressionDelta(
        batch_summary="PROGRESS: some\nDECISIONS:\nFILES:\nNEXT STEPS:\nKEY CONTEXT:",
        batch_members=members,
        working_memory_delta=ops,
    )


def _make_session_state(wm: WorkingMemory | None = None) -> SessionState:
    state = SessionState()
    state._current_wm = wm
    return state


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTask46ContextCompressorWithStrategy:

    @pytest.mark.asyncio
    async def test_strategy_none_fallback_legacy(self):
        """strategy=None → legacy LLM-compress path; strategy.compress() never called."""
        provider = _make_provider()
        strategy = AsyncMock(spec=CompressionStrategy)

        compressor = ContextCompressor(provider=provider, strategy=None)
        msgs = _make_messages(10)
        state = _make_session_state()

        # Should not call strategy.compress at all
        result = await compressor.compress(msgs, context_budget=200_000, session_state=state)
        strategy.compress.assert_not_awaited()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_strategy_success_batch_created_event(self):
        """strategy returns valid delta → Batch written + BatchCreatedEvent emitted."""
        provider = _make_provider()
        strategy = AsyncMock(spec=CompressionStrategy)

        # Messages with ids (as dicts to simulate what compressor might see;
        # we'll pass Message objects but the compressor builds CompressionContext
        # using session messages — use a session_state with id_gen)
        msgs = _make_messages(5)
        wm = _make_wm()
        state = _make_session_state(wm=wm)

        delta = _make_delta(
            batch_members=[BatchMember(id="m0", role="user", preview="msg0")],
        )
        strategy.compress = AsyncMock(return_value=delta)

        bus = EventBus()
        batch_events: list[BatchCreatedEvent] = []
        bus.subscribe(BatchCreatedEvent, batch_events.append)

        compressor = ContextCompressor(provider=provider, strategy=strategy, event_bus=bus)

        # Override should_compress to True so compress path is called
        # We call compress() directly to test the strategy path
        result = await compressor.compress(msgs, context_budget=1, session_state=state)

        strategy.compress.assert_awaited_once()
        assert len(batch_events) == 1
        evt = batch_events[0]
        assert evt.session_id == "s1"
        assert evt.summary == delta.batch_summary
        assert len(evt.members) == 1

    @pytest.mark.asyncio
    async def test_strategy_raises_compression_failed_event(self):
        """strategy raises CompressionError → CompressionFailedEvent; messages unchanged."""
        provider = _make_provider()
        strategy = AsyncMock(spec=CompressionStrategy)
        strategy.compress = AsyncMock(side_effect=CompressionError("retry exhausted"))

        msgs = _make_messages(5)
        original_len = len(msgs)
        wm = _make_wm()
        state = _make_session_state(wm=wm)

        bus = EventBus()
        failed_events: list[CompressionFailedEvent] = []
        bus.subscribe(CompressionFailedEvent, failed_events.append)

        compressor = ContextCompressor(provider=provider, strategy=strategy, event_bus=bus)
        result = await compressor.compress(msgs, context_budget=1, session_state=state)

        # messages should be unchanged (circuit break — no modification)
        assert result == msgs
        assert len(failed_events) == 1
        evt = failed_events[0]
        assert evt.session_id == "s1"
        assert "retry exhausted" in evt.reason

    @pytest.mark.asyncio
    async def test_deterministic_executor_orphan_id_rejected(self):
        """batch_members contains id not in messages → CompressionFailedEvent(reason='orphan_id:...')."""
        provider = _make_provider()
        strategy = AsyncMock(spec=CompressionStrategy)

        msgs = _make_messages(3)
        wm = _make_wm()
        state = _make_session_state(wm=wm)
        # Simulate 3 tracked messages (m1, m2, m3). "m99" is clearly outside range.
        state.id_gen._msg_counter = 3

        # "m99" does not exist in tracked messages (m1..m3)
        delta = _make_delta(
            batch_members=[BatchMember(id="m99", role="user", preview="ghost msg")],
        )
        strategy.compress = AsyncMock(return_value=delta)

        bus = EventBus()
        failed_events: list[CompressionFailedEvent] = []
        bus.subscribe(CompressionFailedEvent, failed_events.append)

        compressor = ContextCompressor(provider=provider, strategy=strategy, event_bus=bus)
        result = await compressor.compress(msgs, context_budget=1, session_state=state)

        assert len(failed_events) == 1
        evt = failed_events[0]
        assert "orphan_id:m99" in evt.reason
        # messages unchanged
        assert result == msgs

    @pytest.mark.asyncio
    async def test_apply_wm_delta_correct(self):
        """delta with append op → _current_wm.key_decisions updated."""
        provider = _make_provider()
        strategy = AsyncMock(spec=CompressionStrategy)

        msgs = _make_messages(3)
        wm = _make_wm()
        state = _make_session_state(wm=wm)

        delta = CompressionDelta(
            batch_summary="test summary",
            batch_members=[],
            working_memory_delta=[
                {"field": "key_decisions", "op": "append", "value": "d01: important decision"},
            ],
        )
        strategy.compress = AsyncMock(return_value=delta)

        compressor = ContextCompressor(provider=provider, strategy=strategy, event_bus=None)
        await compressor.compress(msgs, context_budget=1, session_state=state)

        assert wm.key_decisions == ["d01: important decision"]

    @pytest.mark.asyncio
    async def test_wm_version_at_turn_updated_by_updated(self):
        """apply_delta bumps version +1, sets at_turn, updated_by='framework_compression'."""
        provider = _make_provider()
        strategy = AsyncMock(spec=CompressionStrategy)

        msgs = _make_messages(3)
        wm = _make_wm(version=5)
        state = _make_session_state(wm=wm)

        delta = _make_delta(batch_members=[])
        strategy.compress = AsyncMock(return_value=delta)

        compressor = ContextCompressor(provider=provider, strategy=strategy, event_bus=None)
        # Compress at turn=0 equivalent (session_state has no explicit turn info —
        # compressor should use the turn counter it receives or a sensible default)
        await compressor.compress(msgs, context_budget=1, session_state=state, current_turn=4)

        assert wm.version == 6
        assert wm.at_turn == 4
        assert wm.updated_by == "framework_compression"

    @pytest.mark.asyncio
    async def test_event_bus_none_no_emit(self):
        """event_bus=None → no exceptions when events would otherwise be emitted."""
        provider = _make_provider()
        strategy = AsyncMock(spec=CompressionStrategy)

        msgs = _make_messages(3)
        wm = _make_wm()
        state = _make_session_state(wm=wm)

        delta = _make_delta(batch_members=[])
        strategy.compress = AsyncMock(return_value=delta)

        # No event_bus — should not raise
        compressor = ContextCompressor(provider=provider, strategy=strategy, event_bus=None)
        result = await compressor.compress(msgs, context_budget=1, session_state=state)
        assert isinstance(result, list)
