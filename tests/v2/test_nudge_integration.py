# tests/v2/test_nudge_integration.py
"""TDD tests for Task 7.2 — NudgeCounter subscribed to TurnCompleteEvent.

Spec refs: § 11.4
Contract refs: C2 (MemoryReviewStrategy, MemoryProvider)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neoagent.config import NeoAgentConfig
from neoagent.events import EventBus, TurnCompleteEvent
from neoagent.session import Session, SessionState
from neoagent.v2.abc import MemoryProvider, MemoryReviewStrategy
from neoagent.v2.nudge import NudgeCounter
from neoagent.v2.schema import MemoryEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_turn_complete_event(turn_index: int = 0) -> TurnCompleteEvent:
    return TurnCompleteEvent(
        turn_index=turn_index,
        stop_reason="end_turn",
        tool_call_count=0,
    )


class _StubProvider(MemoryProvider):
    async def search(self, user_id, query, k=5):
        return []
    async def upsert(self, entries):
        pass
    async def delete(self, user_id, memory_ids):
        pass
    async def reinforce(self, user_id, memory_id):
        pass


class _StubReview(MemoryReviewStrategy):
    async def review(self, session_id, user_id, messages, wm):
        return []


def _make_entry() -> MemoryEntry:
    from datetime import datetime
    return MemoryEntry(
        user_id="default",
        memory_id="m1",
        type="fact",
        category=None,
        content="test memory",
        created_at=datetime.now(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_each_turn_count_accumulates():
    """T1: After 10 TurnCompleteEvents, nudge_counter._count == 10."""
    agent = _make_agent()
    session = Session.create()
    agent._current_session = session

    for i in range(10):
        agent._event_bus.emit(_make_turn_complete_event(i))

    # Allow any async tasks to run
    await asyncio.sleep(0)

    assert session.state.nudge_counter._count == 10


@pytest.mark.asyncio
async def test_turn_10_triggers_review():
    """T2: At turn 10, memory_review_strategy.review is called once."""
    mock_review = MagicMock(spec=MemoryReviewStrategy)
    mock_review.review = AsyncMock(return_value=[])
    mock_provider = MagicMock(spec=MemoryProvider)
    mock_provider.upsert = AsyncMock()

    cfg = _make_config(
        memory_review_strategy=mock_review,
        memory_provider=mock_provider,
    )
    agent = _make_agent(cfg)
    session = Session.create()
    agent._current_session = session

    for i in range(10):
        agent._event_bus.emit(_make_turn_complete_event(i))

    # Give async tasks a chance to run
    await asyncio.sleep(0.05)

    mock_review.review.assert_called_once()


@pytest.mark.asyncio
async def test_turn_11_does_not_trigger_review_again():
    """T3: Turn 11 does not trigger a second review (not a multiple of 10)."""
    mock_review = MagicMock(spec=MemoryReviewStrategy)
    mock_review.review = AsyncMock(return_value=[])
    mock_provider = MagicMock(spec=MemoryProvider)
    mock_provider.upsert = AsyncMock()

    cfg = _make_config(
        memory_review_strategy=mock_review,
        memory_provider=mock_provider,
    )
    agent = _make_agent(cfg)
    session = Session.create()
    agent._current_session = session

    for i in range(11):
        agent._event_bus.emit(_make_turn_complete_event(i))

    await asyncio.sleep(0.05)

    # review called once (at turn 10), not twice
    assert mock_review.review.call_count == 1


@pytest.mark.asyncio
async def test_turn_20_triggers_review_second_time():
    """T4: At turn 20, review is called a second time (total 2)."""
    mock_review = MagicMock(spec=MemoryReviewStrategy)
    mock_review.review = AsyncMock(return_value=[])
    mock_provider = MagicMock(spec=MemoryProvider)
    mock_provider.upsert = AsyncMock()

    cfg = _make_config(
        memory_review_strategy=mock_review,
        memory_provider=mock_provider,
    )
    agent = _make_agent(cfg)
    session = Session.create()
    agent._current_session = session

    for i in range(20):
        agent._event_bus.emit(_make_turn_complete_event(i))

    await asyncio.sleep(0.05)

    assert mock_review.review.call_count == 2


@pytest.mark.asyncio
async def test_no_trigger_when_strategy_is_none():
    """T5: When memory_review_strategy=None, 10 turns do not trigger review."""
    mock_provider = MagicMock(spec=MemoryProvider)
    mock_provider.upsert = AsyncMock()

    cfg = _make_config(
        memory_review_strategy=None,
        memory_provider=mock_provider,
    )
    agent = _make_agent(cfg)
    session = Session.create()
    agent._current_session = session

    for i in range(10):
        agent._event_bus.emit(_make_turn_complete_event(i))

    await asyncio.sleep(0.05)

    # No exception and no upsert called
    mock_provider.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_no_trigger_when_provider_is_none():
    """T6: When memory_provider=None, 10 turns do not trigger upsert (and no exception)."""
    mock_review = MagicMock(spec=MemoryReviewStrategy)
    mock_review.review = AsyncMock(return_value=[_make_entry()])

    cfg = _make_config(
        memory_review_strategy=mock_review,
        memory_provider=None,
    )
    agent = _make_agent(cfg)
    session = Session.create()
    agent._current_session = session

    for i in range(10):
        agent._event_bus.emit(_make_turn_complete_event(i))

    await asyncio.sleep(0.05)

    # review is not called when provider is None
    mock_review.review.assert_not_called()


@pytest.mark.asyncio
async def test_empty_entries_do_not_call_upsert():
    """T7: When review returns [], provider.upsert is NOT called."""
    mock_review = MagicMock(spec=MemoryReviewStrategy)
    mock_review.review = AsyncMock(return_value=[])
    mock_provider = MagicMock(spec=MemoryProvider)
    mock_provider.upsert = AsyncMock()

    cfg = _make_config(
        memory_review_strategy=mock_review,
        memory_provider=mock_provider,
    )
    agent = _make_agent(cfg)
    session = Session.create()
    agent._current_session = session

    for i in range(10):
        agent._event_bus.emit(_make_turn_complete_event(i))

    await asyncio.sleep(0.05)

    mock_review.review.assert_called_once()
    mock_provider.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_entries_returned_by_review_are_upserted():
    """T8: When review returns 2 entries, provider.upsert is called once with those entries."""
    entry1 = _make_entry()
    entry2 = _make_entry()

    mock_review = MagicMock(spec=MemoryReviewStrategy)
    mock_review.review = AsyncMock(return_value=[entry1, entry2])
    mock_provider = MagicMock(spec=MemoryProvider)
    mock_provider.upsert = AsyncMock()

    cfg = _make_config(
        memory_review_strategy=mock_review,
        memory_provider=mock_provider,
    )
    agent = _make_agent(cfg)
    session = Session.create()
    agent._current_session = session

    for i in range(10):
        agent._event_bus.emit(_make_turn_complete_event(i))

    await asyncio.sleep(0.05)

    mock_provider.upsert.assert_called_once_with([entry1, entry2])


@pytest.mark.asyncio
async def test_no_current_session_is_noop():
    """T9: When no current session is set, TurnCompleteEvent is a no-op."""
    mock_review = MagicMock(spec=MemoryReviewStrategy)
    mock_review.review = AsyncMock(return_value=[_make_entry()])
    mock_provider = MagicMock(spec=MemoryProvider)
    mock_provider.upsert = AsyncMock()

    cfg = _make_config(
        memory_review_strategy=mock_review,
        memory_provider=mock_provider,
    )
    agent = _make_agent(cfg)
    # No _current_session set

    for i in range(10):
        agent._event_bus.emit(_make_turn_complete_event(i))

    await asyncio.sleep(0.05)

    mock_review.review.assert_not_called()
    mock_provider.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_review_exception_is_soft_failure():
    """T10: If review raises, no exception propagates (soft failure)."""
    mock_review = MagicMock(spec=MemoryReviewStrategy)
    mock_review.review = AsyncMock(side_effect=RuntimeError("LLM exploded"))
    mock_provider = MagicMock(spec=MemoryProvider)
    mock_provider.upsert = AsyncMock()

    cfg = _make_config(
        memory_review_strategy=mock_review,
        memory_provider=mock_provider,
    )
    agent = _make_agent(cfg)
    session = Session.create()
    agent._current_session = session

    for i in range(10):
        agent._event_bus.emit(_make_turn_complete_event(i))

    # Must not raise
    await asyncio.sleep(0.05)
    mock_provider.upsert.assert_not_called()
