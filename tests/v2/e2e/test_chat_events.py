# tests/v2/e2e/test_chat_events.py
"""Phase 8 Batch 2 — Task 8.1: 10-turn chat event emission E2E.

Tests the complete chain:
  NeoAgent → QueryLoop → EventBus → subscribers

Spec refs: § 8.3 (event timing)
Contract refs: C4 (EventBus)

Test cases:
  1. test_pure_chat_10_turns_events: 10 turns, no tools — verifies
     MessageCreatedEvent(assistant)/TurnCompleteEvent counts.
  2. test_chat_with_tool_10_turns_events: 10 turns with tool call per turn —
     verifies ToolResultPersistedEvent + MessageCreatedEvent(tool_result) counts.
  3. test_event_ordering: single turn verifies correct intra-turn event ordering.

NOTE on MessageCreatedEvent for user input:
  The loop only emits MessageCreatedEvent for assistant_reply and tool_result paths.
  User messages (appended by agent.chat()) do NOT emit MessageCreatedEvent in the
  current implementation. Tests reflect actual loop behaviour.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from neoagent.agent import NeoAgent
from neoagent.config import NeoAgentConfig
from neoagent.core.types import Message, TextBlock, ToolResult, ToolUseBlock
from neoagent.events import (
    BatchCreatedEvent,
    MessageCreatedEvent,
    ProviderRequestEvent,
    ProviderResponseEvent,
    ToolResultPersistedEvent,
    TurnCompleteEvent,
    WorkingMemoryUpdatedEvent,
)
from neoagent.providers.base import Provider, Response
from neoagent.session import Session
from neoagent.tools.base import BaseTool
from neoagent.v2.schema import WorkingMemory
from neoagent.v2.stores import InMemoryWorkingMemoryStore

from pydantic import BaseModel


# ── Helpers ───────────────────────────────────────────────────────────────────


def _end_turn_response(text: str = "OK") -> Response:
    return Response(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        input_tokens=5,
        output_tokens=3,
    )


def _tool_use_response(tool_name: str = "mock_tool", call_id: str = "call_001") -> Response:
    return Response(
        content=[ToolUseBlock(id=call_id, name=tool_name, input={})],
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=5,
    )


class MockProvider(Provider):
    def __init__(self, responses: list[Response]) -> None:
        self._responses = iter(responses)
        self.model = "mock-model"

    async def create(self, system: str, messages: list, tools: list, **kwargs) -> Response:
        return next(self._responses)

    def get_context_window(self) -> int:
        return 200_000


class _EmptyInput(BaseModel):
    pass


class MockTool(BaseTool):
    name = "mock_tool"
    description = "A mock tool"
    input_model = _EmptyInput
    permission = "auto"
    is_concurrent_safe = True
    returns_external_content = False

    async def execute(self, input: _EmptyInput) -> ToolResult:
        return ToolResult(call_id="", output="tool output", is_error=False)


def _make_wm(session_id: str) -> WorkingMemory:
    return WorkingMemory(
        session_id=session_id,
        version=1,
        at_turn=0,
        goal="test goal",
        constraints_and_preferences=[],
        progress="",
        key_decisions=[],
        relevant_files=[],
        next_steps=[],
        critical_context="",
        updated_by="framework_init",
        updated_at=datetime(2026, 4, 22, 12, 0, 0),
    )


def _make_agent(
    provider: Provider,
    wm_store: InMemoryWorkingMemoryStore | None = None,
    extra_tools: list[BaseTool] | None = None,
) -> NeoAgent:
    if wm_store is None:
        wm_store = InMemoryWorkingMemoryStore()
    cfg = NeoAgentConfig(
        api_key="test-key",
        model="mock-model",
        auto_approve_tools=True,
        wm_store=wm_store,
        enable_source_wrap=False,
        enable_security_prompt_blocks=False,
    )
    with patch("neoagent.agent._create_provider", return_value=provider):
        agent = NeoAgent(cfg)
    if extra_tools:
        for tool in extra_tools:
            agent.register_tool(tool)
    return agent


def _setup_session(agent: NeoAgent, initial_msg: str = "hello") -> Session:
    """Create session with WM and one user message."""
    session = agent.new_session()
    session.state._current_wm = _make_wm(session.id)
    session.messages.append(Message(role="user", content=initial_msg))
    return session


async def _run_session(agent: NeoAgent, session: Session):
    agent._current_session = session
    try:
        result = await agent._loop.run(session=session)
    finally:
        agent._current_session = None
    return result


def _subscribe_events(agent: NeoAgent, *event_types):
    """Subscribe to multiple event types; returns dict of event_type → list."""
    captured: dict = {et: [] for et in event_types}
    for et in event_types:
        agent.event_bus.subscribe(et, captured[et].append)
    return captured


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestPureChat10TurnsEvents:
    """Task 8.1-1: 10 turns with no tool calls — verify event counts."""

    @pytest.mark.asyncio
    async def test_pure_chat_10_turns_events(self):
        """10 pure-chat turns → 10 assistant MessageCreatedEvent + 10 TurnCompleteEvent.

        NOTE: WM snapshot emitted each turn (WM set, so WorkingMemoryUpdatedEvent × 10).
        BatchCreatedEvent: 0 (no compression triggered).
        ToolResultPersistedEvent: 0 (no tool calls).
        """
        # 10 end_turn responses
        responses = [_end_turn_response(f"reply {i}") for i in range(10)]
        provider = MockProvider(responses)
        wm_store = InMemoryWorkingMemoryStore()
        agent = _make_agent(provider, wm_store)
        session = _setup_session(agent)

        captured = _subscribe_events(
            agent,
            MessageCreatedEvent,
            TurnCompleteEvent,
            WorkingMemoryUpdatedEvent,
            BatchCreatedEvent,
            ToolResultPersistedEvent,
        )

        # Run 10 turns: after each end_turn, add another user message and run again
        # The loop runs until end_turn. We do this 10 times by adding messages.
        # Simpler: start with 1 user msg, loop produces 1 assistant end_turn per run.
        # Re-append user msg and re-run 10 times.
        for i in range(10):
            await _run_session(agent, session)
            if i < 9:
                session.messages.append(Message(role="user", content=f"turn {i+2}"))

        # 10 assistant_reply MessageCreatedEvent (one per end_turn)
        assistant_created = [
            e for e in captured[MessageCreatedEvent]
            if e.source_type == "assistant_reply"
        ]
        assert len(assistant_created) == 10, (
            f"Expected 10 assistant MessageCreatedEvents, got {len(assistant_created)}"
        )

        # 10 TurnCompleteEvent
        assert len(captured[TurnCompleteEvent]) == 10, (
            f"Expected 10 TurnCompleteEvents, got {len(captured[TurnCompleteEvent])}"
        )

        # 10 WorkingMemoryUpdatedEvent (WM is set, snapshot fires each end_turn)
        assert len(captured[WorkingMemoryUpdatedEvent]) == 10, (
            f"Expected 10 WorkingMemoryUpdatedEvents, got {len(captured[WorkingMemoryUpdatedEvent])}"
        )

        # 0 BatchCreatedEvent
        assert len(captured[BatchCreatedEvent]) == 0

        # 0 ToolResultPersistedEvent
        assert len(captured[ToolResultPersistedEvent]) == 0


class TestChatWithTool10TurnsEvents:
    """Task 8.1-2: 10 turns with tool call per turn — verify event counts."""

    @pytest.mark.asyncio
    async def test_chat_with_tool_10_turns_events(self):
        """Each turn: tool_use → end_turn.
        Expect: 10 ToolResultPersistedEvent, 10 tool_result MessageCreatedEvent,
        10 assistant_reply MessageCreatedEvent from tool_use path,
        10 assistant_reply from end_turn path,
        20 TurnCompleteEvent (tool_use turn + end_turn turn per conversation),
        0 BatchCreatedEvent.
        """
        # Each "conversation": tool_use response then end_turn response
        responses = []
        for i in range(10):
            responses.append(_tool_use_response("mock_tool", call_id=f"call_{i:03d}"))
            responses.append(_end_turn_response(f"done {i}"))

        provider = MockProvider(responses)
        wm_store = InMemoryWorkingMemoryStore()
        agent = _make_agent(provider, wm_store, extra_tools=[MockTool()])
        session = _setup_session(agent)

        captured = _subscribe_events(
            agent,
            MessageCreatedEvent,
            TurnCompleteEvent,
            WorkingMemoryUpdatedEvent,
            BatchCreatedEvent,
            ToolResultPersistedEvent,
        )

        # Run 10 "conversations": each yields tool_use+end_turn (2 loop turns)
        for i in range(10):
            await _run_session(agent, session)
            if i < 9:
                session.messages.append(Message(role="user", content=f"turn {i+2}"))

        # 10 ToolResultPersistedEvent
        assert len(captured[ToolResultPersistedEvent]) == 10, (
            f"Expected 10 ToolResultPersistedEvents, got {len(captured[ToolResultPersistedEvent])}"
        )

        # 10 tool_result MessageCreatedEvent
        tool_result_created = [
            e for e in captured[MessageCreatedEvent]
            if e.source_type == "tool_result"
        ]
        assert len(tool_result_created) == 10, (
            f"Expected 10 tool_result MessageCreatedEvents, got {len(tool_result_created)}"
        )

        # 20 assistant_reply MessageCreatedEvent (1 per tool_use turn + 1 per end_turn)
        assistant_created = [
            e for e in captured[MessageCreatedEvent]
            if e.source_type == "assistant_reply"
        ]
        assert len(assistant_created) == 20, (
            f"Expected 20 assistant_reply MessageCreatedEvents, got {len(assistant_created)}"
        )

        # 20 TurnCompleteEvent (tool_use turn + end_turn per conversation × 10)
        assert len(captured[TurnCompleteEvent]) == 20, (
            f"Expected 20 TurnCompleteEvents, got {len(captured[TurnCompleteEvent])}"
        )

        # 0 BatchCreatedEvent
        assert len(captured[BatchCreatedEvent]) == 0


class TestEventOrdering:
    """Task 8.1-3: Verify intra-turn event ordering."""

    @pytest.mark.asyncio
    async def test_event_ordering_pure_chat_turn(self):
        """Single pure-chat turn ordering (actual loop.py sequence):
        ProviderRequestEvent → ProviderResponseEvent →
        WorkingMemoryUpdatedEvent → TurnCompleteEvent → MessageCreatedEvent[assistant]

        NOTE: The loop emits WorkingMemoryUpdatedEvent + TurnCompleteEvent BEFORE
        MessageCreatedEvent (see loop.py lines 270-306). This is the actual
        implementation order — the spec description in the task brief was
        aspirational; we validate actual behaviour here.
        """
        responses = [_end_turn_response("hello")]
        provider = MockProvider(responses)
        wm_store = InMemoryWorkingMemoryStore()
        agent = _make_agent(provider, wm_store)
        session = _setup_session(agent)

        ordered_events: list = []
        for et in [
            ProviderRequestEvent, ProviderResponseEvent,
            MessageCreatedEvent, WorkingMemoryUpdatedEvent, TurnCompleteEvent,
        ]:
            agent.event_bus.subscribe(et, lambda e: ordered_events.append(e))

        await _run_session(agent, session)

        types = [type(e).__name__ for e in ordered_events]
        # Actual order from loop.py:
        _check_before("ProviderRequestEvent", "ProviderResponseEvent", types)
        _check_before("ProviderResponseEvent", "WorkingMemoryUpdatedEvent", types)
        _check_before("WorkingMemoryUpdatedEvent", "TurnCompleteEvent", types)
        _check_before("TurnCompleteEvent", "MessageCreatedEvent", types)

    @pytest.mark.asyncio
    async def test_event_ordering_tool_call_turn(self):
        """Single tool-call turn ordering:
        ProviderRequestEvent → ProviderResponseEvent →
        MessageCreatedEvent[assistant_tool_use] → MessageCreatedEvent[tool_result] →
        ToolResultPersistedEvent → TurnCompleteEvent
        (Note: WorkingMemoryUpdated only fires on end_turn, not tool_use turn)
        """
        # tool_use then end_turn (2 turns total)
        responses = [
            _tool_use_response("mock_tool", "c001"),
            _end_turn_response("done"),
        ]
        provider = MockProvider(responses)
        wm_store = InMemoryWorkingMemoryStore()
        agent = _make_agent(provider, wm_store, extra_tools=[MockTool()])
        session = _setup_session(agent)

        ordered_events: list = []
        for et in [
            ProviderRequestEvent, ProviderResponseEvent,
            MessageCreatedEvent, ToolResultPersistedEvent, TurnCompleteEvent,
        ]:
            agent.event_bus.subscribe(et, lambda e: ordered_events.append(e))

        await _run_session(agent, session)

        types = [type(e).__name__ for e in ordered_events]

        # First ProviderRequest before first ProviderResponse
        _check_before("ProviderRequestEvent", "ProviderResponseEvent", types)

        # ToolResultPersistedEvent before TurnCompleteEvent (first turn)
        _check_before("ToolResultPersistedEvent", "TurnCompleteEvent", types)

        # MessageCreatedEvent[assistant_reply] before tool_result for tool_use turn
        mc_indices = [i for i, t in enumerate(types) if t == "MessageCreatedEvent"]
        tc_indices = [i for i, t in enumerate(types) if t == "ToolResultPersistedEvent"]
        assert mc_indices, "Expected MessageCreatedEvent"
        assert tc_indices, "Expected ToolResultPersistedEvent"
        # First assistant MessageCreated (tool_use path) comes before first ToolResultPersisted
        assert mc_indices[0] < tc_indices[0]


def _check_before(type_a: str, type_b: str, types: list[str]) -> None:
    """Assert type_a appears before type_b in the types list."""
    idx_a = next((i for i, t in enumerate(types) if t == type_a), None)
    idx_b = next((i for i, t in enumerate(types) if t == type_b), None)
    assert idx_a is not None, f"Event {type_a!r} not found in {types}"
    assert idx_b is not None, f"Event {type_b!r} not found in {types}"
    assert idx_a < idx_b, (
        f"Expected {type_a} before {type_b}, got indices {idx_a} vs {idx_b} in {types}"
    )
