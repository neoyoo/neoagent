# tests/v2/e2e/test_update_wm.py
"""Phase 8 Batch 1 — Task 8.6: update_working_memory E2E full round-trip.

Tests the complete chain:
  NeoAgent → UpdateWorkingMemoryTool → WorkingMemory mutation
  → turn-end WM snapshot to wm_store → WorkingMemoryUpdatedEvent emission

Spec refs: § 15.9 (UpdateWorkingMemoryTool), § 2.3a (WM snapshot)
Contract refs: C2 (WorkingMemoryStore), C5 (hook), C6 (WM protocol)

Test setup:
  - MockProvider returns scripted Response objects (tool_use → end_turn)
  - UpdateWorkingMemoryTool manually registered (session_state_ref closure)
  - session.state._current_wm set to a fresh WorkingMemory (framework_init sim)
  - InMemoryWorkingMemoryStore used for snapshot verification
  - agent._loop.run(session=session) used directly to avoid agent.run()'s
    duplicate-messages guard (session already populated before call)
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from neoagent.agent import NeoAgent
from neoagent.config import NeoAgentConfig
from neoagent.core.types import Message, TextBlock, ToolResult, ToolUseBlock
from neoagent.events import WorkingMemoryUpdatedEvent
from neoagent.providers.base import Provider, Response
from neoagent.session import Session
from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryTool
from neoagent.v2.schema import WorkingMemory
from neoagent.v2.stores import InMemoryWorkingMemoryStore


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_wm(session_id: str = "test-session") -> WorkingMemory:
    """Create a fresh WorkingMemory for testing (framework_init simulation)."""
    return WorkingMemory(
        session_id=session_id,
        version=1,
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


def _make_tool_use_response(
    tool_name: str,
    tool_input: dict,
    call_id: str = "call_001",
) -> Response:
    """Return a provider response that calls the given tool with given input."""
    return Response(
        content=[
            ToolUseBlock(id=call_id, name=tool_name, input=tool_input),
        ],
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=5,
    )


def _make_end_turn_response(text: str = "Turn complete.") -> Response:
    """Return a provider response that ends the turn."""
    return Response(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        input_tokens=5,
        output_tokens=3,
    )


class MockProvider(Provider):
    """Minimal mock provider returning pre-configured responses in sequence."""

    def __init__(self, responses: list[Response]) -> None:
        self._responses = iter(responses)
        self.model = "mock-model"

    async def create(self, system: str, messages: list, tools: list, **kwargs) -> Response:
        return next(self._responses)

    def get_context_window(self) -> int:
        return 200_000


def _make_agent(provider: Provider, wm_store: InMemoryWorkingMemoryStore) -> NeoAgent:
    """Construct NeoAgent with mock provider and explicit wm_store."""
    cfg = NeoAgentConfig(
        api_key="test-key",
        model="mock-model",
        auto_approve_tools=True,
        wm_store=wm_store,
        enable_source_wrap=False,            # not testing source_wrap here
        enable_security_prompt_blocks=False,  # keep prompts minimal
    )
    with patch("neoagent.agent._create_provider", return_value=provider):
        agent = NeoAgent(cfg)
    return agent


def _setup_session_with_wm(
    agent: NeoAgent,
    session_id: str = "test-session",
    initial_message: str = "do it",
) -> Session:
    """Create a session, pre-populate one user message, attach WM + UpdateWorkingMemoryTool."""
    session = agent.new_session(session_id=session_id)
    # Simulate framework_init: set _current_wm
    wm = _make_wm(session_id=session.id)
    session.state._current_wm = wm

    # Manually register UpdateWorkingMemoryTool with session_state_ref
    update_tool = UpdateWorkingMemoryTool(
        session_state_ref=lambda: session.state
    )
    agent.register_tool(update_tool)

    # Populate the initial user message directly on session.messages
    session.messages.append(Message(role="user", content=initial_message))

    return session


async def _run_session(agent: NeoAgent, session: Session):
    """Run the agent loop directly (bypasses agent.run()'s duplicate-messages guard)."""
    agent._current_session = session
    try:
        result = await agent._loop.run(session=session)
    finally:
        agent._current_session = None
    return result


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestUpdateWmAppend:
    """Task 8.6 — append list field succeeds."""

    @pytest.mark.asyncio
    async def test_8_6_1_append_key_decisions(self):
        """8.6.1: LLM calls update_wm(field='key_decisions', op='append', value='d01: 曼谷 3 晚')
        → WM.key_decisions contains 'd01: 曼谷 3 晚', tool_result is success."""
        wm_store = InMemoryWorkingMemoryStore()
        responses = [
            _make_tool_use_response(
                "update_working_memory",
                {"field": "key_decisions", "op": "append", "value": "d01: 曼谷 3 晚"},
                call_id="call_001",
            ),
            _make_end_turn_response(),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, wm_store)
        session = _setup_session_with_wm(agent)

        result = await _run_session(agent, session)

        # Verify tool result is success
        assert len(result.turns) >= 1
        tool_turn = result.turns[0]
        assert len(tool_turn.tool_results) == 1
        tr = tool_turn.tool_results[0]
        assert tr.is_error is False
        assert "WM updated" in tr.output

        # Verify WM mutation
        wm = session.state._current_wm
        assert wm is not None
        assert "d01: 曼谷 3 晚" in wm.key_decisions



class TestUpdateWmRemove:
    """Task 8.6 — remove by item_id succeeds."""

    @pytest.mark.asyncio
    async def test_8_6_3_remove_by_item_id(self):
        """8.6.3: append d01 → remove d01 → WM.key_decisions empty."""
        wm_store = InMemoryWorkingMemoryStore()
        responses = [
            # Turn 1: append
            _make_tool_use_response(
                "update_working_memory",
                {"field": "key_decisions", "op": "append", "value": "d01: 曼谷 3 晚"},
                call_id="call_001",
            ),
            # Turn 2: remove
            _make_tool_use_response(
                "update_working_memory",
                {"field": "key_decisions", "op": "remove", "item_id": "d01"},
                call_id="call_002",
            ),
            _make_end_turn_response(),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, wm_store)
        session = _setup_session_with_wm(agent)

        result = await _run_session(agent, session)

        # Both tool results should succeed
        tool_results = [tr for turn in result.turns for tr in turn.tool_results]
        assert len(tool_results) == 2
        assert tool_results[0].is_error is False
        assert tool_results[1].is_error is False

        # WM.key_decisions should be empty after remove
        wm = session.state._current_wm
        matching = [item for item in wm.key_decisions if item.startswith("d01: ")]
        assert len(matching) == 0


class TestUpdateWmScalarSet:
    """Task 8.6 — scalar field set succeeds."""

    @pytest.mark.asyncio
    async def test_8_6_4_scalar_field_set_progress(self):
        """8.6.4: update_wm(field='progress', op='set', value='50%') → WM.progress='50%'."""
        wm_store = InMemoryWorkingMemoryStore()
        responses = [
            _make_tool_use_response(
                "update_working_memory",
                {"field": "progress", "op": "set", "value": "50%"},
                call_id="call_001",
            ),
            _make_end_turn_response(),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, wm_store)
        session = _setup_session_with_wm(agent)

        result = await _run_session(agent, session)

        tr = result.turns[0].tool_results[0]
        assert tr.is_error is False
        assert session.state._current_wm.progress == "50%"


class TestUpdateWmRejections:
    """Task 8.6 — various invalid operations are rejected."""

    @pytest.mark.asyncio
    async def test_8_6_5_scalar_field_append_rejected(self):
        """8.6.5: update_wm(field='progress', op='append') → error (scalar only supports set)."""
        wm_store = InMemoryWorkingMemoryStore()
        responses = [
            _make_tool_use_response(
                "update_working_memory",
                {"field": "progress", "op": "append", "value": "something"},
                call_id="call_001",
            ),
            _make_end_turn_response(),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, wm_store)
        session = _setup_session_with_wm(agent)

        result = await _run_session(agent, session)

        tr = result.turns[0].tool_results[0]
        assert tr.is_error is True
        assert "op=set" in tr.output or "only supports" in tr.output

    @pytest.mark.asyncio
    async def test_8_6_6_list_field_value_no_prefix_rejected(self):
        """8.6.6: update_wm(field='key_decisions', op='append', value='no prefix') → error."""
        wm_store = InMemoryWorkingMemoryStore()
        responses = [
            _make_tool_use_response(
                "update_working_memory",
                {"field": "key_decisions", "op": "append", "value": "no prefix here"},
                call_id="call_001",
            ),
            _make_end_turn_response(),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, wm_store)
        session = _setup_session_with_wm(agent)

        result = await _run_session(agent, session)

        tr = result.turns[0].tool_results[0]
        assert tr.is_error is True
        assert "prefix" in tr.output.lower() or "d" in tr.output.lower()


class TestUpdateWmSnapshot:
    """Task 8.6 — WM snapshot to wm_store at turn end."""

    @pytest.mark.asyncio
    async def test_8_6_7_wm_snapshot_to_store(self):
        """8.6.7: After turn ends, wm_store.get_current(session_id) returns updated WM."""
        wm_store = InMemoryWorkingMemoryStore()
        responses = [
            _make_tool_use_response(
                "update_working_memory",
                {"field": "key_decisions", "op": "append", "value": "d01: 曼谷 3 晚"},
                call_id="call_001",
            ),
            _make_end_turn_response(),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, wm_store)
        session = _setup_session_with_wm(agent)
        session_id = session.id

        await _run_session(agent, session)

        # Verify wm_store.get_current returns the updated WM
        saved_wm = await wm_store.get_current(session_id)
        assert saved_wm is not None
        assert "d01: 曼谷 3 晚" in saved_wm.key_decisions

    @pytest.mark.asyncio
    async def test_8_6_7_wm_snapshot_saved_exactly_once(self):
        """8.6.7: WM snapshot saved exactly once (at end_turn, not during tool_use)."""
        wm_store = InMemoryWorkingMemoryStore()
        responses = [
            _make_tool_use_response(
                "update_working_memory",
                {"field": "key_decisions", "op": "append", "value": "d01: 曼谷 3 晚"},
                call_id="call_001",
            ),
            _make_end_turn_response(),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, wm_store)
        session = _setup_session_with_wm(agent)
        session_id = session.id

        # Intercept wm_store.save calls to count them
        save_calls: list = []
        original_save = wm_store.save

        async def tracking_save(sid: str, wm: WorkingMemory) -> None:
            save_calls.append((sid, wm))
            await original_save(sid, wm)

        wm_store.save = tracking_save

        await _run_session(agent, session)

        # Should be saved exactly once: at end_turn
        assert len(save_calls) == 1
        assert save_calls[0][0] == session_id


class TestUpdateWmEvent:
    """Task 8.6 — WorkingMemoryUpdatedEvent emitted at turn end."""

    @pytest.mark.asyncio
    async def test_8_6_8_wm_updated_event_emitted(self):
        """8.6.8: WorkingMemoryUpdatedEvent is emitted at turn end with correct payload."""
        wm_store = InMemoryWorkingMemoryStore()
        responses = [
            _make_tool_use_response(
                "update_working_memory",
                {"field": "key_decisions", "op": "append", "value": "d01: 曼谷 3 晚"},
                call_id="call_001",
            ),
            _make_end_turn_response(),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, wm_store)
        session = _setup_session_with_wm(agent)
        session_id = session.id

        # Subscribe to WorkingMemoryUpdatedEvent before running
        received_events: list[WorkingMemoryUpdatedEvent] = []
        agent.event_bus.subscribe(WorkingMemoryUpdatedEvent, received_events.append)

        await _run_session(agent, session)

        # Event should have been emitted exactly once
        assert len(received_events) == 1
        evt = received_events[0]
        assert evt.session_id == session_id
        assert isinstance(evt.wm_json, dict)
        assert evt.updated_by in ("llm_tool", "framework_init")

    @pytest.mark.asyncio
    async def test_8_6_8_wm_updated_event_payload_includes_changes(self):
        """8.6.8: WorkingMemoryUpdatedEvent.wm_json reflects the mutation."""
        wm_store = InMemoryWorkingMemoryStore()
        responses = [
            _make_tool_use_response(
                "update_working_memory",
                {"field": "progress", "op": "set", "value": "75%"},
                call_id="call_001",
            ),
            _make_end_turn_response(),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, wm_store)
        session = _setup_session_with_wm(agent)

        received_events: list[WorkingMemoryUpdatedEvent] = []
        agent.event_bus.subscribe(WorkingMemoryUpdatedEvent, received_events.append)

        await _run_session(agent, session)

        assert len(received_events) == 1
        evt = received_events[0]
        assert evt.wm_json.get("progress") == "75%"


class TestUpdateWmFullRoundTrip:
    """Task 8.6 — multi-turn full round-trip: append → remove → end."""

    @pytest.mark.asyncio
    async def test_8_6_full_round_trip(self):
        """Full scenario: append → remove → snapshot + event.

        Turn 1: append d01: 曼谷 3 晚 → success
        Turn 2: remove d01 → success
        Turn 3: end_turn → WM snapshot + event emitted
        """
        wm_store = InMemoryWorkingMemoryStore()
        responses = [
            # Turn 1: append key_decisions
            _make_tool_use_response(
                "update_working_memory",
                {"field": "key_decisions", "op": "append", "value": "d01: 曼谷 3 晚"},
                call_id="call_001",
            ),
            # Turn 2: remove d01
            _make_tool_use_response(
                "update_working_memory",
                {"field": "key_decisions", "op": "remove", "item_id": "d01"},
                call_id="call_002",
            ),
            # Turn 3: end
            _make_end_turn_response("All done."),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, wm_store)
        session = _setup_session_with_wm(agent)
        session_id = session.id

        received_events: list[WorkingMemoryUpdatedEvent] = []
        agent.event_bus.subscribe(WorkingMemoryUpdatedEvent, received_events.append)

        result = await _run_session(agent, session)

        # Collect all tool results
        tool_results = [tr for turn in result.turns for tr in turn.tool_results]
        assert len(tool_results) == 2

        # Turn 1: append succeeded
        assert tool_results[0].is_error is False
        assert "WM updated" in tool_results[0].output

        # Turn 2: remove succeeded
        assert tool_results[1].is_error is False

        # WM state after all turns
        wm = session.state._current_wm
        assert wm is not None
        # d01 should be gone
        assert not any(item.startswith("d01: ") for item in wm.key_decisions)

        # WM snapshot: wm_store has the updated WM
        saved_wm = await wm_store.get_current(session_id)
        assert saved_wm is not None
        assert not any(item.startswith("d01: ") for item in saved_wm.key_decisions)

        # WorkingMemoryUpdatedEvent emitted exactly once (at end_turn)
        assert len(received_events) == 1
        evt = received_events[0]
        assert evt.session_id == session_id
        assert isinstance(evt.wm_json, dict)
