# tests/v2/e2e/test_recall_turn.py
"""E2E tests for RecallTurnTool — session.messages-based interface.

RecallTurnTool now fetches full Message content from session.messages by id.
Phase 2 Batch B4: also fetches from session_state.compressed_messages.

Spec refs: § 14.2, § 15.4
Contract refs: C6
"""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from neoagent.core.compress import ContextCompressor
from neoagent.core.types import Message, TextBlock, ToolUseBlock, ToolResult
from neoagent.events import EventBus
from neoagent.providers.base import Provider, Response
from neoagent.session import Session, SessionState
from neoagent.tools.builtin.recall_turn import RecallTurnInput, RecallTurnTool
from neoagent.v2.abc import CompressionStrategy
from neoagent.v2.schema import (
    Batch,
    BatchMember,
    CompressionContext,
    CompressionDelta,
    WorkingMemory,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


class FakeSession:
    """Minimal session stub exposing .messages list."""
    def __init__(self, messages: list[Message] | None = None):
        self.messages = messages if messages is not None else []


def _make_tool(messages: list[Message] | None = None) -> RecallTurnTool:
    session = FakeSession(messages=messages)
    return RecallTurnTool(session_ref=lambda: session)


class MockProvider(Provider):
    def __init__(self) -> None:
        self.model = "mock-model"

    async def create(self, system: str, messages: list, tools: list, **kwargs) -> Response:
        return Response(
            content=[TextBlock(text="mock")],
            stop_reason="end_turn",
            input_tokens=0,
            output_tokens=0,
        )

    def get_context_window(self) -> int:
        return 200_000


class _MockStrategy(CompressionStrategy):
    """Returns a preset CompressionDelta without calling LLM."""

    def __init__(self, preset_delta: CompressionDelta) -> None:
        self.preset_delta = preset_delta

    async def compress(self, context: CompressionContext) -> CompressionDelta:
        return self.preset_delta


def _make_wm(session_id: str = "sess-e2e") -> WorkingMemory:
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


def _make_session_with_messages(n_user_turns: int = 12) -> tuple[Session, SessionState]:
    """Build a Session + SessionState with n_user_turns of user+assistant pairs."""
    state = SessionState()
    state._current_wm = _make_wm()
    # Pre-seed id_gen counter so generated ids are m1..m(2*n_user_turns)
    state.id_gen._msg_counter = 2 * n_user_turns

    msgs: list[Message] = []
    for i in range(n_user_turns):
        msgs.append(Message(id=f"m{2*i+1}", role="user", content=f"user turn {i}", turn=i))
        msgs.append(Message(id=f"m{2*i+2}", role="assistant", content=f"assistant turn {i}", turn=i))

    session = Session(
        id="sess-e2e",
        messages=msgs,
        state=state,
        created_at=datetime(2026, 4, 22, 12, 0, 0),
        updated_at=datetime(2026, 4, 22, 12, 0, 0),
    )
    return session, state


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRecallSingleMessage:
    """recall single id → returns full content, is_error=False."""

    @pytest.mark.asyncio
    async def test_recall_single_str_content(self):
        """recall ['m1'] → content is original str."""
        msgs = [
            Message(id="m1", role="user", content="User asked about Bangkok hotels"),
            Message(id="m2", role="assistant", content="Here are my recommendations"),
        ]
        tool = _make_tool(msgs)

        result = await tool.execute(RecallTurnInput(msg_ids=["m1"]))

        assert result.is_error is False
        data = json.loads(result.output)
        recalled = data["recalled"]
        assert len(recalled) == 1
        assert recalled[0]["id"] == "m1"
        assert recalled[0]["role"] == "user"
        assert "Bangkok" in recalled[0]["content"]

    @pytest.mark.asyncio
    async def test_recall_single_list_content(self):
        """recall ['m2'] → content is list of block dicts."""
        msgs = [
            Message(id="m2", role="assistant", content=[TextBlock(text="Here are my recommendations")]),
        ]
        tool = _make_tool(msgs)

        result = await tool.execute(RecallTurnInput(msg_ids=["m2"]))

        assert result.is_error is False
        data = json.loads(result.output)
        content = data["recalled"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert "recommendations" in content[0]["text"]


class TestRecallMultipleMessages:
    """recall multiple ids → all returned in request order."""

    @pytest.mark.asyncio
    async def test_recall_multiple_messages(self):
        """recall ['m1', 'm2'] → both returned."""
        msgs = [
            Message(id="m1", role="user", content="m1 content"),
            Message(id="m2", role="assistant", content="m2 content"),
        ]
        tool = _make_tool(msgs)

        result = await tool.execute(RecallTurnInput(msg_ids=["m1", "m2"]))

        assert result.is_error is False
        data = json.loads(result.output)
        recalled = data["recalled"]
        assert len(recalled) == 2
        ids = [r["id"] for r in recalled]
        assert "m1" in ids
        assert "m2" in ids

    @pytest.mark.asyncio
    async def test_recall_order_preserved(self):
        """Request order is preserved: ['m2', 'm1'] → m2 first."""
        msgs = [
            Message(id="m1", role="user", content="m1 content"),
            Message(id="m2", role="assistant", content="m2 content"),
        ]
        tool = _make_tool(msgs)

        result = await tool.execute(RecallTurnInput(msg_ids=["m2", "m1"]))

        assert result.is_error is False
        data = json.loads(result.output)
        recalled = data["recalled"]
        assert recalled[0]["id"] == "m2"
        assert recalled[1]["id"] == "m1"


class TestRecallMissingIds:
    """Missing ids go to 'missing' field, not is_error."""

    @pytest.mark.asyncio
    async def test_all_missing_reported_in_missing_field(self):
        """recall ['m99'] when not in session → is_error=False, id in 'missing'."""
        msgs = [Message(id="m1", role="user", content="m1 content")]
        tool = _make_tool(msgs)

        result = await tool.execute(RecallTurnInput(msg_ids=["m99"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert data["recalled"] == []
        assert "m99" in data["missing"]

    @pytest.mark.asyncio
    async def test_partial_missing_split_correctly(self):
        """recall ['m1', 'm99'] → m1 in recalled, m99 in missing."""
        msgs = [Message(id="m1", role="user", content="m1 content")]
        tool = _make_tool(msgs)

        result = await tool.execute(RecallTurnInput(msg_ids=["m1", "m99"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert len(data["recalled"]) == 1
        assert data["recalled"][0]["id"] == "m1"
        assert "m99" in data["missing"]


class TestRecallEmptyAndEdgeCases:
    """Edge cases: empty list, no-id messages."""

    @pytest.mark.asyncio
    async def test_empty_msg_ids(self):
        """recall [] → {'recalled': []}, is_error=False."""
        msgs = [Message(id="m1", role="user", content="m1 content")]
        tool = _make_tool(msgs)

        result = await tool.execute(RecallTurnInput(msg_ids=[]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert data["recalled"] == []

    @pytest.mark.asyncio
    async def test_message_without_id_not_reachable(self):
        """Messages with id=None are skipped in the index."""
        msgs = [
            Message(id=None, role="user", content="no-id cannot be recalled"),
            Message(id="m2", role="assistant", content="has id"),
        ]
        tool = _make_tool(msgs)

        result = await tool.execute(RecallTurnInput(msg_ids=["m2"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert len(data["recalled"]) == 1
        assert data["recalled"][0]["id"] == "m2"

    @pytest.mark.asyncio
    async def test_no_messages_attr_returns_error(self):
        """session_ref returning object without .messages → is_error=True."""

        class BrokenRef:
            pass

        tool = RecallTurnTool(session_ref=lambda: BrokenRef())
        result = await tool.execute(RecallTurnInput(msg_ids=["m1"]))

        assert result.is_error is True


# ── Phase 2 Batch B4: recall from compressed_messages after compression ───────


class TestRecallAfterCompression:
    """After compression runs, compressed-out messages should still be
    recallable via a merged lookup across session.messages +
    session_state.compressed_messages."""

    @pytest.mark.asyncio
    async def test_recall_compressed_message_via_batch_member(self):
        """After compression moves early messages to compressed_messages, recall
        still returns full original content for a formerly-live message."""
        session, state = _make_session_with_messages(n_user_turns=12)

        # m3 is user turn 1 (index 2 in the message list), which will be in
        # the compressed-out portion (keep_recent_user_turns=5 means ~7+ turns
        # get compressed). Capture its original content before compression.
        original_m3 = next(m for m in session.messages if m.id == "m3")
        original_content = original_m3.content

        # Build a legal delta referencing m3 (among others)
        delta = CompressionDelta(
            batch_members=[
                BatchMember(id="m1", role="user", preview="preview of m1"),
                BatchMember(id="m2", role="assistant", preview="preview of m2"),
                BatchMember(id="m3", role="user", preview="preview of m3"),
            ],
            working_memory_delta=[],
        )
        strategy = _MockStrategy(preset_delta=delta)
        bus = EventBus()
        compressor = ContextCompressor(
            provider=MockProvider(),
            strategy=strategy,
            event_bus=bus,
        )

        # Run compression — this moves the to_compress portion to compressed_messages
        to_keep = await compressor._compress_with_strategy(
            messages=session.messages,
            session_state=state,
            current_turn=12,
        )

        # Compression must have moved some messages
        assert state.compressed_messages, "compression should have populated compressed_messages"
        # m3 must be in compressed_messages now
        compressed_ids = {m.id for m in state.compressed_messages}
        assert "m3" in compressed_ids, "m3 should be in compressed_messages after compression"

        # Now replace session.messages with to_keep (simulating what the loop does)
        session.messages = to_keep or session.messages

        # m3 must NOT be in session.messages anymore
        live_ids = {m.id for m in session.messages}
        assert "m3" not in live_ids, "m3 should have been removed from live messages"

        # Recall via RecallTurnTool — should still find m3 in compressed_messages
        tool = RecallTurnTool(session_ref=lambda: session)
        result = await tool.execute(RecallTurnInput(msg_ids=["m3"]))

        assert result.is_error is False
        data = json.loads(result.output)
        recalled = data["recalled"]
        assert len(recalled) == 1
        assert recalled[0]["id"] == "m3"
        # Full original content returned, not the preview
        assert recalled[0]["content"] == original_content
        assert "missing" not in data or "m3" not in data.get("missing", [])

    @pytest.mark.asyncio
    async def test_recall_same_id_in_compressed_and_live_messages(self):
        """Edge case: same msg_id in both session.messages and
        session_state.compressed_messages (buggy partial trim) — live wins."""
        state = SessionState()
        state._current_wm = _make_wm()

        live_msg = Message(id="m5", role="user", content="live content X")
        compressed_msg = Message(id="m5", role="user", content="compressed content Y")

        # Manually plant the id collision (simulates a buggy state)
        state.compressed_messages = [compressed_msg]

        session = Session(
            id="sess-collision",
            messages=[live_msg],
            state=state,
            created_at=datetime(2026, 4, 22, 12, 0, 0),
            updated_at=datetime(2026, 4, 22, 12, 0, 0),
        )

        tool = RecallTurnTool(session_ref=lambda: session)
        result = await tool.execute(RecallTurnInput(msg_ids=["m5"]))

        assert result.is_error is False
        data = json.loads(result.output)
        recalled = data["recalled"]
        assert len(recalled) == 1
        # Live copy must win
        assert recalled[0]["content"] == "live content X"
