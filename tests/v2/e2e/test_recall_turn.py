# tests/v2/e2e/test_recall_turn.py
"""E2E tests for RecallTurnTool — session.messages-based interface.

RecallTurnTool now fetches full Message content from session.messages by id.
Tests that previously depended on BatchMember.preview or compression-triggered
batch building are skipped with reason "deferred to Phase 2 Batch B".

Spec refs: § 14.2, § 15.4
Contract refs: C6
"""
from __future__ import annotations

import json

import pytest

from neoagent.core.types import Message, TextBlock, ToolUseBlock, ToolResult
from neoagent.tools.builtin.recall_turn import RecallTurnInput, RecallTurnTool


# ── Helpers ───────────────────────────────────────────────────────────────────


class FakeSession:
    """Minimal session stub exposing .messages list."""
    def __init__(self, messages: list[Message] | None = None):
        self.messages = messages if messages is not None else []


def _make_tool(messages: list[Message] | None = None) -> RecallTurnTool:
    session = FakeSession(messages=messages)
    return RecallTurnTool(session_ref=lambda: session)


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

        result = await tool.execute(RecallTurnInput(turn_ids=["m1"]))

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

        result = await tool.execute(RecallTurnInput(turn_ids=["m2"]))

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

        result = await tool.execute(RecallTurnInput(turn_ids=["m1", "m2"]))

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

        result = await tool.execute(RecallTurnInput(turn_ids=["m2", "m1"]))

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

        result = await tool.execute(RecallTurnInput(turn_ids=["m99"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert data["recalled"] == []
        assert "m99" in data["missing"]

    @pytest.mark.asyncio
    async def test_partial_missing_split_correctly(self):
        """recall ['m1', 'm99'] → m1 in recalled, m99 in missing."""
        msgs = [Message(id="m1", role="user", content="m1 content")]
        tool = _make_tool(msgs)

        result = await tool.execute(RecallTurnInput(turn_ids=["m1", "m99"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert len(data["recalled"]) == 1
        assert data["recalled"][0]["id"] == "m1"
        assert "m99" in data["missing"]


class TestRecallEmptyAndEdgeCases:
    """Edge cases: empty list, no-id messages."""

    @pytest.mark.asyncio
    async def test_empty_turn_ids(self):
        """recall [] → {'recalled': []}, is_error=False."""
        msgs = [Message(id="m1", role="user", content="m1 content")]
        tool = _make_tool(msgs)

        result = await tool.execute(RecallTurnInput(turn_ids=[]))

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

        result = await tool.execute(RecallTurnInput(turn_ids=["m2"]))

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
        result = await tool.execute(RecallTurnInput(turn_ids=["m1"]))

        assert result.is_error is True


# ── Skipped: compression-dependent tests ─────────────────────────────────────
# These tests verified "recall from compressed batch after token_threshold
# triggered compression". Compression is not yet integrated in Phase 1 —
# messages trimmed by compression won't appear in session.messages until
# Phase 2 Batch B stores and merges compressed_messages.
# Deferred to Phase 2 Batch B.

@pytest.mark.skip(reason="deferred to Phase 2 Batch B")
class TestRecallAfterCompression:
    """After compression runs, compressed-out messages should still be
    recallable via a merged lookup across session.messages +
    session.compressed_messages. This requires Phase 2 Batch B work."""

    @pytest.mark.asyncio
    async def test_recall_compressed_message_via_batch_member(self):
        """Placeholder: after compression, recall still works for evicted messages."""
        pass

    @pytest.mark.asyncio
    async def test_recall_same_id_in_compressed_and_live_messages(self):
        """Placeholder: id that exists in both lists — live wins or merge rules apply."""
        pass
