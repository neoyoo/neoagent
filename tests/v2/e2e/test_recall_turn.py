# tests/v2/e2e/test_recall_turn.py
"""Phase 8 Batch 2 — Task 8.4: recall_turn E2E.

Tests RecallTurnTool.execute() directly with pre-built session state
containing Batch objects with BatchMember entries.

Spec refs: § 14.2 (recall_turn), § 15.4 (C6 tool contract)
Contract refs: C6 (recall_turn simplification note — returns preview, not full content)

SIMPLIFICATION (Phase 5 carryover):
  The tool returns BatchMember.preview, not full original message content.
  Tests verify current preview-based implementation only.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from neoagent.core.types import ToolResult
from neoagent.session import Session, SessionState
from neoagent.tools.builtin.recall_turn import RecallTurnInput, RecallTurnTool
from neoagent.v2.schema import Batch, BatchMember, WorkingMemory


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_member(mid: str, role: str = "user", preview: str | None = None) -> BatchMember:
    return BatchMember(
        id=mid,
        role=role,
        preview=preview or f"preview of {mid}",
    )


def _make_batch(
    session_id: str,
    batch_id: str,
    members: list[BatchMember],
    summary: str = "batch summary",
) -> Batch:
    now = datetime(2026, 4, 22, 12, 0, 0)
    return Batch(
        session_id=session_id,
        batch_id=batch_id,
        turns_from=0,
        turns_to=5,
        time_from=now,
        time_to=now,
        summary=summary,
        members=members,
        trigger="token_threshold",
        created_at=now,
    )


def _make_state_with_batches(batches: list[Batch]) -> SessionState:
    """Create SessionState with batch list pre-populated."""
    state = SessionState()
    object.__setattr__(state, "batches", list(batches))
    return state


def _make_tool(state: SessionState) -> RecallTurnTool:
    return RecallTurnTool(session_state_ref=lambda: state)


def _execute_recall(tool: RecallTurnTool, turn_ids: list[str]) -> ToolResult:
    """Synchronous wrapper — use with asyncio.run or pytest-asyncio."""
    import asyncio
    inp = RecallTurnInput(turn_ids=turn_ids)
    return asyncio.get_event_loop().run_until_complete(tool.execute(inp))


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRecallSingleMember:
    """Task 8.4-1: recall single member id → output contains its preview."""

    @pytest.mark.asyncio
    async def test_recall_single_member(self):
        """recall ['m1'] → output contains m1 preview, is_error=False."""
        m1 = _make_member("m1", "user", "User asked about Bangkok")
        m2 = _make_member("m2", "assistant", "Assistant replied about Bangkok")
        batch = _make_batch("sess-001", "cm_1", [m1, m2])
        state = _make_state_with_batches([batch])
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=["m1"]))

        assert result.is_error is False
        data = json.loads(result.output)
        recalled = data["recalled"]
        assert len(recalled) == 1
        assert recalled[0]["id"] == "m1"
        assert recalled[0]["role"] == "user"
        assert "Bangkok" in recalled[0]["preview"]


class TestRecallMultipleMembers:
    """Task 8.4-2: recall multiple ids → both returned."""

    @pytest.mark.asyncio
    async def test_recall_multiple_members(self):
        """recall ['m1', 'm2'] → output contains both."""
        m1 = _make_member("m1", "user", "m1 preview")
        m2 = _make_member("m2", "assistant", "m2 preview")
        batch = _make_batch("sess-001", "cm_1", [m1, m2])
        state = _make_state_with_batches([batch])
        tool = _make_tool(state)

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
        """Recall preserves request order: ['m2', 'm1'] → m2 first, then m1."""
        m1 = _make_member("m1", "user", "m1 preview")
        m2 = _make_member("m2", "assistant", "m2 preview")
        batch = _make_batch("sess-001", "cm_1", [m1, m2])
        state = _make_state_with_batches([batch])
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=["m2", "m1"]))

        assert result.is_error is False
        data = json.loads(result.output)
        recalled = data["recalled"]
        assert recalled[0]["id"] == "m2"
        assert recalled[1]["id"] == "m1"


class TestRecallMissingId:
    """Task 8.4-3: recall missing id → is_error=True."""

    @pytest.mark.asyncio
    async def test_recall_missing_id_error(self):
        """recall ['m99'] when m99 not in any batch → is_error=True."""
        m1 = _make_member("m1", "user", "m1 preview")
        batch = _make_batch("sess-001", "cm_1", [m1])
        state = _make_state_with_batches([batch])
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=["m99"]))

        assert result.is_error is True
        assert "m99" in result.output

    @pytest.mark.asyncio
    async def test_error_output_includes_available_ids(self):
        """Error output mentions available recoverable ids."""
        m1 = _make_member("m1")
        batch = _make_batch("sess-001", "cm_1", [m1])
        state = _make_state_with_batches([batch])
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=["m99"]))

        assert result.is_error is True
        # Output should mention which ids are available
        assert "m1" in result.output


class TestRecallPartialMissing:
    """Task 8.4-4: partial missing → is_error=True (all-or-nothing validation)."""

    @pytest.mark.asyncio
    async def test_recall_partial_missing_error(self):
        """recall ['m1', 'm99'] where m99 missing → is_error=True, m99 in error output."""
        m1 = _make_member("m1", "user", "m1 preview")
        batch = _make_batch("sess-001", "cm_1", [m1])
        state = _make_state_with_batches([batch])
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=["m1", "m99"]))

        assert result.is_error is True
        assert "m99" in result.output


class TestRecallNoBatches:
    """Task 8.4-5: no batches in session → recall any id → is_error=True."""

    @pytest.mark.asyncio
    async def test_recall_no_batches_error(self):
        """Session with no batches → recall 'm1' → is_error=True."""
        state = SessionState()  # no batches attribute
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=["m1"]))

        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_recall_empty_batches_list_error(self):
        """Session with empty batches list → recall 'm1' → is_error=True."""
        state = _make_state_with_batches([])
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=["m1"]))

        assert result.is_error is True


class TestRecallEmptyList:
    """Task 8.4-6: recall [] → output empty array, is_error=False."""

    @pytest.mark.asyncio
    async def test_recall_empty_list(self):
        """recall [] → {'recalled': []}, is_error=False."""
        m1 = _make_member("m1")
        batch = _make_batch("sess-001", "cm_1", [m1])
        state = _make_state_with_batches([batch])
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=[]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert data["recalled"] == []

    @pytest.mark.asyncio
    async def test_recall_empty_list_no_batches_still_ok(self):
        """recall [] with no batches → still is_error=False (empty list is trivially satisfied)."""
        state = SessionState()
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=[]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert data["recalled"] == []


class TestRecallAcrossBatches:
    """Task 8.4-7: recall across multiple batches."""

    @pytest.mark.asyncio
    async def test_recall_across_batches(self):
        """Session with 2 batches: batch1=[m1,m2], batch2=[m3,m4].
        recall ['m1','m4'] → both found across batches, is_error=False."""
        m1 = _make_member("m1", "user", "m1 from batch1")
        m2 = _make_member("m2", "assistant", "m2 from batch1")
        m3 = _make_member("m3", "user", "m3 from batch2")
        m4 = _make_member("m4", "assistant", "m4 from batch2")

        batch1 = _make_batch("sess-001", "cm_1", [m1, m2])
        batch2 = _make_batch("sess-001", "cm_2", [m3, m4])
        state = _make_state_with_batches([batch1, batch2])
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=["m1", "m4"]))

        assert result.is_error is False
        data = json.loads(result.output)
        recalled = data["recalled"]
        assert len(recalled) == 2
        ids = [r["id"] for r in recalled]
        assert "m1" in ids
        assert "m4" in ids

    @pytest.mark.asyncio
    async def test_recall_all_members_from_two_batches(self):
        """recall all 4 members from 2 batches → all returned."""
        members_b1 = [_make_member("m1"), _make_member("m2")]
        members_b2 = [_make_member("m3"), _make_member("m4")]
        batch1 = _make_batch("sess-001", "cm_1", members_b1)
        batch2 = _make_batch("sess-001", "cm_2", members_b2)
        state = _make_state_with_batches([batch1, batch2])
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=["m1", "m2", "m3", "m4"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert len(data["recalled"]) == 4

    @pytest.mark.asyncio
    async def test_recall_same_id_later_batch_takes_last(self):
        """If two batches have the same member id (edge case), later batch's entry is used
        (index overwrite). is_error=False, preview from last occurrence."""
        m1_v1 = _make_member("m1", "user", "first occurrence")
        m1_v2 = _make_member("m1", "assistant", "second occurrence")
        batch1 = _make_batch("sess-001", "cm_1", [m1_v1])
        batch2 = _make_batch("sess-001", "cm_2", [m1_v2])
        state = _make_state_with_batches([batch1, batch2])
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=["m1"]))

        assert result.is_error is False
        data = json.loads(result.output)
        recalled = data["recalled"]
        assert len(recalled) == 1
        # Second batch's entry wins (dict overwrite during indexing)
        assert recalled[0]["preview"] == "second occurrence"


class TestRecallPreviewContent:
    """Verify preview content matches BatchMember.preview exactly."""

    @pytest.mark.asyncio
    async def test_preview_content_returned_verbatim(self):
        """RecallTurnTool returns preview verbatim from BatchMember.preview."""
        precise_preview = "User asked: what hotels near Sukhumvit?"
        m1 = _make_member("m1", "user", precise_preview)
        batch = _make_batch("sess-001", "cm_1", [m1])
        state = _make_state_with_batches([batch])
        tool = _make_tool(state)

        result = await tool.execute(RecallTurnInput(turn_ids=["m1"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert data["recalled"][0]["preview"] == precise_preview
