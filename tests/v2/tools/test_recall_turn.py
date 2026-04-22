# tests/v2/tools/test_recall_turn.py
"""TDD tests for Task 5.2 — RecallTurnTool.

Spec refs: § 14.2, § 15.4
Contract refs: C6
"""
from __future__ import annotations

import json
import pytest
from datetime import datetime

from neoagent.v2.schema import Batch, BatchMember


def _make_batch(batch_id: str, members: list[BatchMember]) -> Batch:
    now = datetime(2026, 4, 22, 12, 0, 0)
    return Batch(
        session_id="s1",
        batch_id=batch_id,
        turns_from=0,
        turns_to=5,
        time_from=now,
        time_to=now,
        summary="test batch",
        members=members,
        trigger="token_threshold",
        created_at=now,
    )


class FakeState:
    """Minimal session_state stub with optional batches."""
    def __init__(self, batches=None):
        self.batches = batches if batches is not None else []


def _make_tool(batches=None):
    from neoagent.tools.builtin.recall_turn import RecallTurnTool
    state = FakeState(batches=batches)
    return RecallTurnTool(session_state_ref=lambda: state), state


# ── 1. turn_ids all in recoverable → returns preview ─────────────────────────

class TestRecallSuccess:
    @pytest.mark.asyncio
    async def test_single_turn_id_returns_preview(self):
        members = [
            BatchMember(id="m1", role="user", preview="user asked about X"),
            BatchMember(id="m2", role="assistant", preview="assistant answered Y"),
        ]
        batch = _make_batch("cm_1", members)
        tool, _ = _make_tool(batches=[batch])

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        inp = RecallTurnInput(turn_ids=["m1"])
        result = await tool.execute(inp)

        assert result.is_error is False
        data = json.loads(result.output)
        assert len(data["recalled"]) == 1
        assert data["recalled"][0]["id"] == "m1"
        assert data["recalled"][0]["preview"] == "user asked about X"
        assert data["recalled"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_multiple_turn_ids_returned(self):
        members = [
            BatchMember(id="m3", role="user", preview="msg 3"),
            BatchMember(id="m4", role="assistant", preview="msg 4"),
        ]
        batch = _make_batch("cm_1", members)
        tool, _ = _make_tool(batches=[batch])

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        inp = RecallTurnInput(turn_ids=["m3", "m4"])
        result = await tool.execute(inp)

        assert result.is_error is False
        data = json.loads(result.output)
        assert len(data["recalled"]) == 2
        ids = {item["id"] for item in data["recalled"]}
        assert ids == {"m3", "m4"}


# ── 2. turn_ids partially not in recoverable → error ─────────────────────────

class TestPartialMissing:
    @pytest.mark.asyncio
    async def test_partial_missing_returns_error(self):
        members = [BatchMember(id="m1", role="user", preview="msg 1")]
        batch = _make_batch("cm_1", members)
        tool, _ = _make_tool(batches=[batch])

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        inp = RecallTurnInput(turn_ids=["m1", "m99"])
        result = await tool.execute(inp)

        assert result.is_error is True
        assert "m99" in result.output


# ── 3. turn_ids completely not in recoverable → error ─────────────────────────

class TestAllMissing:
    @pytest.mark.asyncio
    async def test_all_missing_returns_error(self):
        members = [BatchMember(id="m1", role="user", preview="msg 1")]
        batch = _make_batch("cm_1", members)
        tool, _ = _make_tool(batches=[batch])

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        inp = RecallTurnInput(turn_ids=["x1", "x2"])
        result = await tool.execute(inp)

        assert result.is_error is True
        assert "x1" in result.output or "x2" in result.output


# ── 4. session has no batches → error (no recoverable at all) ─────────────────

class TestNoBatches:
    @pytest.mark.asyncio
    async def test_no_batches_returns_error(self):
        tool, _ = _make_tool(batches=[])

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        inp = RecallTurnInput(turn_ids=["m1"])
        result = await tool.execute(inp)

        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_state_without_batches_attr_returns_error(self):
        """If state has no batches attribute at all, treat as no recoverable."""
        from neoagent.tools.builtin.recall_turn import RecallTurnTool

        class StateNoBatches:
            pass  # no batches attribute

        tool = RecallTurnTool(session_state_ref=lambda: StateNoBatches())
        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        inp = RecallTurnInput(turn_ids=["m1"])
        result = await tool.execute(inp)

        assert result.is_error is True


# ── 5. empty turn_ids → returns empty recalled (not error) ────────────────────

class TestEmptyTurnIds:
    @pytest.mark.asyncio
    async def test_empty_turn_ids_returns_empty_list(self):
        members = [BatchMember(id="m1", role="user", preview="msg 1")]
        batch = _make_batch("cm_1", members)
        tool, _ = _make_tool(batches=[batch])

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        inp = RecallTurnInput(turn_ids=[])
        result = await tool.execute(inp)

        assert result.is_error is False
        data = json.loads(result.output)
        assert data["recalled"] == []


# ── 6. multiple batches across → correct members returned ─────────────────────

class TestMultipleBatches:
    @pytest.mark.asyncio
    async def test_recall_across_multiple_batches(self):
        batch1 = _make_batch("cm_1", [
            BatchMember(id="m1", role="user", preview="batch1 msg1"),
            BatchMember(id="m2", role="assistant", preview="batch1 msg2"),
        ])
        batch2 = _make_batch("cm_2", [
            BatchMember(id="m5", role="user", preview="batch2 msg5"),
            BatchMember(id="m6", role="assistant", preview="batch2 msg6"),
        ])
        tool, _ = _make_tool(batches=[batch1, batch2])

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        inp = RecallTurnInput(turn_ids=["m2", "m5"])
        result = await tool.execute(inp)

        assert result.is_error is False
        data = json.loads(result.output)
        assert len(data["recalled"]) == 2
        ids = {item["id"] for item in data["recalled"]}
        assert ids == {"m2", "m5"}

        previews = {item["id"]: item["preview"] for item in data["recalled"]}
        assert previews["m2"] == "batch1 msg2"
        assert previews["m5"] == "batch2 msg5"
