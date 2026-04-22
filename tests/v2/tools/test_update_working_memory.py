# tests/v2/tools/test_update_working_memory.py
"""TDD tests for Task 5.1 — UpdateWorkingMemoryTool.

Spec refs: § 15.9
Contract refs: C6, C1
"""
from __future__ import annotations

import pytest
from datetime import datetime

from neoagent.v2.schema import WorkingMemory


def _make_wm(**overrides) -> WorkingMemory:
    defaults = dict(
        session_id="s1",
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
        updated_at=datetime(2026, 4, 22, 10, 0, 0),
    )
    defaults.update(overrides)
    return WorkingMemory(**defaults)


class FakeState:
    """Minimal session_state stub for testing."""
    def __init__(self, wm: WorkingMemory | None):
        self._current_wm = wm


def _make_tool(wm: WorkingMemory | None):
    from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryTool
    state = FakeState(wm)
    return UpdateWorkingMemoryTool(session_state_ref=lambda: state), state


# ── 1. goal field rejected (immutable) ────────────────────────────────────────

class TestGoalImmutable:
    @pytest.mark.asyncio
    async def test_goal_field_rejected(self):
        """goal is immutable after framework_init — must return error."""
        tool, _ = _make_tool(_make_wm())
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="goal", value="new goal", op="set")
        result = await tool.execute(inp)
        assert result.is_error is True
        assert "immutable" in result.output.lower()


# ── 2. scalar field + op=append rejected ──────────────────────────────────────

class TestScalarFieldOpRestriction:
    @pytest.mark.asyncio
    async def test_progress_append_rejected(self):
        tool, _ = _make_tool(_make_wm())
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="progress", op="append", value="extra")
        result = await tool.execute(inp)
        assert result.is_error is True
        assert "scalar" in result.output.lower() or "set" in result.output.lower()

    @pytest.mark.asyncio
    async def test_critical_context_remove_rejected(self):
        tool, _ = _make_tool(_make_wm())
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="critical_context", op="remove", item_id="c01")
        result = await tool.execute(inp)
        assert result.is_error is True


# ── 3. list field value no prefix rejected ────────────────────────────────────

class TestListFieldPrefixValidation:
    @pytest.mark.asyncio
    async def test_constraints_no_prefix_rejected(self):
        tool, _ = _make_tool(_make_wm())
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="constraints_and_preferences", op="append",
                                       value="no prefix here")
        result = await tool.execute(inp)
        assert result.is_error is True
        assert "prefix" in result.output.lower()

    @pytest.mark.asyncio
    async def test_key_decisions_wrong_prefix_rejected(self):
        """key_decisions needs 'd' prefix, not 'c'."""
        tool, _ = _make_tool(_make_wm())
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="key_decisions", op="append",
                                       value="c01: wrong prefix")
        result = await tool.execute(inp)
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_next_steps_no_number_rejected(self):
        """next_steps needs 'nNN: ' format."""
        tool, _ = _make_tool(_make_wm())
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="next_steps", op="append",
                                       value="n: missing number")
        result = await tool.execute(inp)
        assert result.is_error is True


# ── 4. list field value with correct prefix accepted ──────────────────────────

class TestListFieldPrefixAccepted:
    @pytest.mark.asyncio
    async def test_constraints_correct_prefix_accepted(self):
        tool, state = _make_tool(_make_wm())
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="constraints_and_preferences", op="append",
                                       value="c01: keep short")
        result = await tool.execute(inp)
        assert result.is_error is False
        assert "c01: keep short" in state._current_wm.constraints_and_preferences

    @pytest.mark.asyncio
    async def test_key_decisions_correct_prefix_accepted(self):
        tool, state = _make_tool(_make_wm())
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="key_decisions", op="append",
                                       value="d01: use TDD")
        result = await tool.execute(inp)
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_relevant_files_correct_prefix_accepted(self):
        tool, state = _make_tool(_make_wm())
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="relevant_files", op="append",
                                       value="f01: session.py")
        result = await tool.execute(inp)
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_next_steps_correct_prefix_accepted(self):
        tool, state = _make_tool(_make_wm())
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="next_steps", op="append",
                                       value="n01: write tests")
        result = await tool.execute(inp)
        assert result.is_error is False


# ── 5. op=remove without item_id rejected ────────────────────────────────────

class TestRemoveWithoutItemId:
    @pytest.mark.asyncio
    async def test_remove_without_item_id_rejected(self):
        wm = _make_wm(constraints_and_preferences=["c01: keep short"])
        tool, _ = _make_tool(wm)
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="constraints_and_preferences", op="remove",
                                       item_id=None)
        result = await tool.execute(inp)
        assert result.is_error is True
        assert "item_id" in result.output.lower()


# ── 6. op=remove with item_id removes correct entry ──────────────────────────

class TestRemoveCorrectEntry:
    @pytest.mark.asyncio
    async def test_remove_existing_item(self):
        wm = _make_wm(
            constraints_and_preferences=["c01: keep short", "c02: be concise"]
        )
        tool, state = _make_tool(wm)
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="constraints_and_preferences", op="remove",
                                       item_id="c01")
        result = await tool.execute(inp)
        assert result.is_error is False
        assert state._current_wm.constraints_and_preferences == ["c02: be concise"]


# ── 7. op=remove item_id not found returns error ──────────────────────────────

class TestRemoveItemNotFound:
    @pytest.mark.asyncio
    async def test_remove_nonexistent_item(self):
        wm = _make_wm(constraints_and_preferences=["c01: keep short"])
        tool, _ = _make_tool(wm)
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="constraints_and_preferences", op="remove",
                                       item_id="c99")
        result = await tool.execute(inp)
        assert result.is_error is True
        assert "c99" in result.output


# ── 8. op=set scalar field works ─────────────────────────────────────────────

class TestSetScalarField:
    @pytest.mark.asyncio
    async def test_set_progress(self):
        tool, state = _make_tool(_make_wm(progress="old"))
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="progress", op="set", value="done")
        result = await tool.execute(inp)
        assert result.is_error is False
        assert state._current_wm.progress == "done"

    @pytest.mark.asyncio
    async def test_set_critical_context(self):
        tool, state = _make_tool(_make_wm())
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="critical_context", op="set",
                                       value="system is down")
        result = await tool.execute(inp)
        assert result.is_error is False
        assert state._current_wm.critical_context == "system is down"


# ── 9. op=append list field works ─────────────────────────────────────────────

class TestAppendListField:
    @pytest.mark.asyncio
    async def test_append_to_key_decisions(self):
        wm = _make_wm(key_decisions=["d01: use TDD"])
        tool, state = _make_tool(wm)
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="key_decisions", op="append",
                                       value="d02: keep interfaces small")
        result = await tool.execute(inp)
        assert result.is_error is False
        assert state._current_wm.key_decisions == [
            "d01: use TDD", "d02: keep interfaces small"
        ]


# ── 10. op=set list field (replace whole list) works ─────────────────────────

class TestSetListField:
    @pytest.mark.asyncio
    async def test_set_next_steps_replaces_list(self):
        wm = _make_wm(next_steps=["n01: old step"])
        tool, state = _make_tool(wm)
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="next_steps", op="set",
                                       value="n01: new step")
        result = await tool.execute(inp)
        assert result.is_error is False
        assert state._current_wm.next_steps == ["n01: new step"]


# ── 11. WM not initialized (None) returns error ───────────────────────────────

class TestWMNotInitialized:
    @pytest.mark.asyncio
    async def test_wm_none_returns_error(self):
        tool, _ = _make_tool(None)
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="progress", op="set", value="done")
        result = await tool.execute(inp)
        assert result.is_error is True
        assert "not initialized" in result.output.lower() or "none" in result.output.lower()


# ── 12. updated_by / updated_at are updated ──────────────────────────────────

class TestMetadataUpdated:
    @pytest.mark.asyncio
    async def test_updated_by_set_to_llm_tool(self):
        wm = _make_wm(updated_by="framework_init",
                      updated_at=datetime(2026, 1, 1))
        tool, state = _make_tool(wm)
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="progress", op="set", value="done")
        await tool.execute(inp)
        assert state._current_wm.updated_by == "llm_tool"

    @pytest.mark.asyncio
    async def test_updated_at_bumped(self):
        old_dt = datetime(2026, 1, 1)
        wm = _make_wm(updated_at=old_dt)
        tool, state = _make_tool(wm)
        from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryInput
        inp = UpdateWorkingMemoryInput(field="progress", op="set", value="done")
        await tool.execute(inp)
        assert state._current_wm.updated_at > old_dt
