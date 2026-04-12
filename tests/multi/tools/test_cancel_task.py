from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neoagent.multi.tools.cancel_task import CancelTaskInput, CancelTaskTool


def _make_orchestrator():
    orch = MagicMock()
    orch._task_tracker = MagicMock()
    return orch


class TestCancelTaskTool:
    def test_name_and_permission(self):
        orch = _make_orchestrator()
        tool = CancelTaskTool(orchestrator=orch)
        assert tool.name == "cancel_task"
        assert tool.permission == "auto"

    @pytest.mark.asyncio
    async def test_cancel_existing_task(self):
        orch = _make_orchestrator()
        # cancel() is synchronous — return True
        orch._task_tracker.cancel = MagicMock(return_value=True)
        tool = CancelTaskTool(orchestrator=orch)
        inp = CancelTaskInput(task_id="abc123")
        result = await tool.execute(inp)
        assert result.is_error is False
        assert "abc123" in result.output
        orch._task_tracker.cancel.assert_called_once_with("abc123")

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self):
        orch = _make_orchestrator()
        orch._task_tracker.cancel = MagicMock(return_value=False)
        tool = CancelTaskTool(orchestrator=orch)
        inp = CancelTaskInput(task_id="ghost-id")
        result = await tool.execute(inp)
        assert result.is_error is True
        assert "ghost-id" in result.output

    @pytest.mark.asyncio
    async def test_cancel_with_reason(self):
        orch = _make_orchestrator()
        orch._task_tracker.cancel = MagicMock(return_value=True)
        tool = CancelTaskTool(orchestrator=orch)
        inp = CancelTaskInput(task_id="t1", reason="no longer needed")
        result = await tool.execute(inp)
        assert result.is_error is False
        assert "no longer needed" in result.output
