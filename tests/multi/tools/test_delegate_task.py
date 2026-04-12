from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neoagent.multi.task import TaskResult, TokenUsage
from neoagent.multi.tools.delegate_task import DelegateTaskInput, DelegateTaskTool
from neoagent.multi.worker import WorkerCard


def _make_card():
    return WorkerCard(
        name="reviewer",
        description="reviews code",
        instruction="You review code.",
        tags=(),
        model=None,
        tools=(),
    )


def _make_orchestrator_with_pool():
    card = _make_card()
    orch = MagicMock()
    orch.max_depth = 2
    orch.config = MagicMock()
    orch.config.model = "claude-sonnet-4-20250514"
    orch.config.api_key = "sk-test"
    orch.config.provider = "anthropic"
    orch.config.base_url = None
    orch._tool_pool = {}
    orch._event_bus = MagicMock()
    orch._semaphore = asyncio.Semaphore(5)
    orch._worker_pool = MagicMock()
    orch._worker_pool.find.return_value = card
    orch._task_tracker = MagicMock()
    orch._task_tracker.track = MagicMock()
    orch._task_tracker.complete = MagicMock()
    return orch, card


class TestDelegateTaskTool:
    def test_name_and_permission(self):
        orch, _ = _make_orchestrator_with_pool()
        tool = DelegateTaskTool(orchestrator=orch, depth=0)
        assert tool.name == "delegate_task"
        assert tool.permission == "auto"

    @pytest.mark.asyncio
    @patch("neoagent.multi.tools.delegate_task._create_worker_agent")
    @patch("neoagent.multi.tools.delegate_task._run_worker")
    async def test_execute_success(self, mock_run, mock_create):
        orch, _ = _make_orchestrator_with_pool()
        mock_create.return_value = MagicMock()
        result = TaskResult(
            task_id="t3",
            status="completed",
            output="review done",
            error=None,
            usage=TokenUsage(input_tokens=80, output_tokens=40),
            work_summary=None,
            turns_completed=2,
        )
        mock_run.return_value = result

        tool = DelegateTaskTool(orchestrator=orch, depth=0)
        inp = DelegateTaskInput(worker_name="reviewer", instruction="Review auth.py")
        tool_result = await tool.execute(inp)

        assert tool_result.is_error is False
        assert "completed" in tool_result.output
        assert "review done" in tool_result.output
        orch._worker_pool.find.assert_called_once_with("reviewer")

    @pytest.mark.asyncio
    async def test_execute_worker_not_found(self):
        orch, _ = _make_orchestrator_with_pool()
        orch._worker_pool.find.return_value = None

        tool = DelegateTaskTool(orchestrator=orch, depth=0)
        inp = DelegateTaskInput(worker_name="nonexistent", instruction="Do work")
        tool_result = await tool.execute(inp)

        assert tool_result.is_error is True
        assert "nonexistent" in tool_result.output
        assert "list_workers" in tool_result.output.lower()

    @pytest.mark.asyncio
    @patch("neoagent.multi.tools.delegate_task._create_worker_agent")
    @patch("neoagent.multi.tools.delegate_task._run_worker")
    async def test_execute_worker_failed(self, mock_run, mock_create):
        orch, _ = _make_orchestrator_with_pool()
        mock_create.return_value = MagicMock()
        result = TaskResult(
            task_id="t4",
            status="failed",
            output="",
            error="timeout",
            usage=None,
            work_summary=None,
            turns_completed=5,
        )
        mock_run.return_value = result

        tool = DelegateTaskTool(orchestrator=orch, depth=0)
        inp = DelegateTaskInput(worker_name="reviewer", instruction="Review utils.py")
        tool_result = await tool.execute(inp)

        assert "failed" in tool_result.output
        assert "timeout" in tool_result.output

    @pytest.mark.asyncio
    @patch("neoagent.multi.tools.delegate_task._create_worker_agent")
    @patch("neoagent.multi.tools.delegate_task._run_worker")
    async def test_passes_context(self, mock_run, mock_create):
        orch, _ = _make_orchestrator_with_pool()
        mock_create.return_value = MagicMock()
        mock_run.return_value = TaskResult(
            task_id="t5",
            status="completed",
            output="ok",
            error=None,
            usage=None,
            work_summary=None,
            turns_completed=1,
        )

        tool = DelegateTaskTool(orchestrator=orch, depth=0)
        inp = DelegateTaskInput(
            worker_name="reviewer",
            instruction="Review this",
            context=[{"role": "user", "content": "Background info"}],
        )
        await tool.execute(inp)
        # _run_worker should have received the task with context
        _, task, _ = mock_run.call_args[0]
        assert len(task.context) == 1
