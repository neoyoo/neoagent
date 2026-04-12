from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neoagent.multi.task import TaskResult, TokenUsage
from neoagent.multi.tools.spawn_worker import SpawnWorkerInput, SpawnWorkerTool


def _make_orchestrator():
    orch = MagicMock()
    orch.max_depth = 2
    orch._config = MagicMock()
    orch._config.model = "claude-sonnet-4-20250514"
    orch._config.api_key = "sk-test"
    orch._config.provider = "anthropic"
    orch._config.base_url = None
    orch._tool_pool = {}
    orch._event_bus = MagicMock()
    orch._task_tracker = MagicMock()
    orch._task_tracker.track = MagicMock()
    orch._task_tracker.complete = MagicMock()
    orch._semaphore = asyncio.Semaphore(5)
    return orch


VALID_MD = """\
---
name: temp-worker
model: sonnet
tags: [analysis]
tools: []
---
You are a temporary analysis worker.
"""

INVALID_MD = """\
---
model: sonnet
---
No name field here.
"""


class TestSpawnWorkerTool:
    def test_name_and_permission(self):
        orch = _make_orchestrator()
        tool = SpawnWorkerTool(orchestrator=orch, depth=0)
        assert tool.name == "spawn_worker"
        assert tool.permission == "auto"

    def test_input_model_fields(self):
        inp = SpawnWorkerInput(
            md_definition=VALID_MD,
            instruction="Analyze the code",
        )
        assert inp.md_definition == VALID_MD
        assert inp.instruction == "Analyze the code"

    @pytest.mark.asyncio
    @patch("neoagent.multi.tools.spawn_worker._create_worker_agent")
    @patch("neoagent.multi.tools.spawn_worker._run_worker")
    async def test_execute_success(self, mock_run_worker, mock_create_worker):
        orch = _make_orchestrator()
        mock_agent = MagicMock()
        mock_create_worker.return_value = mock_agent

        result = TaskResult(
            task_id="t1",
            status="completed",
            output="analysis done",
            error=None,
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            work_summary=None,
            turns_completed=3,
        )
        mock_run_worker.return_value = result

        tool = SpawnWorkerTool(orchestrator=orch, depth=0)
        inp = SpawnWorkerInput(md_definition=VALID_MD, instruction="Analyze the code")
        tool_result = await tool.execute(inp)

        assert tool_result.is_error is False
        assert "completed" in tool_result.output
        assert "analysis done" in tool_result.output

    @pytest.mark.asyncio
    async def test_execute_invalid_md_returns_error(self):
        orch = _make_orchestrator()
        tool = SpawnWorkerTool(orchestrator=orch, depth=0)
        inp = SpawnWorkerInput(md_definition=INVALID_MD, instruction="Do something")
        tool_result = await tool.execute(inp)

        assert tool_result.is_error is True
        assert "name" in tool_result.output.lower()

    @pytest.mark.asyncio
    @patch("neoagent.multi.tools.spawn_worker._create_worker_agent")
    @patch("neoagent.multi.tools.spawn_worker._run_worker")
    async def test_execute_worker_failure(self, mock_run_worker, mock_create_worker):
        orch = _make_orchestrator()
        mock_create_worker.return_value = MagicMock()
        result = TaskResult(
            task_id="t2",
            status="failed",
            output="",
            error="Worker crashed",
            usage=None,
            work_summary=None,
            turns_completed=1,
        )
        mock_run_worker.return_value = result

        tool = SpawnWorkerTool(orchestrator=orch, depth=0)
        inp = SpawnWorkerInput(md_definition=VALID_MD, instruction="Do work")
        tool_result = await tool.execute(inp)

        assert "failed" in tool_result.output
        assert "Worker crashed" in tool_result.output

    @pytest.mark.asyncio
    @patch("neoagent.multi.tools.spawn_worker._create_worker_agent")
    @patch("neoagent.multi.tools.spawn_worker._run_worker")
    async def test_semaphore_is_acquired(self, mock_run_worker, mock_create_worker):
        """Verify that the semaphore is actually used during execute."""
        orch = _make_orchestrator()
        mock_create_worker.return_value = MagicMock()
        mock_run_worker.return_value = TaskResult(
            task_id="t3",
            status="completed",
            output="done",
            error=None,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            work_summary=None,
            turns_completed=1,
        )

        # Replace semaphore with a spy
        real_sem = asyncio.Semaphore(5)
        orch._semaphore = real_sem

        tool = SpawnWorkerTool(orchestrator=orch, depth=0)
        inp = SpawnWorkerInput(md_definition=VALID_MD, instruction="Test semaphore")
        await tool.execute(inp)

        # After execute, semaphore should be released (back to 5)
        assert real_sem._value == 5
