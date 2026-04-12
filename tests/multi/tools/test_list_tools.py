from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from neoagent.multi.tools.list_tools import ListTasksTool, ListWorkersTool
from neoagent.multi.worker import WorkerCard


def _make_orchestrator_with_workers():
    orch = MagicMock()
    cards = [
        WorkerCard(
            name="reviewer",
            description="reviews code for security issues",
            instruction="You review code.",
            tags=("review", "security"),
            model=None,
            tools=("read", "grep"),
        ),
        WorkerCard(
            name="writer",
            description="writes documentation",
            instruction="You write docs.",
            tags=("docs",),
            model=None,
            tools=("read", "write"),
        ),
    ]
    orch._worker_pool = MagicMock()
    orch._worker_pool.list_all.return_value = cards
    orch._task_tracker = MagicMock()
    return orch


class EmptyInput(BaseModel):
    pass


class TestListWorkersTool:
    def test_name_and_permission(self):
        orch = _make_orchestrator_with_workers()
        tool = ListWorkersTool(orchestrator=orch)
        assert tool.name == "list_workers"
        assert tool.permission == "auto"
        assert tool.is_concurrent_safe is True

    @pytest.mark.asyncio
    async def test_lists_all_workers(self):
        orch = _make_orchestrator_with_workers()
        tool = ListWorkersTool(orchestrator=orch)
        result = await tool.execute(EmptyInput())
        assert result.is_error is False
        assert "reviewer" in result.output
        assert "writer" in result.output
        assert "reviews code for security issues" in result.output

    @pytest.mark.asyncio
    async def test_empty_pool(self):
        orch = _make_orchestrator_with_workers()
        orch._worker_pool.list_all.return_value = []
        tool = ListWorkersTool(orchestrator=orch)
        result = await tool.execute(EmptyInput())
        assert result.is_error is False
        assert "no workers" in result.output.lower() or result.output.strip() == ""

    @pytest.mark.asyncio
    async def test_shows_tags_and_tools(self):
        orch = _make_orchestrator_with_workers()
        tool = ListWorkersTool(orchestrator=orch)
        result = await tool.execute(EmptyInput())
        # Should contain tag and tool info
        assert "review" in result.output
        assert "read" in result.output


class TestListTasksTool:
    def test_name_and_permission(self):
        orch = _make_orchestrator_with_workers()
        tool = ListTasksTool(orchestrator=orch)
        assert tool.name == "list_tasks"
        assert tool.permission == "auto"
        assert tool.is_concurrent_safe is True

    @pytest.mark.asyncio
    async def test_lists_all_tasks(self):
        orch = _make_orchestrator_with_workers()
        orch._task_tracker.list_all.return_value = [
            {"task_id": "t1", "status": "completed", "worker_name": "reviewer"},
            {"task_id": "t2", "status": "active", "worker_name": "writer"},
        ]
        tool = ListTasksTool(orchestrator=orch)
        result = await tool.execute(EmptyInput())
        assert result.is_error is False
        assert "t1" in result.output
        assert "t2" in result.output
        assert "completed" in result.output

    @pytest.mark.asyncio
    async def test_empty_tasks(self):
        orch = _make_orchestrator_with_workers()
        orch._task_tracker.list_all.return_value = []
        tool = ListTasksTool(orchestrator=orch)
        result = await tool.execute(EmptyInput())
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_shows_instruction_field(self):
        """list_all() returns dicts with instruction and status from TaskTracker."""
        orch = _make_orchestrator_with_workers()
        orch._task_tracker.list_all.return_value = [
            {"task_id": "t99", "instruction": "do the thing", "status": "active"},
        ]
        tool = ListTasksTool(orchestrator=orch)
        result = await tool.execute(EmptyInput())
        assert "t99" in result.output
        assert "active" in result.output
