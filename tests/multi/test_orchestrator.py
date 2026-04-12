from __future__ import annotations
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import tempfile
from neoagent.multi.orchestrator import Orchestrator
from neoagent.multi.worker import WorkerCard
from neoagent.config import NeoAgentConfig
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult
from pydantic import BaseModel


def _make_config():
    return NeoAgentConfig(api_key="sk-test", model="claude-sonnet-4-20250514")


class DummyInput(BaseModel):
    x: str = ""


class DummyTool(BaseTool):
    name: str = "read"
    description: str = "reads files"
    input_model: type[BaseModel] = DummyInput
    permission: str = "auto"

    async def execute(self, input):
        return ToolResult(call_id="", output="content")


class TestOrchestratorConstruction:
    @patch("neoagent.multi.orchestrator._create_provider")
    def test_creates_successfully(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        orch = Orchestrator(_make_config())
        assert orch is not None

    @patch("neoagent.multi.orchestrator._create_provider")
    def test_default_params(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        orch = Orchestrator(_make_config())
        assert orch.max_depth == 2
        assert orch.max_concurrent_workers == 5

    @patch("neoagent.multi.orchestrator._create_provider")
    def test_custom_params(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        orch = Orchestrator(_make_config(), max_depth=3, max_concurrent_workers=10)
        assert orch.max_depth == 3
        assert orch.max_concurrent_workers == 10

    @patch("neoagent.multi.orchestrator._create_provider")
    def test_registers_5_internal_tools(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        orch = Orchestrator(_make_config())
        schemas = orch._brain._registry.get_schemas()
        tool_names = {s["name"] for s in schemas}
        assert "spawn_worker" in tool_names
        assert "delegate_task" in tool_names
        assert "cancel_task" in tool_names
        assert "list_workers" in tool_names
        assert "list_tasks" in tool_names

    @patch("neoagent.multi.orchestrator._create_provider")
    def test_semaphore_limit(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        orch = Orchestrator(_make_config(), max_concurrent_workers=3)
        # Semaphore value should equal max_concurrent_workers
        assert orch._semaphore._value == 3


class TestOrchestratorWorkerRegistration:
    @patch("neoagent.multi.orchestrator._create_provider")
    def test_register_worker(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        orch = Orchestrator(_make_config())
        card = WorkerCard(name="reviewer", description="reviews", instruction="You review.", tags=(), tools=(), model=None)
        orch.register_worker(card)
        assert orch._worker_pool.find("reviewer") is card

    @patch("neoagent.multi.orchestrator._create_provider")
    def test_load_workers_from_directory(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        orch = Orchestrator(_make_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            md_content = (
                "---\nname: file-worker\nmodel: sonnet\ntags: [test]\ntools: []\n---\n"
                "You are a file worker."
            )
            (Path(tmpdir) / "file-worker.md").write_text(md_content)
            orch.load_workers(tmpdir)
        assert orch._worker_pool.find("file-worker") is not None

    @patch("neoagent.multi.orchestrator._create_provider")
    def test_register_tool_adds_to_pool(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        orch = Orchestrator(_make_config())
        tool = DummyTool()
        orch.register_tool(tool)
        assert orch._tool_pool.get("read") is tool


class TestOrchestratorRun:
    @patch("neoagent.multi.orchestrator._create_provider")
    async def test_run_delegates_to_brain_chat(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        orch = Orchestrator(_make_config())
        orch._brain.chat = AsyncMock(return_value="task dispatched")
        result = await orch.run("Review these files")
        orch._brain.chat.assert_called_once_with("Review these files")
        assert result == "task dispatched"

    @patch("neoagent.multi.orchestrator._create_provider")
    async def test_run_returns_string(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        orch = Orchestrator(_make_config())
        orch._brain.chat = AsyncMock(return_value="All tasks done.")
        result = await orch.run("Do the work")
        assert isinstance(result, str)


class TestOrchestratorClose:
    @patch("neoagent.multi.orchestrator._create_provider")
    async def test_close_cancels_active_tasks(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        orch = Orchestrator(_make_config())

        # Register a fake running asyncio task via the existing TaskTracker API
        from neoagent.multi.task import Task
        async def never_ending():
            await asyncio.sleep(9999)

        task_data = Task.create(instruction="never ending work")
        asyncio_task = asyncio.create_task(never_ending())
        orch._task_tracker.track(task_data)
        orch._task_tracker.set_asyncio_task(task_data.task_id, asyncio_task)

        await orch.close()
        # Give the event loop a chance to process the cancellation
        await asyncio.sleep(0)
        assert asyncio_task.cancelled() or asyncio_task.cancelling() > 0 or asyncio_task.done()

    @patch("neoagent.multi.orchestrator._create_provider")
    async def test_close_calls_brain_close(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        orch = Orchestrator(_make_config())
        orch._brain.close = AsyncMock()
        await orch.close()
        orch._brain.close.assert_called_once()

    @patch("neoagent.multi.orchestrator._create_provider")
    async def test_context_manager_calls_close(self, mock_create_provider):
        mock_create_provider.return_value = MagicMock(get_context_window=lambda: 200_000)
        closed = []

        async with Orchestrator(_make_config()) as orch:
            orch._brain.close = AsyncMock(side_effect=lambda: closed.append(True))

        assert closed  # close() was called
