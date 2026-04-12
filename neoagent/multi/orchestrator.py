from __future__ import annotations
import asyncio
import logging
from pathlib import Path

from neoagent.config import NeoAgentConfig
from neoagent.agent import NeoAgent
from neoagent.multi.worker import WorkerPool
from neoagent.multi.task import TaskTracker
from neoagent.tools.base import BaseTool

logger = logging.getLogger(__name__)


def _create_provider(config: NeoAgentConfig):
    """Re-export to allow patching in tests."""
    from neoagent.agent import _create_provider as _cp
    return _cp(config)


class Orchestrator:
    """Multi-agent coordinator.

    Composes a NeoAgent (brain) with WorkerPool and TaskTracker.
    The brain's LLM has access to 5 internal tools for spawning and delegating tasks.
    Workers are independent NeoAgent instances created on demand.

    Usage::

        orch = Orchestrator(NeoAgentConfig(api_key="..."))
        orch.register_worker(WorkerCard(name="reviewer", ...))
        orch.register_tool(ReadTool())
        result = await orch.run("Review these three files for security issues")
    """

    def __init__(
        self,
        config: NeoAgentConfig,
        max_depth: int = 2,
        max_concurrent_workers: int = 5,
    ) -> None:
        self.config = config
        self.max_depth = max_depth
        self.max_concurrent_workers = max_concurrent_workers

        # Internal components
        self._worker_pool = WorkerPool()
        self._task_tracker = TaskTracker()
        self._tool_pool: dict[str, BaseTool] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_workers)

        # Brain: the orchestrator's own NeoAgent
        self._brain = NeoAgent(config)

        # Expose the brain's EventBus directly on the orchestrator for tools
        # and event subscribers to use (e.g. TaskDispatchEvent, WorkerEvent).
        self._event_bus = self._brain._event_bus

        # Register 5 internal tools onto the brain
        self._register_internal_tools()

    def _register_internal_tools(self) -> None:
        from neoagent.multi.tools.spawn_worker import SpawnWorkerTool
        from neoagent.multi.tools.delegate_task import DelegateTaskTool
        from neoagent.multi.tools.cancel_task import CancelTaskTool
        from neoagent.multi.tools.list_tools import ListWorkersTool, ListTasksTool

        self._brain.register_tool(SpawnWorkerTool(orchestrator=self, depth=0))
        self._brain.register_tool(DelegateTaskTool(orchestrator=self, depth=0))
        self._brain.register_tool(CancelTaskTool(orchestrator=self))
        self._brain.register_tool(ListWorkersTool(orchestrator=self))
        self._brain.register_tool(ListTasksTool(orchestrator=self))

    # ── Worker registration ───────────────────────────────────────────────────

    def register_worker(self, card) -> None:
        """Register a static WorkerCard by reference."""
        self._worker_pool.register(card)

    def load_workers(self, directory: str | Path) -> None:
        """Load all .md WorkerCard files from a directory and register them."""
        from neoagent.multi.parser import load_workers
        for card in load_workers(Path(directory)):
            self._worker_pool.register(card)

    # ── Tool pool management ──────────────────────────────────────────────────

    def register_tool(self, tool: BaseTool) -> None:
        """Add a tool to the global pool available for Worker authorization."""
        self._tool_pool[tool.name] = tool

    # ── Execution ─────────────────────────────────────────────────────────────

    async def run(self, message: str) -> str:
        """Run the orchestrator with a user message.

        The brain's LLM decides how to decompose and delegate the work.
        Returns the final response string.
        """
        return await self._brain.chat(message)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Cancel all running worker tasks and clean up the brain."""
        # Cancel all tracked active tasks via their asyncio.Task handles
        active_ids = self._task_tracker.list_active()
        for task_id in active_ids:
            self._task_tracker.cancel(task_id)

        # Close brain (MCP connections etc.)
        await self._brain.close()

    async def __aenter__(self) -> "Orchestrator":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
