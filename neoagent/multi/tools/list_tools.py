from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from neoagent.core.types import ToolResult
from neoagent.tools.base import BaseTool

if TYPE_CHECKING:
    from neoagent.multi.orchestrator import Orchestrator


class _EmptyInput(BaseModel):
    pass


class ListWorkersTool(BaseTool):
    """List all registered static workers with their descriptions and tags."""

    name: str = "list_workers"
    description: str = (
        "List all registered static Workers with their names, descriptions, and tags. "
        "Use this before calling delegate_task to confirm the worker name."
    )
    input_model: type[BaseModel] = _EmptyInput
    permission: str = "auto"
    is_concurrent_safe: bool = True

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self._orchestrator = orchestrator

    async def execute(self, input: BaseModel) -> ToolResult:  # type: ignore[override]
        workers = self._orchestrator._worker_pool.list_all()
        if not workers:
            return ToolResult(call_id="", output="No workers registered.")
        lines = []
        for card in workers:
            tags_str = ", ".join(card.tags) if card.tags else "(none)"
            tools_str = ", ".join(card.tools) if card.tools else "(none)"
            lines.append(
                f"- {card.name}: {card.description}\n"
                f"  tags: {tags_str} | tools: {tools_str}"
            )
        return ToolResult(call_id="", output="\n".join(lines))


class ListTasksTool(BaseTool):
    """List all tasks (active and completed) with their current status."""

    name: str = "list_tasks"
    description: str = (
        "List all tasks (running, completed, failed, cancelled) with their status. "
        "Use cancel_task with a task_id to cancel a running task."
    )
    input_model: type[BaseModel] = _EmptyInput
    permission: str = "auto"
    is_concurrent_safe: bool = True

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self._orchestrator = orchestrator

    async def execute(self, input: BaseModel) -> ToolResult:  # type: ignore[override]
        tasks = self._orchestrator._task_tracker.list_all()
        if not tasks:
            return ToolResult(call_id="", output="No tasks recorded.")
        lines = []
        for t in tasks:
            task_id = t.get("task_id", "?")
            status = t.get("status", "?")
            instruction = t.get("instruction", "")
            worker = t.get("worker_name", "?")
            # Build a compact line; include instruction if available
            if instruction:
                lines.append(
                    f"- [{status}] task_id={task_id!r} worker={worker!r}  {instruction!r}"
                )
            else:
                lines.append(f"- [{status}] task_id={task_id!r} worker={worker!r}")
        return ToolResult(call_id="", output="\n".join(lines))
