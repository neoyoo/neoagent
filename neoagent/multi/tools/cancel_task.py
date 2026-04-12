from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from neoagent.core.types import ToolResult
from neoagent.tools.base import BaseTool

if TYPE_CHECKING:
    from neoagent.multi.orchestrator import Orchestrator


class CancelTaskInput(BaseModel):
    task_id: str = Field(description="ID of the task to cancel")
    reason: str | None = Field(None, description="Optional cancellation reason")


class CancelTaskTool(BaseTool):
    """Cancel a running task by task_id.

    Returns success if the task was found and cancelled, error if not found
    or already completed.
    """

    name: str = "cancel_task"
    description: str = (
        "Cancel a running task by its task_id. "
        "Use list_tasks to find task IDs of running tasks."
    )
    input_model: type[BaseModel] = CancelTaskInput
    permission: str = "auto"

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self._orchestrator = orchestrator

    async def execute(self, input: BaseModel) -> ToolResult:  # type: ignore[override]
        assert isinstance(input, CancelTaskInput)
        # TaskTracker.cancel() is synchronous — call it directly
        cancelled = self._orchestrator._task_tracker.cancel(input.task_id)
        if cancelled:
            reason_str = f" (reason: {input.reason})" if input.reason else ""
            return ToolResult(
                call_id="",
                output=f"Task {input.task_id!r} cancelled successfully{reason_str}.",
            )
        return ToolResult(
            call_id="",
            output=f"Task {input.task_id!r} not found or already completed.",
            is_error=True,
        )
