from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from neoagent.core.types import ToolResult
from neoagent.multi.task import Task, _run_worker
from neoagent.multi.worker import _create_worker_agent
from neoagent.tools.base import BaseTool

if TYPE_CHECKING:
    from neoagent.multi.orchestrator import Orchestrator
    from neoagent.multi.task import TaskResult

logger = logging.getLogger(__name__)


def _emit_dispatch_event(
    orchestrator: "Orchestrator",
    task_id: str,
    worker_name: str,
    instruction: str,
    depth: int,
) -> None:
    """Emit TaskDispatchEvent on the orchestrator's event bus. Never raises."""
    from neoagent.events import TaskDispatchEvent
    try:
        orchestrator._event_bus.emit(TaskDispatchEvent(
            task_id=task_id,
            worker_name=worker_name,
            instruction=instruction,
            depth=depth,
        ))
    except Exception:
        logger.debug("_emit_dispatch_event: failed to emit for task %r", task_id)


def _emit_complete_event(
    orchestrator: "Orchestrator",
    result: "TaskResult",
    worker_name: str,
    depth: int,
) -> None:
    """Emit TaskCompleteEvent on the orchestrator's event bus. Never raises."""
    from neoagent.events import TaskCompleteEvent
    try:
        orchestrator._event_bus.emit(TaskCompleteEvent(
            task_id=result.task_id,
            worker_name=worker_name,
            status=result.status,
            turns_completed=result.turns_completed,
            usage=result.usage,
        ))
    except Exception:
        logger.debug("_emit_complete_event: failed to emit for task %r", result.task_id)


class DelegateTaskInput(BaseModel):
    worker_name: str = Field(
        description=(
            "Name of the registered worker to delegate to. "
            "Use list_workers to see available workers."
        )
    )
    instruction: str = Field(
        description="The task instruction to send to the worker."
    )
    context: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "Optional context messages to include "
            "(list of {role, content} dicts)."
        ),
    )


class DelegateTaskTool(BaseTool):
    """Route a task to a registered static Worker by name.

    Looks up the WorkerCard from WorkerPool, creates a fresh NeoAgent,
    and runs the task synchronously (from the caller's perspective).
    Returns a TaskResult summary.
    """

    name: str = "delegate_task"
    description: str = (
        "Delegate a task to a registered static Worker by name. "
        "Use list_workers to see available workers and their capabilities. "
        "Returns the worker's output when complete."
    )
    input_model: type[BaseModel] = DelegateTaskInput
    permission: str = "auto"

    def __init__(self, orchestrator: "Orchestrator", depth: int = 0) -> None:
        self._orchestrator = orchestrator
        self._depth = depth

    async def execute(self, input: BaseModel) -> ToolResult:  # type: ignore[override]
        assert isinstance(input, DelegateTaskInput)

        from neoagent.multi.tools import _format_task_result  # noqa: PLC0415

        # 1. Look up worker in pool
        card = self._orchestrator._worker_pool.find(input.worker_name)
        if card is None:
            return ToolResult(
                call_id="",
                output=(
                    f"Worker {input.worker_name!r} not found. "
                    "Use list_workers to see available workers."
                ),
                is_error=True,
            )

        # 2. Build Task first so task_id is known for event emission
        task_id = str(uuid.uuid4())[:8]

        # 3. Build context tuple — store dicts as JSON strings so Task stays
        #    pure (tuple[str, ...]) while _run_worker can decode them back.
        context_strs: tuple[str, ...] = ()
        if input.context:
            context_strs = tuple(
                json.dumps(m) for m in input.context
                if isinstance(m, dict) and "role" in m and "content" in m
            )

        task = Task(
            task_id=task_id,
            instruction=input.instruction,
            context=context_strs,
        )

        # 4. Create worker agent with the known task_id for correct event labelling
        worker_agent = _create_worker_agent(
            card, self._orchestrator, self._depth + 1, task_id=task_id
        )

        # 5. Emit dispatch event
        _emit_dispatch_event(
            self._orchestrator, task_id, input.worker_name, input.instruction, self._depth
        )

        # 6. Track then execute under semaphore
        self._orchestrator._task_tracker.track(task)
        async with self._orchestrator._semaphore:
            result = await _run_worker(worker_agent, task, self._orchestrator._task_tracker)

        # 7. Record completion and emit complete event
        self._orchestrator._task_tracker.complete(task_id, result)
        _emit_complete_event(self._orchestrator, result, input.worker_name, self._depth)

        # 8. Format and return
        return _format_task_result(result)
