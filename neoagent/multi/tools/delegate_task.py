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

logger = logging.getLogger(__name__)


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

        # 2. Create worker agent
        task_id = str(uuid.uuid4())[:8]
        worker_agent = _create_worker_agent(card, self._orchestrator, self._depth + 1)

        # 3. Build context tuple — store dicts as JSON strings so Task stays
        #    pure (tuple[str, ...]) while _run_worker can decode them back.
        context_strs: tuple[str, ...] = ()
        if input.context:
            context_strs = tuple(
                json.dumps(m) for m in input.context
                if isinstance(m, dict) and "role" in m and "content" in m
            )

        # 4. Build Task
        task = Task(
            task_id=task_id,
            instruction=input.instruction,
            context=context_strs,
        )

        # 5. Track then execute under semaphore
        self._orchestrator._task_tracker.track(task)
        async with self._orchestrator._semaphore:
            result = await _run_worker(worker_agent, task, self._orchestrator._task_tracker)

        # 6. Record completion
        self._orchestrator._task_tracker.complete(task_id, result)

        # 7. Format and return
        return _format_task_result(result)
