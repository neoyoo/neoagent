from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from neoagent.core.types import ToolResult
from neoagent.multi.parser import parse_worker_md
from neoagent.multi.task import Task, _run_worker
from neoagent.multi.worker import _create_worker_agent
from neoagent.tools.base import BaseTool

if TYPE_CHECKING:
    from neoagent.multi.orchestrator import Orchestrator
    from neoagent.multi.task import TaskResult

logger = logging.getLogger(__name__)


class SpawnWorkerInput(BaseModel):
    md_definition: str = Field(
        description=(
            "WorkerCard definition in .md format:\n"
            "---\nname: worker-name\nmodel: sonnet\ntags: [tag1, tag2]\ntools: [read, grep]\n---\n"
            "Worker system prompt here."
        )
    )
    instruction: str = Field(
        description="The task instruction to send to the spawned worker."
    )


class SpawnWorkerTool(BaseTool):
    """Dynamically spawn a temporary Worker from a markdown definition and run a task.

    The WorkerCard is NOT registered to WorkerPool — it is ephemeral.
    Parses md_definition → WorkerCard → NeoAgent → runs task → returns result.
    """

    name: str = "spawn_worker"
    description: str = (
        "Dynamically create a temporary Worker from a markdown definition and run a task. "
        "The md_definition must be a valid WorkerCard .md format with a 'name' field. "
        "The worker is ephemeral — not added to the worker pool."
    )
    input_model: type[BaseModel] = SpawnWorkerInput
    permission: str = "auto"
    is_concurrent_safe: bool = True

    def __init__(self, orchestrator: "Orchestrator", depth: int = 0) -> None:
        self._orchestrator = orchestrator
        self._depth = depth

    async def execute(self, input: BaseModel) -> ToolResult:  # type: ignore[override]
        assert isinstance(input, SpawnWorkerInput)

        from neoagent.multi.tools import (  # noqa: PLC0415
            _emit_complete_event,
            _emit_dispatch_event,
            _format_task_result,
        )

        # 1. Parse md_definition → WorkerCard
        try:
            card = parse_worker_md(input.md_definition)
        except ValueError as exc:
            return ToolResult(
                call_id="",
                output=f"Failed to parse worker definition: {exc}",
                is_error=True,
            )

        # 2. Build Task with full UUID task_id
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            instruction=input.instruction,
        )

        # 3. Create worker agent with the known task_id for correct event labelling
        worker_agent = _create_worker_agent(
            card, self._orchestrator, self._depth + 1, task_id=task_id
        )

        # 4. Emit dispatch event
        _emit_dispatch_event(
            self._orchestrator, task_id, card.name, input.instruction, self._depth
        )

        # 5. Track then execute under semaphore; register asyncio.Task for cancellation
        self._orchestrator._task_tracker.track(task, worker_name=card.name)
        async with self._orchestrator._semaphore:
            asyncio_task = asyncio.create_task(
                _run_worker(worker_agent, task, self._orchestrator._task_tracker)
            )
            self._orchestrator._task_tracker.set_asyncio_task(task_id, asyncio_task)
            result = await asyncio_task

        # 6. Record completion and emit complete event
        self._orchestrator._task_tracker.complete(task_id, result)
        _emit_complete_event(self._orchestrator, result, card.name, self._depth)

        # 7. Format and return
        return _format_task_result(result)
