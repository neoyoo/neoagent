from __future__ import annotations

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

        from neoagent.multi.tools import _format_task_result  # noqa: PLC0415

        # 1. Parse md_definition → WorkerCard
        try:
            card = parse_worker_md(input.md_definition)
        except ValueError as exc:
            return ToolResult(
                call_id="",
                output=f"Failed to parse worker definition: {exc}",
                is_error=True,
            )

        # 2. Create worker agent (depth + 1 for spawned sub-workers)
        task_id = str(uuid.uuid4())[:8]
        worker_agent = _create_worker_agent(card, self._orchestrator, self._depth + 1)

        # 3. Build Task
        task = Task(
            task_id=task_id,
            instruction=input.instruction,
        )

        # 4. Track then execute under semaphore
        self._orchestrator._task_tracker.track(task)
        async with self._orchestrator._semaphore:
            result = await _run_worker(worker_agent, task, self._orchestrator._task_tracker)

        # 5. Record completion
        self._orchestrator._task_tracker.complete(task_id, result)

        # 6. Format and return
        return _format_task_result(result)
