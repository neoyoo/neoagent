from __future__ import annotations

from pydantic import BaseModel

from neoagent.core.types import ToolResult
from neoagent.tools.base import BaseTool


class SpawnWorkerInput(BaseModel):
    worker_name: str
    task: str


class SpawnWorkerTool(BaseTool):
    """Stub — full implementation in Task 7.

    Allows an orchestrator agent to spawn a named worker agent to handle a
    sub-task. Task 7 will flesh out the actual dispatch logic.
    """

    name = "spawn_worker"
    description = "Spawn a named worker agent to handle a sub-task."
    input_model = SpawnWorkerInput
    permission = "auto"
    is_concurrent_safe = True

    async def execute(self, input: BaseModel) -> ToolResult:  # type: ignore[override]
        raise NotImplementedError("SpawnWorkerTool is a stub; Task 7 will implement this.")
