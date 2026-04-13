from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from neoagent.core.types import ToolResult as _ToolResult
from neoagent.multi.tools.cancel_task import CancelTaskTool
from neoagent.multi.tools.delegate_task import DelegateTaskTool
from neoagent.multi.tools.list_tools import ListTasksTool, ListWorkersTool
from neoagent.multi.tools.spawn_worker import SpawnWorkerTool

if TYPE_CHECKING:
    from neoagent.multi.orchestrator import Orchestrator
    from neoagent.multi.task import TaskResult

logger = logging.getLogger(__name__)

__all__ = [
    "SpawnWorkerTool",
    "DelegateTaskTool",
    "CancelTaskTool",
    "ListWorkersTool",
    "ListTasksTool",
]


def _emit_dispatch_event(
    orchestrator: "Orchestrator",
    task_id: str,
    worker_name: str,
    instruction: str,
    depth: int,
) -> None:
    """Emit TaskDispatchEvent on the orchestrator's event bus. Never raises."""
    from neoagent.events import TaskDispatchEvent  # noqa: PLC0415
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
    from neoagent.events import TaskCompleteEvent  # noqa: PLC0415
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


def _format_task_result(result: object) -> _ToolResult:  # type: ignore[return]
    """Format a TaskResult as a ToolResult text summary.

    Three formats:
    - completed: output + token stats
    - failed: error message (is_error=True)
    - cancelled: work_summary (is_error=True)
    """
    status = getattr(result, "status", None)
    task_id = getattr(result, "task_id", "?")
    turns = getattr(result, "turns_completed", 0)
    output = getattr(result, "output", "") or ""
    error = getattr(result, "error", None)
    usage = getattr(result, "usage", None)
    work_summary = getattr(result, "work_summary", None)

    if status == "completed":
        usage_str = ""
        if usage is not None:
            in_tok = getattr(usage, "input_tokens", 0)
            out_tok = getattr(usage, "output_tokens", 0)
            usage_str = f"\nTokens: {in_tok} in / {out_tok} out"
        return _ToolResult(
            call_id="",
            output=(
                f"[completed] task_id={task_id!r} "
                f"turns={turns}{usage_str}\n\n"
                f"{output}"
            ),
        )
    elif status == "failed":
        return _ToolResult(
            call_id="",
            output=f"[failed] task_id={task_id!r} error={error!r}",
            is_error=True,
        )
    else:
        # cancelled
        summary = work_summary or "(no summary)"
        return _ToolResult(
            call_id="",
            output=f"[cancelled] task_id={task_id!r}\nWork done: {summary}",
            is_error=True,
        )
