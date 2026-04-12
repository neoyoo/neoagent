from __future__ import annotations

from neoagent.core.types import ToolResult as _ToolResult
from neoagent.multi.tools.cancel_task import CancelTaskTool
from neoagent.multi.tools.delegate_task import DelegateTaskTool
from neoagent.multi.tools.list_tools import ListTasksTool, ListWorkersTool
from neoagent.multi.tools.spawn_worker import SpawnWorkerTool

__all__ = [
    "SpawnWorkerTool",
    "DelegateTaskTool",
    "CancelTaskTool",
    "ListWorkersTool",
    "ListTasksTool",
    "_format_task_result",
]


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
