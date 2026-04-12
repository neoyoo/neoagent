from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from neoagent.core.types import Message


@dataclass(frozen=True)
class TokenUsage:
    """Token consumption for a single task execution."""

    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Task:
    """An immutable unit of work dispatched to a subagent."""

    task_id: str
    instruction: str
    context: tuple[str, ...] = ()
    max_turns: int = 20
    timeout: int = 1800          # seconds
    max_output_tokens: int = 2000
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        instruction: str,
        context: tuple[str, ...] = (),
        max_turns: int = 20,
        timeout: int = 1800,
        max_output_tokens: int = 2000,
        metadata: dict | None = None,
    ) -> Task:
        """Factory that generates a UUID task_id automatically."""
        return cls(
            task_id=str(uuid.uuid4()),
            instruction=instruction,
            context=context,
            max_turns=max_turns,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            metadata=metadata if metadata is not None else {},
        )


@dataclass(frozen=True)
class TaskResult:
    """Immutable result of a completed, failed, or cancelled task."""

    task_id: str
    status: Literal["completed", "failed", "cancelled"]
    output: str | None
    error: str | None
    usage: TokenUsage
    work_summary: str
    turns_completed: int

    @property
    def is_success(self) -> bool:
        return self.status == "completed"


# ---------------------------------------------------------------------------
# TaskTracker
# ---------------------------------------------------------------------------

_INSTRUCTION_TRUNCATE_LEN = 80


def _truncate(text: str, max_len: int = _INSTRUCTION_TRUNCATE_LEN) -> str:
    """Truncate *text* to *max_len* characters, appending '...' when cut."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


class TaskTracker:
    """Thread-safe registry that tracks active and completed tasks.

    Lifecycle:
        track(task)                        — register as active
        set_asyncio_task(task_id, coro)    — associate asyncio.Task for cancel
        complete(task_id, result)          — mark done, store result
        cancel(task_id) -> bool            — cancel via asyncio.Task.cancel()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # task_id -> Task (all tasks ever registered)
        self._tasks: dict[str, Task] = {}
        # task_id -> TaskResult (only completed tasks)
        self._results: dict[str, TaskResult] = {}
        # task_id -> asyncio.Task (for cancellation)
        self._asyncio_tasks: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def track(self, task: Task) -> None:
        """Register *task* as active."""
        with self._lock:
            self._tasks[task.task_id] = task

    def set_asyncio_task(
        self, task_id: str, asyncio_task: "asyncio.Task[object]"
    ) -> None:
        """Associate an asyncio.Task with *task_id* for cancellation."""
        with self._lock:
            self._asyncio_tasks[task_id] = asyncio_task

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def complete(self, task_id: str, result: TaskResult) -> None:
        """Mark *task_id* as completed and store *result*."""
        with self._lock:
            self._results[task_id] = result
            # Remove asyncio task reference — no longer cancellable
            self._asyncio_tasks.pop(task_id, None)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Task | None:
        """Return the Task for *task_id*, or None if unknown."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> TaskResult | None:
        """Return the TaskResult for *task_id*, or None if not yet completed."""
        with self._lock:
            return self._results.get(task_id)

    def list_active(self) -> list[str]:
        """Return task_ids that have been tracked but not yet completed."""
        with self._lock:
            completed = set(self._results)
            return [tid for tid in self._tasks if tid not in completed]

    def list_all(self) -> list[dict]:
        """Return summary dicts for all tracked tasks.

        Each dict has: task_id, instruction (truncated), status.
        Status is the TaskResult.status for completed tasks, else "active".
        """
        with self._lock:
            summaries = []
            for task_id, task in self._tasks.items():
                result = self._results.get(task_id)
                status = result.status if result is not None else "active"
                summaries.append(
                    {
                        "task_id": task_id,
                        "instruction": _truncate(task.instruction),
                        "status": status,
                    }
                )
            return summaries

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self, task_id: str) -> bool:
        """Cancel *task_id* via its associated asyncio.Task.

        Returns False if the task is unknown, already completed, or has no
        associated asyncio.Task (i.e. cancel has no effect).
        Returns True after requesting cancellation.
        """
        with self._lock:
            if task_id not in self._tasks:
                return False
            if task_id in self._results:
                # Already completed — nothing to cancel
                return False
            asyncio_task = self._asyncio_tasks.get(task_id)

        if asyncio_task is None:
            # Registered but no asyncio.Task attached yet — treat as no-op
            return False

        asyncio_task.cancel()
        return True


# ---------------------------------------------------------------------------
# _extract_work_summary
# ---------------------------------------------------------------------------

_MAX_TOOL_CALLS = 5


def _extract_work_summary(messages: "list[Message]") -> str:  # noqa: UP006
    """Programmatically extract a work summary from a conversation message list.

    Collects:
    - Tool calls (ToolUseBlock) from assistant messages — keep last 5.
    - Last assistant text content.

    Returns a formatted string, or "（无有效工作摘要）" when nothing useful found.
    No LLM calls are made.
    """
    # Import here to avoid circular imports at module load time
    from neoagent.core.types import TextBlock, ToolUseBlock  # noqa: PLC0415

    tool_calls: list[str] = []
    last_assistant_text: str | None = None

    for msg in messages:
        if msg.role != "assistant":
            continue

        content = msg.content

        # String content — treat as text directly
        if isinstance(content, str):
            if content.strip():
                last_assistant_text = content.strip()
            continue

        # List of content blocks
        for block in content:
            if isinstance(block, ToolUseBlock):
                tool_calls.append(block.name)
            elif isinstance(block, TextBlock) and block.text.strip():
                last_assistant_text = block.text.strip()

    # Keep only last 5 tool calls
    recent_tools = tool_calls[-_MAX_TOOL_CALLS:]

    parts: list[str] = []

    if recent_tools:
        tools_str = ", ".join(recent_tools)
        parts.append(f"工具调用（最近 {len(recent_tools)} 次）: {tools_str}")

    if last_assistant_text is not None:
        parts.append(f"最后输出: {last_assistant_text}")

    if not parts:
        return "（无有效工作摘要）"

    return "\n".join(parts)
