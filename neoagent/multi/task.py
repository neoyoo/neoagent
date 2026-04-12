from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal


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
