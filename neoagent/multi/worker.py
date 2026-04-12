from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerCard:
    """Immutable descriptor for a worker agent loaded from a .md definition file."""

    name: str                   # unique identifier
    description: str            # capability description for LLM matching
    instruction: str            # system prompt (md body content)
    tags: tuple[str, ...]       # capability tags
    model: str | None           # None = inherit Orchestrator config
    tools: tuple[str, ...]      # authorized tool names
