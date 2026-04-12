from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerCard:
    """Immutable descriptor for a worker agent loaded from a .md definition file."""

    name: str                   # unique identifier
    description: str            # capability description for LLM matching
    instruction: str            # system prompt (md body content)
    tags: tuple[str, ...]       # capability tags
    model: str | None           # None = inherit Orchestrator config
    tools: tuple[str, ...]      # authorized tool names


class WorkerPool:
    """Thread-safe registry of WorkerCard instances keyed by name."""

    def __init__(self) -> None:
        self._cards: dict[str, WorkerCard] = {}
        self._lock = threading.Lock()

    def register(self, card: WorkerCard) -> None:
        """Register a WorkerCard. Overwrites an existing card with the same name and logs a warning."""
        with self._lock:
            if card.name in self._cards:
                logger.warning(
                    "WorkerPool: overwriting existing worker %r with new registration",
                    card.name,
                )
            self._cards[card.name] = card

    def find(self, name: str) -> WorkerCard | None:
        """Return the WorkerCard for *name*, or None if not registered."""
        with self._lock:
            return self._cards.get(name)

    def list_all(self) -> list[WorkerCard]:
        """Return a snapshot copy of all registered WorkerCards."""
        with self._lock:
            return list(self._cards.values())

    def remove(self, name: str) -> bool:
        """Remove the worker with *name*. Returns True if removed, False if not found."""
        with self._lock:
            if name in self._cards:
                del self._cards[name]
                return True
            return False

    @property
    def size(self) -> int:
        """Number of registered workers."""
        with self._lock:
            return len(self._cards)
