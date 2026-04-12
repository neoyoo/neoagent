from __future__ import annotations

import copy
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neoagent.agent import NeoAgent

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


# ---------------------------------------------------------------------------
# Worker agent factory
# ---------------------------------------------------------------------------


def _create_worker_agent(
    card: WorkerCard,
    orchestrator: object,
    depth: int,
    task_id: str = "",
) -> "NeoAgent":
    """Create an independent NeoAgent instance configured from *card*.

    Steps:
    1. Build a NeoAgentConfig using card.model (or orchestrator model as fallback).
       Set system_prompt = card.instruction.
    2. Instantiate NeoAgent(config).
    3. Register only the tools from card.tools that exist in orchestrator._tool_pool.
    4. If depth < orchestrator.max_depth, also register SpawnWorkerTool.
    5. Wire event bubbling via _setup_event_bubble.
    6. Return the agent.
    """
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig
    from neoagent.multi.events import _setup_event_bubble
    from neoagent.multi.tools.spawn_worker import SpawnWorkerTool

    orch_config = orchestrator.config  # type: ignore[attr-defined]
    tool_pool: dict[str, object] = orchestrator._tool_pool  # type: ignore[attr-defined]

    worker_model = card.model if card.model is not None else orch_config.model

    worker_config = NeoAgentConfig(
        api_key=orch_config.api_key,
        model=worker_model,
        provider=orch_config.provider,
        base_url=orch_config.base_url,
        max_turns=orch_config.max_turns,
        context_budget=orch_config.context_budget,
        max_result_size=orch_config.max_result_size,
        auto_approve_tools=orch_config.auto_approve_tools,
        system_prompt=card.instruction,
    )

    agent = NeoAgent(worker_config)

    # Register only tools the card is authorised to use that exist in the pool
    for tool_name in card.tools:
        tool = tool_pool.get(tool_name)
        if tool is None:
            logger.debug(
                "_create_worker_agent: tool %r not found in orchestrator pool for worker %r — skipping",
                tool_name,
                card.name,
            )
            continue
        agent.register_tool(copy.copy(tool))  # type: ignore[arg-type]

    # Provide spawn capability only to workers below max_depth
    max_depth: int = orchestrator.max_depth  # type: ignore[attr-defined]
    if depth < max_depth:
        agent.register_tool(SpawnWorkerTool(orchestrator=orchestrator, depth=depth))

    # Use the provided task_id for event-bubble labelling; fall back to a new UUID
    # if not supplied (e.g. when worker is created before a Task is assigned).
    if not task_id:
        import uuid
        task_id = str(uuid.uuid4())

    _setup_event_bubble(agent, orchestrator, card.name, task_id, depth)

    return agent
