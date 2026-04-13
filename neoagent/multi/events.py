from __future__ import annotations

import logging
from typing import Callable

from neoagent.events import Event, TaskCompleteEvent, TaskDispatchEvent, WorkerEvent, _ALL_EVENT_TYPES

__all__ = [
    "_setup_event_bubble",
    "WorkerEvent",
    "TaskDispatchEvent",
    "TaskCompleteEvent",
]

logger = logging.getLogger(__name__)


def _setup_event_bubble(
    worker_agent: object,
    orchestrator: object,
    worker_name: str,
    task_id: str,
    depth: int,
) -> Callable[[], None]:
    """Subscribe to all events on *worker_agent* and bubble them to *orchestrator*.

    Each event emitted on the worker's event_bus is wrapped in a WorkerEvent
    (carrying worker_name, task_id, depth, and the original event as `inner`)
    and re-emitted on the orchestrator's event_bus.

    Exceptions in the bubble handler are logged but never propagate, so a
    misbehaving orchestrator listener cannot crash the worker.

    Returns a teardown callable that unsubscribes the bubble handler from all
    event types. Call it when the worker task completes to prevent handler leaks.
    """
    worker_bus = worker_agent._event_bus  # type: ignore[attr-defined]
    orchestrator_bus = orchestrator._event_bus  # type: ignore[attr-defined]

    def _bubble(event: Event) -> None:
        try:
            wrapped = WorkerEvent(
                worker_name=worker_name,
                task_id=task_id,
                depth=depth,
                inner=event,
            )
            orchestrator_bus.emit(wrapped)
        except Exception:
            logger.exception(
                "_setup_event_bubble: error bubbling %s from worker %r (task %s)",
                type(event).__name__,
                worker_name,
                task_id,
            )

    worker_bus.subscribe_all(_bubble)

    def _teardown() -> None:
        for event_type in _ALL_EVENT_TYPES:
            worker_bus.unsubscribe(event_type, _bubble)

    return _teardown
