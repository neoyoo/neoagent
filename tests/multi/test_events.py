from __future__ import annotations

import dataclasses

import pytest

from neoagent.events import (
    EventBus,
    ProviderResponseEvent,
    TaskCompleteEvent,
    TaskDispatchEvent,
    TurnCompleteEvent,
    WorkerEvent,
    _ALL_EVENT_TYPES,
)
from neoagent.multi.events import _setup_event_bubble


# ---------------------------------------------------------------------------
# WorkerEvent
# ---------------------------------------------------------------------------


class TestWorkerEvent:
    def test_fields(self) -> None:
        inner = TurnCompleteEvent(turn_index=1, stop_reason="end_turn", tool_call_count=0)
        evt = WorkerEvent(
            worker_name="coder",
            task_id="t-001",
            depth=1,
            inner=inner,
        )
        assert evt.worker_name == "coder"
        assert evt.task_id == "t-001"
        assert evt.depth == 1
        assert evt.inner is inner

    def test_frozen(self) -> None:
        inner = TurnCompleteEvent(turn_index=0, stop_reason="end_turn", tool_call_count=0)
        evt = WorkerEvent(worker_name="w", task_id="t", depth=0, inner=inner)
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.worker_name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TaskDispatchEvent
# ---------------------------------------------------------------------------


class TestTaskDispatchEvent:
    def test_fields(self) -> None:
        evt = TaskDispatchEvent(
            task_id="t-002",
            worker_name="analyst",
            instruction="Analyze the logs",
            depth=2,
        )
        assert evt.task_id == "t-002"
        assert evt.worker_name == "analyst"
        assert evt.instruction == "Analyze the logs"
        assert evt.depth == 2

    def test_frozen(self) -> None:
        evt = TaskDispatchEvent(
            task_id="t",
            worker_name="w",
            instruction="do it",
            depth=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.task_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TaskCompleteEvent
# ---------------------------------------------------------------------------


class TestTaskCompleteEvent:
    def test_fields(self) -> None:
        evt = TaskCompleteEvent(
            task_id="t-003",
            worker_name="coder",
            status="completed",
            turns_completed=5,
            usage=None,
        )
        assert evt.task_id == "t-003"
        assert evt.worker_name == "coder"
        assert evt.status == "completed"
        assert evt.turns_completed == 5
        assert evt.usage is None

    def test_nullable_usage(self) -> None:
        from neoagent.multi.task import TokenUsage

        usage = TokenUsage(input_tokens=100, output_tokens=50)
        evt = TaskCompleteEvent(
            task_id="t-004",
            worker_name="coder",
            status="completed",
            turns_completed=3,
            usage=usage,
        )
        assert evt.usage is usage
        assert evt.usage.total == 150

    def test_frozen(self) -> None:
        evt = TaskCompleteEvent(
            task_id="t",
            worker_name="w",
            status="completed",
            turns_completed=0,
            usage=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.status = "failed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _ALL_EVENT_TYPES registration
# ---------------------------------------------------------------------------


class TestAllEventTypes:
    def test_worker_event_in_all_types(self) -> None:
        assert WorkerEvent in _ALL_EVENT_TYPES

    def test_task_dispatch_event_in_all_types(self) -> None:
        assert TaskDispatchEvent in _ALL_EVENT_TYPES

    def test_task_complete_event_in_all_types(self) -> None:
        assert TaskCompleteEvent in _ALL_EVENT_TYPES


# ---------------------------------------------------------------------------
# _setup_event_bubble
# ---------------------------------------------------------------------------


def _make_mock_agent() -> object:
    """Build a minimal mock agent with an event_bus attribute."""
    class MockAgent:
        def __init__(self) -> None:
            self._event_bus = EventBus()

        @property
        def event_bus(self) -> EventBus:
            return self._event_bus

    return MockAgent()


class TestSetupEventBubble:
    def test_worker_events_bubble_to_orchestrator(self) -> None:
        """Events emitted on worker bubble to orchestrator wrapped in WorkerEvent."""
        worker = _make_mock_agent()
        orchestrator = _make_mock_agent()

        _setup_event_bubble(
            worker_agent=worker,
            orchestrator=orchestrator,
            worker_name="coder",
            task_id="task-1",
            depth=1,
        )

        received: list[WorkerEvent] = []
        orchestrator._event_bus.subscribe(WorkerEvent, received.append)

        inner = TurnCompleteEvent(turn_index=0, stop_reason="end_turn", tool_call_count=2)
        worker._event_bus.emit(inner)

        assert len(received) == 1
        bubble = received[0]
        assert isinstance(bubble, WorkerEvent)
        assert bubble.worker_name == "coder"
        assert bubble.task_id == "task-1"
        assert bubble.depth == 1
        assert bubble.inner is inner

    def test_bubble_wraps_all_event_types(self) -> None:
        """A ProviderResponseEvent from worker also bubbles as WorkerEvent."""
        worker = _make_mock_agent()
        orchestrator = _make_mock_agent()

        _setup_event_bubble(
            worker_agent=worker,
            orchestrator=orchestrator,
            worker_name="analyst",
            task_id="task-2",
            depth=0,
        )

        received: list[WorkerEvent] = []
        orchestrator._event_bus.subscribe(WorkerEvent, received.append)

        inner = ProviderResponseEvent(
            content=(),
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=5,
            turn=1,
        )
        worker._event_bus.emit(inner)

        assert len(received) == 1
        assert isinstance(received[0].inner, ProviderResponseEvent)

    def test_subscribe_all_includes_worker_event(self) -> None:
        """EventBus.subscribe_all should subscribe to WorkerEvent."""
        bus = EventBus()
        received: list[object] = []
        bus.subscribe_all(received.append)

        inner = TurnCompleteEvent(turn_index=0, stop_reason="end_turn", tool_call_count=0)
        evt = WorkerEvent(worker_name="w", task_id="t", depth=0, inner=inner)
        bus.emit(evt)

        assert len(received) == 1
        assert received[0] is evt
