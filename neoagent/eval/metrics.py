from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neoagent.events import (
        ProviderRequestEvent,
        ProviderResponseEvent,
        ToolCallEvent,
        TurnCompleteEvent,
    )


@dataclass
class TurnMetrics:
    """Per-turn metrics collected during a single query-loop turn."""
    turn_index: int
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    tool_call_count: int = 0
    tool_names: list[str] = field(default_factory=list)


@dataclass
class SessionMetrics:
    """Aggregated metrics for an entire session (all turns)."""
    turns: list[TurnMetrics] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def duration_ms(self) -> float:
        return sum(t.latency_ms for t in self.turns)

    @property
    def tool_calls(self) -> int:
        return sum(t.tool_call_count for t in self.turns)


class MetricsCollector:
    """Collects per-turn and session-level metrics via EventBus subscriptions.

    Subscribe to the four key events::

        collector = MetricsCollector()
        bus.subscribe(ProviderRequestEvent, collector._on_provider_request)
        bus.subscribe(ProviderResponseEvent, collector._on_provider_response)
        bus.subscribe(ToolCallEvent, collector._on_tool_call)
        bus.subscribe(TurnCompleteEvent, collector._on_turn_complete)

    Or simply call ``agent.enable_metrics()`` which sets this up automatically.
    """

    def __init__(self, max_turns: int = 10000) -> None:
        self._max_turns = max_turns
        # _pending[turn] accumulates intermediate data before TurnCompleteEvent
        self._pending: dict[int, dict] = {}
        self._completed_turns: list[TurnMetrics] = []

    def _get_or_create_pending(self, turn: int) -> dict:
        if turn not in self._pending:
            self._pending[turn] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "request_time": None,
                "response_time": None,
                "tool_call_count": 0,
                "tool_names": [],
            }
        return self._pending[turn]

    def _on_provider_request(self, event: "ProviderRequestEvent") -> None:
        pending = self._get_or_create_pending(event.turn)
        pending["request_time"] = time.monotonic()

    def _on_provider_response(self, event: "ProviderResponseEvent") -> None:
        pending = self._get_or_create_pending(event.turn)
        pending["response_time"] = time.monotonic()
        pending["input_tokens"] += event.input_tokens
        pending["output_tokens"] += event.output_tokens

    def _on_tool_call(self, event: "ToolCallEvent") -> None:
        # ToolCallEvent has no turn; assign to the most recent pending turn
        if not self._pending:
            # No turn started yet; create a bucket for turn 0
            turn = 0
        else:
            turn = max(self._pending.keys())
        pending = self._get_or_create_pending(turn)
        pending["tool_call_count"] += 1
        pending["tool_names"].append(event.name)

    def _on_turn_complete(self, event: "TurnCompleteEvent") -> None:
        turn_index = event.turn_index
        # Use pending data for this turn if available; fall back to the latest
        pending = self._pending.pop(turn_index, None)
        if pending is None and self._pending:
            # Fallback: grab the only remaining pending bucket
            oldest_key = min(self._pending.keys())
            pending = self._pending.pop(oldest_key)
        if pending is None:
            pending = {
                "input_tokens": 0,
                "output_tokens": 0,
                "request_time": None,
                "response_time": None,
                "tool_call_count": 0,
                "tool_names": [],
            }

        request_time = pending["request_time"]
        response_time = pending["response_time"]
        if request_time is not None and response_time is not None:
            latency_ms = (response_time - request_time) * 1000.0
        else:
            latency_ms = 0.0

        turn_metrics = TurnMetrics(
            turn_index=turn_index,
            input_tokens=pending["input_tokens"],
            output_tokens=pending["output_tokens"],
            latency_ms=latency_ms,
            tool_call_count=pending["tool_call_count"],
            tool_names=list(pending["tool_names"]),
        )
        self._completed_turns.append(turn_metrics)
        if len(self._completed_turns) > self._max_turns:
            self._completed_turns = self._completed_turns[-self._max_turns :]

    def get_turn_metrics(self) -> list[TurnMetrics]:
        """Return a copy of the list of completed TurnMetrics, in order."""
        return list(self._completed_turns)

    def get_session_metrics(self) -> SessionMetrics:
        """Return a SessionMetrics aggregating all completed turns."""
        return SessionMetrics(turns=list(self._completed_turns))

    def reset(self) -> None:
        """Clear all collected data."""
        self._pending.clear()
        self._completed_turns.clear()
