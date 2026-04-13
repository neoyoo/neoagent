from __future__ import annotations
import pytest
from neoagent.eval.metrics import MetricsCollector, TurnMetrics, SessionMetrics
from neoagent.events import (
    EventBus,
    ProviderRequestEvent,
    ProviderResponseEvent,
    ToolCallEvent,
    TurnCompleteEvent,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _req(turn: int = 0) -> ProviderRequestEvent:
    return ProviderRequestEvent(
        system="sys",
        messages=(),
        tools=(),
        turn=turn,
    )


def _resp(turn: int = 0, input_tokens: int = 100, output_tokens: int = 50) -> ProviderResponseEvent:
    return ProviderResponseEvent(
        content=(),
        stop_reason="end_turn",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        turn=turn,
    )


def _tool(name: str = "bash", call_id: str = "c1") -> ToolCallEvent:
    return ToolCallEvent(name=name, input_data={}, call_id=call_id)


def _turn_complete(turn_index: int = 0, tool_call_count: int = 0) -> TurnCompleteEvent:
    return TurnCompleteEvent(
        turn_index=turn_index,
        stop_reason="end_turn",
        tool_call_count=tool_call_count,
    )


# ── TurnMetrics dataclass ─────────────────────────────────────────────────────

def test_turn_metrics_defaults():
    tm = TurnMetrics(turn_index=0)
    assert tm.turn_index == 0
    assert tm.input_tokens == 0
    assert tm.output_tokens == 0
    assert tm.latency_ms == 0.0
    assert tm.tool_call_count == 0
    assert tm.tool_names == []


def test_turn_metrics_fields():
    tm = TurnMetrics(
        turn_index=2,
        input_tokens=200,
        output_tokens=80,
        latency_ms=123.4,
        tool_call_count=3,
        tool_names=["bash", "read", "write"],
    )
    assert tm.turn_index == 2
    assert tm.input_tokens == 200
    assert tm.output_tokens == 80
    assert tm.latency_ms == 123.4
    assert tm.tool_call_count == 3
    assert tm.tool_names == ["bash", "read", "write"]


# ── SessionMetrics properties ─────────────────────────────────────────────────

def test_session_metrics_empty():
    sm = SessionMetrics()
    assert sm.total_input_tokens == 0
    assert sm.total_output_tokens == 0
    assert sm.duration_ms == 0.0
    assert sm.tool_calls == 0
    assert sm.turns == []


def test_session_metrics_aggregates():
    t0 = TurnMetrics(turn_index=0, input_tokens=100, output_tokens=50, latency_ms=200.0, tool_call_count=1, tool_names=["bash"])
    t1 = TurnMetrics(turn_index=1, input_tokens=150, output_tokens=70, latency_ms=300.0, tool_call_count=2, tool_names=["read", "write"])
    sm = SessionMetrics(turns=[t0, t1])
    assert sm.total_input_tokens == 250
    assert sm.total_output_tokens == 120
    assert sm.duration_ms == pytest.approx(500.0)
    assert sm.tool_calls == 3


# ── MetricsCollector initial state ────────────────────────────────────────────

def test_collector_initial_state():
    collector = MetricsCollector()
    assert collector.get_turn_metrics() == []
    sm = collector.get_session_metrics()
    assert sm.turns == []
    assert sm.total_input_tokens == 0


# ── Single turn, full flow ────────────────────────────────────────────────────

def test_collector_single_turn_tokens():
    collector = MetricsCollector()
    collector._on_provider_request(_req(turn=0))
    collector._on_provider_response(_resp(turn=0, input_tokens=100, output_tokens=50))
    collector._on_turn_complete(_turn_complete(turn_index=0))

    turns = collector.get_turn_metrics()
    assert len(turns) == 1
    assert turns[0].turn_index == 0
    assert turns[0].input_tokens == 100
    assert turns[0].output_tokens == 50


def test_collector_latency_measured():
    collector = MetricsCollector()
    collector._on_provider_request(_req(turn=0))
    # Simulate some time passing by setting timestamps manually
    collector._pending[0]["request_time"] = 1000.0
    collector._pending[0]["response_time"] = 1000.5  # 500ms later
    collector._on_turn_complete(_turn_complete(turn_index=0))

    turns = collector.get_turn_metrics()
    assert turns[0].latency_ms == pytest.approx(500.0)


def test_collector_latency_zero_without_request():
    """If no request event was received, latency_ms defaults to 0.0."""
    collector = MetricsCollector()
    collector._on_provider_response(_resp(turn=0))
    collector._on_turn_complete(_turn_complete(turn_index=0))

    turns = collector.get_turn_metrics()
    assert turns[0].latency_ms == 0.0


# ── Tool call tracking ────────────────────────────────────────────────────────

def test_collector_tool_calls_assigned_to_latest_turn():
    collector = MetricsCollector()
    collector._on_provider_request(_req(turn=0))
    collector._on_tool_call(_tool("bash", "c1"))
    collector._on_tool_call(_tool("read", "c2"))
    collector._on_turn_complete(_turn_complete(turn_index=0))

    turns = collector.get_turn_metrics()
    assert turns[0].tool_call_count == 2
    assert turns[0].tool_names == ["bash", "read"]


# ── Multi-turn ────────────────────────────────────────────────────────────────

def test_collector_multi_turn():
    collector = MetricsCollector()

    # Turn 0
    collector._on_provider_request(_req(turn=0))
    collector._on_provider_response(_resp(turn=0, input_tokens=100, output_tokens=50))
    collector._on_tool_call(_tool("bash", "c1"))
    collector._on_turn_complete(_turn_complete(turn_index=0, tool_call_count=1))

    # Turn 1
    collector._on_provider_request(_req(turn=1))
    collector._on_provider_response(_resp(turn=1, input_tokens=200, output_tokens=80))
    collector._on_turn_complete(_turn_complete(turn_index=1))

    turns = collector.get_turn_metrics()
    assert len(turns) == 2
    assert turns[0].turn_index == 0
    assert turns[0].input_tokens == 100
    assert turns[0].tool_call_count == 1
    assert turns[1].turn_index == 1
    assert turns[1].input_tokens == 200
    assert turns[1].tool_call_count == 0


def test_collector_get_session_metrics_multi_turn():
    collector = MetricsCollector()

    collector._on_provider_request(_req(turn=0))
    collector._on_provider_response(_resp(turn=0, input_tokens=100, output_tokens=50))
    collector._on_tool_call(_tool("bash", "c1"))
    collector._on_turn_complete(_turn_complete(turn_index=0, tool_call_count=1))

    collector._on_provider_request(_req(turn=1))
    collector._on_provider_response(_resp(turn=1, input_tokens=200, output_tokens=80))
    collector._on_turn_complete(_turn_complete(turn_index=1))

    sm = collector.get_session_metrics()
    assert sm.total_input_tokens == 300
    assert sm.total_output_tokens == 130
    assert sm.tool_calls == 1


# ── reset() ───────────────────────────────────────────────────────────────────

def test_collector_reset():
    collector = MetricsCollector()
    collector._on_provider_request(_req(turn=0))
    collector._on_provider_response(_resp(turn=0, input_tokens=100, output_tokens=50))
    collector._on_turn_complete(_turn_complete(turn_index=0))

    assert len(collector.get_turn_metrics()) == 1
    collector.reset()
    assert collector.get_turn_metrics() == []
    assert collector.get_session_metrics().total_input_tokens == 0


# ── NeoAgent.enable_metrics() ─────────────────────────────────────────────────

def test_enable_metrics_returns_collector():
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig

    config = NeoAgentConfig(api_key="test-key", provider="anthropic", model="claude-3-5-sonnet")
    agent = NeoAgent(config)
    collector = agent.enable_metrics()
    assert isinstance(collector, MetricsCollector)


def test_collector_max_turns_cap():
    """_completed_turns list is trimmed to max_turns when exceeded."""
    collector = MetricsCollector(max_turns=5)

    # Simulate 8 turns
    for i in range(8):
        collector._on_provider_request(_req(turn=i))
        collector._on_provider_response(_resp(turn=i, input_tokens=10, output_tokens=5))
        collector._on_turn_complete(_turn_complete(turn_index=i))

    turns = collector.get_turn_metrics()
    # Only the most recent 5 should be retained
    assert len(turns) == 5
    # The newest turn index should be 7 (last emitted)
    assert turns[-1].turn_index == 7
    # The oldest retained turn index should be 3 (8 - 5 = 3)
    assert turns[0].turn_index == 3


def test_collector_max_turns_default_large():
    """Default max_turns=10000 does not trim small turn counts."""
    collector = MetricsCollector()  # default max_turns=10000

    for i in range(50):
        collector._on_provider_request(_req(turn=i))
        collector._on_provider_response(_resp(turn=i))
        collector._on_turn_complete(_turn_complete(turn_index=i))

    assert len(collector.get_turn_metrics()) == 50


def test_enable_metrics_subscribes_to_events():
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig

    config = NeoAgentConfig(api_key="test-key", provider="anthropic", model="claude-3-5-haiku")
    agent = NeoAgent(config)
    collector = agent.enable_metrics()

    # Emit events through the agent's event bus
    agent.event_bus.emit(_req(turn=0))
    agent.event_bus.emit(_resp(turn=0, input_tokens=150, output_tokens=60))
    agent.event_bus.emit(_tool("bash", "c1"))
    agent.event_bus.emit(_turn_complete(turn_index=0, tool_call_count=1))

    turns = collector.get_turn_metrics()
    assert len(turns) == 1
    assert turns[0].input_tokens == 150
    assert turns[0].output_tokens == 60
    assert turns[0].tool_call_count == 1
    assert turns[0].tool_names == ["bash"]
