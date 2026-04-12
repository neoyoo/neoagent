from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from neoagent.eval.usage import ModelUsage, UsageTracker
from neoagent.events import EventBus, ProviderResponseEvent


def _make_response_event(input_tokens: int, output_tokens: int, turn: int = 1) -> ProviderResponseEvent:
    return ProviderResponseEvent(
        content=(),
        stop_reason="end_turn",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        turn=turn,
    )


# ── UsageTracker state tests ──────────────────────────────────────────────────

def test_usage_tracker_initial_state():
    tracker = UsageTracker()
    assert tracker.total_input_tokens == 0
    assert tracker.total_output_tokens == 0
    assert tracker.per_model_usage == {}
    assert tracker._current_model == "unknown"


def test_usage_tracker_records_tokens():
    tracker = UsageTracker()
    tracker._current_model = "claude-3-5-sonnet"
    event = _make_response_event(100, 50)
    tracker._handle_response(event)
    assert tracker.total_input_tokens == 100
    assert tracker.total_output_tokens == 50


def test_usage_tracker_accumulates_multiple_events():
    tracker = UsageTracker()
    tracker._current_model = "gpt-4o"
    tracker._handle_response(_make_response_event(100, 50, turn=1))
    tracker._handle_response(_make_response_event(200, 80, turn=2))
    assert tracker.total_input_tokens == 300
    assert tracker.total_output_tokens == 130


def test_usage_tracker_per_model_usage():
    tracker = UsageTracker()
    tracker._current_model = "claude-3-5-sonnet"
    tracker._handle_response(_make_response_event(100, 50))
    tracker._current_model = "gpt-4o"
    tracker._handle_response(_make_response_event(200, 80))

    per_model = tracker.per_model_usage
    assert "claude-3-5-sonnet" in per_model
    assert "gpt-4o" in per_model
    assert per_model["claude-3-5-sonnet"].input_tokens == 100
    assert per_model["claude-3-5-sonnet"].output_tokens == 50
    assert per_model["gpt-4o"].input_tokens == 200
    assert per_model["gpt-4o"].output_tokens == 80


def test_usage_tracker_unknown_model_records():
    tracker = UsageTracker()
    # _current_model starts as "unknown"
    tracker._handle_response(_make_response_event(10, 5))
    per_model = tracker.per_model_usage
    assert "unknown" in per_model
    assert per_model["unknown"].input_tokens == 10


def test_usage_tracker_request_count():
    tracker = UsageTracker()
    tracker._current_model = "claude-3-5-sonnet"
    tracker._handle_response(_make_response_event(100, 50, turn=1))
    tracker._handle_response(_make_response_event(200, 80, turn=2))
    tracker._handle_response(_make_response_event(150, 60, turn=3))
    per_model = tracker.per_model_usage
    assert per_model["claude-3-5-sonnet"].request_count == 3


def test_usage_tracker_reset():
    tracker = UsageTracker()
    tracker._current_model = "gpt-4o"
    tracker._handle_response(_make_response_event(100, 50))
    assert tracker.total_input_tokens == 100

    tracker.reset()
    assert tracker.total_input_tokens == 0
    assert tracker.total_output_tokens == 0
    assert tracker.per_model_usage == {}


# ── ModelUsage dataclass tests ─────────────────────────────────────────────────

def test_model_usage_dataclass_fields():
    usage = ModelUsage(model="test-model")
    assert usage.model == "test-model"
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.request_count == 0

    usage2 = ModelUsage(model="m", input_tokens=10, output_tokens=5, request_count=2)
    assert usage2.input_tokens == 10
    assert usage2.output_tokens == 5
    assert usage2.request_count == 2


# ── NeoAgent.enable_usage_tracking() tests ─────────────────────────────────────

def test_enable_usage_tracking_returns_tracker():
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig

    config = NeoAgentConfig(api_key="test-key", provider="anthropic", model="claude-3-5-sonnet")
    agent = NeoAgent(config)
    tracker = agent.enable_usage_tracking()
    assert isinstance(tracker, UsageTracker)


def test_enable_usage_tracking_subscribes_to_event_bus():
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig

    config = NeoAgentConfig(api_key="test-key", provider="anthropic", model="claude-3-5-haiku")
    agent = NeoAgent(config)
    tracker = agent.enable_usage_tracking()

    # Verify model name was set from provider
    assert tracker._current_model == "claude-3-5-haiku"

    # Emit a ProviderResponseEvent and verify tracker picks it up
    event = _make_response_event(123, 45)
    agent.event_bus.emit(event)

    assert tracker.total_input_tokens == 123
    assert tracker.total_output_tokens == 45
    per_model = tracker.per_model_usage
    assert "claude-3-5-haiku" in per_model
    assert per_model["claude-3-5-haiku"].request_count == 1
