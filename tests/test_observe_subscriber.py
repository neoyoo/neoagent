from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from neoagent.events import (
    EventBus, ProviderRequestEvent, ProviderResponseEvent,
    ToolCallEvent, ToolResultEvent,
    CompressCheckEvent, CompressDoneEvent, CompressFallbackEvent,
    MemoryExtractEvent, SkillChangeEvent,
)
from neoagent.observe_subscriber import ObserverSubscriber


def _make_observer():
    obs = MagicMock()
    return obs


def test_tool_call_event_routes_to_observer():
    bus = EventBus()
    obs = _make_observer()
    ObserverSubscriber(obs, bus)
    bus.emit(ToolCallEvent(name="bash", input_data={"cmd": "ls"}, call_id="c1"))
    obs.on_tool_call.assert_called_once_with("bash", {"cmd": "ls"})


def test_tool_result_event_routes_to_observer():
    bus = EventBus()
    obs = _make_observer()
    ObserverSubscriber(obs, bus)
    bus.emit(ToolResultEvent(name="bash", call_id="c1", output="a" * 400, is_error=False))
    obs.on_tool_result.assert_called_once()
    name, output, is_error = obs.on_tool_result.call_args[0]
    assert len(output) <= 300  # truncated by subscriber


def test_provider_request_routes_to_observer():
    bus = EventBus()
    obs = _make_observer()
    ObserverSubscriber(obs, bus)
    bus.emit(ProviderRequestEvent(system="sys", messages=(1,), tools=(2,), turn=0))
    obs.on_provider_request.assert_called_once_with("sys", [1], [2], 0)


def test_provider_response_routes_to_observer():
    bus = EventBus()
    obs = _make_observer()
    ObserverSubscriber(obs, bus)
    bus.emit(ProviderResponseEvent(content=(1,), stop_reason="end_turn",
                                   input_tokens=10, output_tokens=5, turn=0))
    obs.on_provider_response.assert_called_once_with([1], "end_turn", 10, 5, 0)


def test_compress_check_routes_to_observer():
    bus = EventBus()
    obs = _make_observer()
    ObserverSubscriber(obs, bus)
    bus.emit(CompressCheckEvent(msg_tokens=100, tool_tokens=20, budget=500, should_compress=True))
    obs.on_compress_check.assert_called_once_with(100, 20, 500, True)


def test_compress_done_routes_to_observer():
    bus = EventBus()
    obs = _make_observer()
    ObserverSubscriber(obs, bus)
    bus.emit(CompressDoneEvent(summary="new", previous_summary="old"))
    obs.on_compress_done.assert_called_once_with("new", "old")


def test_compress_fallback_routes_to_observer():
    bus = EventBus()
    obs = _make_observer()
    ObserverSubscriber(obs, bus)
    bus.emit(CompressFallbackEvent(reason="max failures"))
    obs.on_compress_fallback.assert_called_once_with("max failures")


def test_memory_extract_triggered_routes_done():
    bus = EventBus()
    obs = _make_observer()
    ObserverSubscriber(obs, bus)
    bus.emit(MemoryExtractEvent(triggered=True, tool_calls=3, token_delta=100,
                                 items_stored=2, filenames=("a.md", "b.md")))
    obs.on_memory_extract_done.assert_called_once_with(2, ["a.md", "b.md"])
    obs.on_memory_extract_skip.assert_not_called()


def test_memory_extract_not_triggered_routes_skip():
    bus = EventBus()
    obs = _make_observer()
    ObserverSubscriber(obs, bus)
    bus.emit(MemoryExtractEvent(triggered=False, tool_calls=1, token_delta=50))
    obs.on_memory_extract_skip.assert_called_once_with(1, 50)
    obs.on_memory_extract_done.assert_not_called()


def test_skill_activate_routes_to_observer():
    bus = EventBus()
    obs = _make_observer()
    ObserverSubscriber(obs, bus)
    bus.emit(SkillChangeEvent(name="memory", active=True))
    obs.on_skill_activate.assert_called_once_with("memory")


def test_skill_deactivate_routes_to_observer():
    bus = EventBus()
    obs = _make_observer()
    ObserverSubscriber(obs, bus)
    bus.emit(SkillChangeEvent(name="memory", active=False))
    obs.on_skill_deactivate.assert_called_once_with("memory")
