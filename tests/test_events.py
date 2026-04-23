from __future__ import annotations
import pytest
from neoagent.events import (
    EventBus, Event,
    ProviderRequestEvent, ProviderResponseEvent,
    ToolCallEvent, ToolResultEvent,
    CompressCheckEvent, CompressDoneEvent, CompressFallbackEvent,
    MemoryExtractEvent, SkillChangeEvent, TurnCompleteEvent,
)


# ── Event immutability ────────────────────────────────────────────────────────

def test_events_are_frozen():
    e = ToolCallEvent(name="bash", input_data={"command": "ls"}, call_id="c1")
    with pytest.raises((AttributeError, TypeError)):
        e.name = "other"  # type: ignore


def test_tool_call_event_fields():
    e = ToolCallEvent(name="bash", input_data={"cmd": "ls"}, call_id="abc")
    assert e.name == "bash"
    assert e.call_id == "abc"


def test_tool_result_event_fields():
    e = ToolResultEvent(name="bash", call_id="c1", output="hello", is_error=False)
    assert e.output == "hello"
    assert not e.is_error


def test_provider_request_event_tuple_args():
    e = ProviderRequestEvent(system="sys", messages=(1, 2), tools=(3,), turn=0)
    assert e.messages == (1, 2)


def test_compress_check_event():
    e = CompressCheckEvent(msg_tokens=100, tool_tokens=20, budget=1000, should_compress=False)
    assert not e.should_compress


def test_compress_done_event_no_previous():
    e = CompressDoneEvent(summary="new", previous_summary=None)
    assert e.previous_summary is None


def test_memory_extract_event_defaults():
    e = MemoryExtractEvent(triggered=False, tool_calls=2, token_delta=500)
    assert e.items_stored == 0
    assert e.filenames == ()


def test_turn_complete_event():
    e = TurnCompleteEvent(turn_index=3, stop_reason="end_turn", tool_call_count=0)
    assert e.turn_index == 3


# ── EventBus: subscribe + emit ────────────────────────────────────────────────

def test_subscribe_and_emit():
    bus = EventBus()
    received = []
    bus.subscribe(ToolCallEvent, lambda e: received.append(e))
    e = ToolCallEvent(name="bash", input_data={}, call_id="x")
    bus.emit(e)
    assert len(received) == 1
    assert received[0] is e


def test_emit_only_matching_type():
    bus = EventBus()
    received = []
    bus.subscribe(ToolCallEvent, lambda e: received.append(e))
    bus.emit(ToolResultEvent(name="bash", call_id="x", output="ok", is_error=False))
    assert received == []


def test_multiple_handlers_same_type():
    bus = EventBus()
    a, b = [], []
    bus.subscribe(ToolCallEvent, lambda e: a.append(1))
    bus.subscribe(ToolCallEvent, lambda e: b.append(2))
    bus.emit(ToolCallEvent(name="x", input_data={}, call_id="y"))
    assert a == [1]
    assert b == [2]


def test_unsubscribe_removes_handler():
    bus = EventBus()
    received = []
    handler = lambda e: received.append(e)
    bus.subscribe(ToolCallEvent, handler)
    bus.unsubscribe(ToolCallEvent, handler)
    bus.emit(ToolCallEvent(name="x", input_data={}, call_id="z"))
    assert received == []


def test_unsubscribe_missing_handler_no_error():
    bus = EventBus()
    handler = lambda e: None
    bus.unsubscribe(ToolCallEvent, handler)  # not subscribed — should not raise


# ── emit-never-throws semantics ───────────────────────────────────────────────

def test_handler_exception_does_not_crash_bus():
    bus = EventBus()
    results = []

    def bad_handler(e):
        raise RuntimeError("handler blew up")

    def good_handler(e):
        results.append("ok")

    bus.subscribe(ToolCallEvent, bad_handler)
    bus.subscribe(ToolCallEvent, good_handler)
    # Must not raise, good_handler must still run
    bus.emit(ToolCallEvent(name="x", input_data={}, call_id="z"))
    assert results == ["ok"]


def test_handler_exception_logged(caplog):
    import logging
    bus = EventBus()
    bus.subscribe(ToolCallEvent, lambda e: 1 / 0)
    with caplog.at_level(logging.ERROR):
        bus.emit(ToolCallEvent(name="x", input_data={}, call_id="z"))
    assert any("handler error" in r.message.lower() or "ZeroDivisionError" in r.message
               for r in caplog.records)


# ── subscribe_all ─────────────────────────────────────────────────────────────

def test_subscribe_all_receives_all_event_types():
    bus = EventBus()
    received_types = []
    bus.subscribe_all(lambda e: received_types.append(type(e).__name__))

    bus.emit(ToolCallEvent(name="x", input_data={}, call_id="a"))
    bus.emit(ToolResultEvent(name="x", call_id="a", output="ok", is_error=False))
    bus.emit(TurnCompleteEvent(turn_index=0, stop_reason="end_turn", tool_call_count=0))

    assert "ToolCallEvent" in received_types
    assert "ToolResultEvent" in received_types
    assert "TurnCompleteEvent" in received_types


# ── v2 Events: construction + EventBus round-trip ────────────────────────────

class TestMessageCreatedEvent:
    def test_construct(self):
        from neoagent.events import MessageCreatedEvent
        e = MessageCreatedEvent(
            session_id="s1",
            msg_id="m5",
            turn=2,
            role="user",
            source_type="user_input",
            content=[{"type": "text", "text": "hello"}],
        )
        assert e.session_id == "s1"
        assert e.msg_id == "m5"
        assert e.turn == 2
        assert e.role == "user"
        assert e.source_type == "user_input"
        assert e.content == [{"type": "text", "text": "hello"}]

    def test_source_type_variants(self):
        from neoagent.events import MessageCreatedEvent
        for st in ("user_input", "assistant_reply", "tool_result",
                   "system_injected_compression", "system_injected_memory",
                   "system_injected_recall"):
            e = MessageCreatedEvent(session_id="s1", msg_id="m1", turn=0,
                                    role="user", source_type=st, content=[])
            assert e.source_type == st

    def test_eventbus_roundtrip(self):
        from neoagent.events import MessageCreatedEvent
        bus = EventBus()
        received = []
        bus.subscribe(MessageCreatedEvent, lambda e: received.append(e))
        ev = MessageCreatedEvent(session_id="s1", msg_id="m1", turn=0,
                                 role="assistant", source_type="assistant_reply", content=[])
        bus.emit(ev)
        assert len(received) == 1
        assert received[0] is ev

    def test_is_frozen(self):
        from neoagent.events import MessageCreatedEvent
        e = MessageCreatedEvent(session_id="s1", msg_id="m1", turn=0,
                                role="user", source_type="user_input", content=[])
        with pytest.raises((AttributeError, TypeError)):
            e.session_id = "other"  # type: ignore


class TestToolResultPersistedEvent:
    def test_construct(self):
        from neoagent.events import ToolResultPersistedEvent
        e = ToolResultPersistedEvent(
            session_id="s1",
            tool_use_id="tu1",
            turn=3,
            tool_name="bash",
            output="ok",
            size_bytes=2,
            is_error=False,
        )
        assert e.session_id == "s1"
        assert e.tool_use_id == "tu1"
        assert e.turn == 3
        assert e.tool_name == "bash"
        assert e.output == "ok"
        assert e.size_bytes == 2
        assert e.is_error is False

    def test_eventbus_roundtrip(self):
        from neoagent.events import ToolResultPersistedEvent
        bus = EventBus()
        received = []
        bus.subscribe(ToolResultPersistedEvent, lambda e: received.append(e))
        ev = ToolResultPersistedEvent(session_id="s1", tool_use_id="tu1", turn=0,
                                      tool_name="bash", output="out", size_bytes=3, is_error=False)
        bus.emit(ev)
        assert len(received) == 1

    def test_is_frozen(self):
        from neoagent.events import ToolResultPersistedEvent
        e = ToolResultPersistedEvent(session_id="s1", tool_use_id="tu1", turn=0,
                                     tool_name="bash", output="out", size_bytes=3, is_error=False)
        with pytest.raises((AttributeError, TypeError)):
            e.session_id = "x"  # type: ignore


class TestBatchCreatedEvent:
    def test_construct(self):
        from neoagent.events import BatchCreatedEvent
        from neoagent.v2.schema import BatchMember
        members = [BatchMember(id="m1", role="user", preview="hello")]
        e = BatchCreatedEvent(
            session_id="s1",
            batch_id="cm_1",
            turns_from=0,
            turns_to=5,
            summary="batch summary",
            members=members,
        )
        assert e.session_id == "s1"
        assert e.batch_id == "cm_1"
        assert e.turns_from == 0
        assert e.turns_to == 5
        assert e.summary == "batch summary"
        assert len(e.members) == 1

    def test_eventbus_roundtrip(self):
        from neoagent.events import BatchCreatedEvent
        bus = EventBus()
        received = []
        bus.subscribe(BatchCreatedEvent, lambda e: received.append(e))
        ev = BatchCreatedEvent(session_id="s1", batch_id="cm_1", turns_from=0,
                               turns_to=3, summary="s", members=[])
        bus.emit(ev)
        assert len(received) == 1

    def test_is_frozen(self):
        from neoagent.events import BatchCreatedEvent
        e = BatchCreatedEvent(session_id="s1", batch_id="cm_1", turns_from=0,
                              turns_to=3, summary="s", members=[])
        with pytest.raises((AttributeError, TypeError)):
            e.session_id = "x"  # type: ignore


class TestWorkingMemoryUpdatedEvent:
    def test_construct(self):
        from neoagent.events import WorkingMemoryUpdatedEvent
        e = WorkingMemoryUpdatedEvent(
            session_id="s1",
            version=2,
            at_turn=4,
            wm_json={"progress": "explore"},
            updated_by="llm_tool",
        )
        assert e.session_id == "s1"
        assert e.version == 2
        assert e.at_turn == 4
        assert e.wm_json == {"progress": "explore"}
        assert e.updated_by == "llm_tool"

    def test_eventbus_roundtrip(self):
        from neoagent.events import WorkingMemoryUpdatedEvent
        bus = EventBus()
        received = []
        bus.subscribe(WorkingMemoryUpdatedEvent, lambda e: received.append(e))
        ev = WorkingMemoryUpdatedEvent(session_id="s1", version=1, at_turn=0,
                                       wm_json={}, updated_by="framework_init")
        bus.emit(ev)
        assert len(received) == 1

    def test_is_frozen(self):
        from neoagent.events import WorkingMemoryUpdatedEvent
        e = WorkingMemoryUpdatedEvent(session_id="s1", version=1, at_turn=0,
                                      wm_json={}, updated_by="framework_init")
        with pytest.raises((AttributeError, TypeError)):
            e.session_id = "x"  # type: ignore


class TestCompressionFailedEvent:
    def test_construct(self):
        from neoagent.events import CompressionFailedEvent
        e = CompressionFailedEvent(
            session_id="s1",
            reason="invalid json",
            retry_count=3,
        )
        assert e.session_id == "s1"
        assert e.reason == "invalid json"
        assert e.retry_count == 3

    def test_eventbus_roundtrip(self):
        from neoagent.events import CompressionFailedEvent
        bus = EventBus()
        received = []
        bus.subscribe(CompressionFailedEvent, lambda e: received.append(e))
        ev = CompressionFailedEvent(session_id="s1", reason="timeout", retry_count=3)
        bus.emit(ev)
        assert len(received) == 1
        assert received[0].reason == "timeout"

    def test_is_frozen(self):
        from neoagent.events import CompressionFailedEvent
        e = CompressionFailedEvent(session_id="s1", reason="x", retry_count=0)
        with pytest.raises((AttributeError, TypeError)):
            e.session_id = "y"  # type: ignore
