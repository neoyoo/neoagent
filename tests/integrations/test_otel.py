"""Tests for OtelSubscriber — OpenTelemetry integration."""
from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from neoagent.core.types import Message, TextBlock
from neoagent.events import (
    BatchCreatedEvent,
    EventBus,
    MemoryExtractEvent,
    ProviderRequestEvent,
    ProviderResponseEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from neoagent.integrations.otel import OtelSubscriber


@pytest.fixture
def span_exporter():
    """Return (exporter, tracer) for a fresh TracerProvider per test.

    We do NOT call trace.set_tracer_provider() because OTEL only allows one
    global provider and caches tracer instances — subsequent overrides are
    silently ignored. Instead we pass the tracer directly to OtelSubscriber
    to keep each test isolated.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("neoagent")
    yield exporter, tracer
    exporter.clear()


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_provider_request(turn: int = 0) -> ProviderRequestEvent:
    msg = Message(role="user", content="hello")
    tool = {"name": "search", "description": "web search", "input_schema": {}}
    return ProviderRequestEvent(
        system="You are helpful.",
        messages=(msg,),
        tools=(tool,),
        turn=turn,
    )


def _make_provider_response(turn: int = 0) -> ProviderResponseEvent:
    block = TextBlock(text="Hello!")
    return ProviderResponseEvent(
        content=(block,),
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=20,
        turn=turn,
    )


# ── tests ──────────────────────────────────────────────────────────────────────

def test_subscriber_attach_detach():
    """attach/detach should not raise; repeated detach is safe."""
    bus = EventBus()
    sub = OtelSubscriber()
    sub.attach(bus)
    sub.detach(bus)
    sub.detach(bus)  # second detach — must be safe (no-op)


def test_provider_call_creates_llm_span(span_exporter):
    """ProviderRequestEvent + ProviderResponseEvent produce a completed llm.call span."""
    exporter, tracer = span_exporter
    bus = EventBus()
    sub = OtelSubscriber(tracer=tracer)
    sub.attach(bus)

    bus.emit(_make_provider_request(turn=1))
    bus.emit(_make_provider_response(turn=1))

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.name == "llm.call"]
    assert len(llm_spans) == 1, f"Expected 1 llm.call span, got {len(llm_spans)}"

    attrs = llm_spans[0].attributes
    assert attrs["gen_ai.request.model"] is not None
    assert attrs["gen_ai.response.finish_reason"] == "end_turn"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 20
    assert attrs["gen_ai.prompt.0.role"] == "system"
    assert attrs["gen_ai.prompt.0.content"] == "You are helpful."
    assert attrs["turn.index"] == 1


def test_tool_call_pairs_by_call_id(span_exporter):
    """ToolCallEvent + ToolResultEvent produce a completed tool.<name> span."""
    exporter, tracer = span_exporter
    bus = EventBus()
    sub = OtelSubscriber(tracer=tracer)
    sub.attach(bus)

    bus.emit(ToolCallEvent(name="web_search", input_data={"query": "test"}, call_id="call_1"))
    bus.emit(ToolResultEvent(name="web_search", call_id="call_1", output="result text", is_error=False))

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name == "tool.web_search"]
    assert len(tool_spans) == 1

    attrs = tool_spans[0].attributes
    assert attrs["tool.name"] == "web_search"
    assert attrs["tool.output"] == "result text"
    assert attrs["tool.is_error"] is False
    assert "query" in attrs["tool.input"]


def test_tool_error_sets_error_status(span_exporter):
    """ToolResultEvent with is_error=True sets span status to ERROR."""
    exporter, tracer = span_exporter
    bus = EventBus()
    sub = OtelSubscriber(tracer=tracer)
    sub.attach(bus)

    bus.emit(ToolCallEvent(name="bash", input_data={"cmd": "bad"}, call_id="err_1"))
    bus.emit(ToolResultEvent(name="bash", call_id="err_1", output="error msg", is_error=True))

    spans = exporter.get_finished_spans()
    err_spans = [s for s in spans if s.name == "tool.bash"]
    assert len(err_spans) == 1
    assert err_spans[0].status.status_code == StatusCode.ERROR
    assert err_spans[0].attributes["tool.is_error"] is True


def test_compression_event_creates_span(span_exporter):
    """BatchCreatedEvent produces a compression.batch_created span."""
    exporter, tracer = span_exporter
    bus = EventBus()
    sub = OtelSubscriber(tracer=tracer)
    sub.attach(bus)

    bus.emit(BatchCreatedEvent(
        session_id="sess_1",
        batch_id="batch_42",
        turns_from=0,
        turns_to=5,
        summary="A summary of past turns.",
        members=[],
    ))

    spans = exporter.get_finished_spans()
    comp_spans = [s for s in spans if s.name == "compression.batch_created"]
    assert len(comp_spans) == 1

    attrs = comp_spans[0].attributes
    assert attrs["compression.batch_id"] == "batch_42"
    assert attrs["compression.turns_from"] == 0
    assert attrs["compression.turns_to"] == 5
    assert attrs["compression.summary_length"] == len("A summary of past turns.")


def test_no_subscription_to_text_delta(span_exporter):
    """TextDeltaEvent must not produce any spans (not subscribed)."""
    exporter, tracer = span_exporter
    bus = EventBus()
    sub = OtelSubscriber(tracer=tracer)
    sub.attach(bus)

    for _ in range(10):
        bus.emit(TextDeltaEvent(delta="token "))

    spans = exporter.get_finished_spans()
    assert len(spans) == 0, f"Expected 0 spans for TextDeltaEvent, got {len(spans)}"
