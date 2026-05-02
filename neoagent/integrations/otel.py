from __future__ import annotations
import dataclasses
import json
import os
from typing import Callable

from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode, Status

from neoagent.events import (
    EventBus,
    ProviderRequestEvent,
    ProviderResponseEvent,
    ToolCallEvent,
    ToolResultEvent,
    BatchCreatedEvent,
    MemoryExtractEvent,
)


class OtelSubscriber:
    """订阅 EventBus 事件并转成 OTEL spans。

    设计：参考 ObserverSubscriber 的适配器模式 (neoagent/observe_subscriber.py)。
    Span 起止靠事件配对，所有 attributes 遵循 OpenTelemetry GenAI semantic conventions。

    Usage:
        subscriber = OtelSubscriber()
        subscriber.attach(bus)
        ...
        subscriber.detach(bus)
    """

    def __init__(self, tracer=None) -> None:
        self._tracer = tracer or trace.get_tracer("neoagent")
        self._pending_llm_span: Span | None = None
        self._pending_llm_ctx = None
        self._tool_spans: dict[str, Span] = {}
        self._handlers: list[tuple[type, Callable]] = []

    def attach(self, bus: EventBus) -> None:
        """Subscribe all handlers to the bus and track them for later detach."""
        pairs: list[tuple[type, Callable]] = [
            (ProviderRequestEvent, self._on_provider_request),
            (ProviderResponseEvent, self._on_provider_response),
            (ToolCallEvent, self._on_tool_call),
            (ToolResultEvent, self._on_tool_result),
            (BatchCreatedEvent, self._on_compression),
            (MemoryExtractEvent, self._on_memory_extract),
        ]
        for event_type, handler in pairs:
            bus.subscribe(event_type, handler)
            self._handlers.append((event_type, handler))

    def detach(self, bus: EventBus) -> None:
        """Unsubscribe all tracked handlers from the bus."""
        for event_type, handler in self._handlers:
            bus.unsubscribe(event_type, handler)
        self._handlers.clear()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _serialize_content(content) -> str:
        """Serialize content that may be str, list/tuple of blocks (Pydantic or dataclass)."""
        if isinstance(content, str):
            return content
        items = list(content)
        if not items:
            return ""
        try:
            serialized = []
            for b in items:
                if dataclasses.is_dataclass(b) and not isinstance(b, type):
                    serialized.append(dataclasses.asdict(b))
                elif hasattr(b, "model_dump"):
                    # Pydantic v2
                    serialized.append(b.model_dump())
                elif hasattr(b, "dict"):
                    # Pydantic v1
                    serialized.append(b.dict())
                else:
                    serialized.append(str(b))
            return json.dumps(serialized, ensure_ascii=False)
        except Exception:
            return str(items)

    # ── handlers ──────────────────────────────────────────────────────────────

    def _on_provider_request(self, ev: ProviderRequestEvent) -> None:
        span = self._tracer.start_span("llm.call")
        span.set_attribute("gen_ai.system", os.environ.get("LLM_PROVIDER", "anthropic"))
        span.set_attribute("gen_ai.request.model", os.environ.get("LLM_MODEL", "unknown"))
        span.set_attribute("turn.index", ev.turn)

        # system prompt as prompt.0
        span.set_attribute("gen_ai.prompt.0.role", "system")
        span.set_attribute("gen_ai.prompt.0.content", ev.system)

        # messages as prompt.1+
        for i, msg in enumerate(ev.messages):
            prefix = f"gen_ai.prompt.{i + 1}"
            span.set_attribute(f"{prefix}.role", msg.role)
            span.set_attribute(f"{prefix}.content", self._serialize_content(msg.content))

        # tools
        for j, tool in enumerate(ev.tools):
            span.set_attribute(f"gen_ai.tools.{j}.name", tool["name"])
            span.set_attribute(f"gen_ai.tools.{j}.description", tool.get("description", ""))

        self._pending_llm_span = span

    def _on_provider_response(self, ev: ProviderResponseEvent) -> None:
        if self._pending_llm_span is None:
            return
        span = self._pending_llm_span
        span.set_attribute("gen_ai.response.finish_reason", ev.stop_reason)
        span.set_attribute("gen_ai.usage.input_tokens", ev.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", ev.output_tokens)
        span.set_attribute("gen_ai.completion.0.role", "assistant")
        span.set_attribute("gen_ai.completion.0.content", self._serialize_content(ev.content))
        span.end()
        self._pending_llm_span = None

    def _on_tool_call(self, ev: ToolCallEvent) -> None:
        span = self._tracer.start_span(f"tool.{ev.name}")
        span.set_attribute("tool.name", ev.name)
        span.set_attribute("tool.call_id", ev.call_id)
        span.set_attribute(
            "tool.input",
            json.dumps(dict(ev.input_data), ensure_ascii=False),
        )
        self._tool_spans[ev.call_id] = span

    def _on_tool_result(self, ev: ToolResultEvent) -> None:
        span = self._tool_spans.pop(ev.call_id, None)
        if span is None:
            return
        span.set_attribute("tool.output", ev.output[:8000])
        span.set_attribute("tool.is_error", ev.is_error)
        if ev.is_error:
            span.set_status(Status(StatusCode.ERROR))
        span.end()

    def _on_compression(self, ev: BatchCreatedEvent) -> None:
        with self._tracer.start_as_current_span("compression.batch_created") as span:
            span.set_attribute("compression.batch_id", ev.batch_id)
            span.set_attribute("compression.turns_from", ev.turns_from)
            span.set_attribute("compression.turns_to", ev.turns_to)
            span.set_attribute("compression.summary_length", len(ev.summary or ""))

    def _on_memory_extract(self, ev: MemoryExtractEvent) -> None:
        with self._tracer.start_as_current_span("memory.extract") as span:
            span.set_attribute("memory.triggered", ev.triggered)
            span.set_attribute("memory.tool_calls", ev.tool_calls)
            span.set_attribute("memory.token_delta", ev.token_delta)
