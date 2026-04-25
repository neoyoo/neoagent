from __future__ import annotations
from typing import Callable
from neoagent.events import (
    EventBus,
    ProviderRequestEvent, ProviderResponseEvent,
    ToolCallEvent, ToolResultEvent,
    CompressCheckEvent, CompressDoneEvent, CompressFallbackEvent,
    MemoryExtractEvent,
    ToolResultFreedEvent,
)
from neoagent.observe import Observer


class ObserverSubscriber:
    """将现有 Observer 的方法映射到 EventBus 订阅。

    适配器模式：Observer 保留全部日志逻辑，ObserverSubscriber 负责事件转发。
    Observer 本身不需要知道 EventBus 的存在。

    Usage:
        subscriber = ObserverSubscriber(observer)
        subscriber.attach(bus)   # subscribe all handlers
        ...
        subscriber.detach(bus)   # unsubscribe all handlers (cleanup / re-attach)
    """

    def __init__(self, observer: Observer, bus: EventBus | None = None) -> None:
        self._observer = observer
        self._handlers: list[tuple[type, Callable]] = []
        if bus is not None:
            self.attach(bus)

    def attach(self, bus: EventBus) -> None:
        """Subscribe all handlers to the bus and track them for later detach."""
        pairs: list[tuple[type, Callable]] = [
            (ProviderRequestEvent, self._on_provider_request),
            (ProviderResponseEvent, self._on_provider_response),
            (ToolCallEvent, self._on_tool_call),
            (ToolResultEvent, self._on_tool_result),
            (CompressCheckEvent, self._on_compress_check),
            (CompressDoneEvent, self._on_compress_done),
            (CompressFallbackEvent, self._on_compress_fallback),
            (MemoryExtractEvent, self._on_memory_extract),
            (ToolResultFreedEvent, self._on_tool_result_freed),
            # NOTE: SkillChangeEvent is defined but not yet wired; PromptBuilder needs EventBus access (planned for future).
        ]
        for event_type, handler in pairs:
            bus.subscribe(event_type, handler)
            self._handlers.append((event_type, handler))

    def detach(self, bus: EventBus) -> None:
        """Unsubscribe all tracked handlers from the bus."""
        for event_type, handler in self._handlers:
            bus.unsubscribe(event_type, handler)
        self._handlers.clear()

    def _on_provider_request(self, e: ProviderRequestEvent) -> None:
        self._observer.on_provider_request(e.system, list(e.messages), list(e.tools), e.turn)

    def _on_provider_response(self, e: ProviderResponseEvent) -> None:
        self._observer.on_provider_response(list(e.content), e.stop_reason, e.input_tokens, e.output_tokens, e.turn)

    def _on_tool_call(self, e: ToolCallEvent) -> None:
        self._observer.on_tool_call(e.name, e.input_data)

    def _on_tool_result(self, e: ToolResultEvent) -> None:
        self._observer.on_tool_result(e.name, e.output[:300], e.is_error)

    def _on_compress_check(self, e: CompressCheckEvent) -> None:
        self._observer.on_compress_check(e.msg_tokens, e.tool_tokens, e.budget, e.should_compress)

    def _on_compress_done(self, e: CompressDoneEvent) -> None:
        self._observer.on_compress_done(e.summary, e.previous_summary)

    def _on_compress_fallback(self, e: CompressFallbackEvent) -> None:
        self._observer.on_compress_fallback(e.reason)

    def _on_memory_extract(self, e: MemoryExtractEvent) -> None:
        if e.triggered:
            self._observer.on_memory_extract_done(e.items_stored, list(e.filenames))
        else:
            self._observer.on_memory_extract_skip(e.tool_calls, e.token_delta)

    def _on_tool_result_freed(self, e: ToolResultFreedEvent) -> None:
        self._observer.on_tool_result_freed(
            e.tool_use_id, e.tool_name, e.size, e.preview, e.reason
        )
