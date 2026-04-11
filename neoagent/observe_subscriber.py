from __future__ import annotations
from neoagent.events import (
    EventBus,
    ProviderRequestEvent, ProviderResponseEvent,
    ToolCallEvent, ToolResultEvent,
    CompressCheckEvent, CompressDoneEvent, CompressFallbackEvent,
    MemoryExtractEvent, SkillChangeEvent,
)
from neoagent.observe import Observer


class ObserverSubscriber:
    """将现有 Observer 的方法映射到 EventBus 订阅。

    适配器模式：Observer 保留全部日志逻辑，ObserverSubscriber 负责事件转发。
    Observer 本身不需要知道 EventBus 的存在。
    """

    def __init__(self, observer: Observer, bus: EventBus) -> None:
        self._observer = observer
        bus.subscribe(ProviderRequestEvent, self._on_provider_request)
        bus.subscribe(ProviderResponseEvent, self._on_provider_response)
        bus.subscribe(ToolCallEvent, self._on_tool_call)
        bus.subscribe(ToolResultEvent, self._on_tool_result)
        bus.subscribe(CompressCheckEvent, self._on_compress_check)
        bus.subscribe(CompressDoneEvent, self._on_compress_done)
        bus.subscribe(CompressFallbackEvent, self._on_compress_fallback)
        bus.subscribe(MemoryExtractEvent, self._on_memory_extract)
        bus.subscribe(SkillChangeEvent, self._on_skill_change)

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

    def _on_skill_change(self, e: SkillChangeEvent) -> None:
        if e.active:
            self._observer.on_skill_activate(e.name)
        else:
            self._observer.on_skill_deactivate(e.name)
