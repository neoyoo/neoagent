from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="Event")
Handler = Callable[["Event"], None]


@dataclass(frozen=True)
class Event:
    """所有事件的基类。frozen 确保事件不可变（安全传递给多个 handler）。"""
    pass


# ── Provider events ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProviderRequestEvent(Event):
    system: str
    messages: tuple          # list → tuple 保证 frozen
    tools: tuple
    turn: int


@dataclass(frozen=True)
class ProviderResponseEvent(Event):
    content: tuple
    stop_reason: str
    input_tokens: int
    output_tokens: int
    turn: int


# ── Tool events ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolCallEvent(Event):
    name: str
    input_data: dict
    call_id: str


@dataclass(frozen=True)
class ToolResultEvent(Event):
    name: str
    call_id: str
    output: str
    is_error: bool


# ── Compression events ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompressCheckEvent(Event):
    msg_tokens: int
    tool_tokens: int
    budget: int
    should_compress: bool


@dataclass(frozen=True)
class CompressDoneEvent(Event):
    summary: str
    previous_summary: str | None


@dataclass(frozen=True)
class CompressFallbackEvent(Event):
    reason: str


# ── Memory events ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MemoryExtractEvent(Event):
    triggered: bool
    tool_calls: int
    token_delta: int
    items_stored: int = 0
    filenames: tuple = ()


# ── Skill events ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SkillChangeEvent(Event):
    """Emitted when a skill is activated or deactivated.

    TODO v3.2: wire into PromptBuilder.activate_skill() / deactivate_skill()
    once PromptBuilder receives EventBus access.
    """
    name: str
    active: bool             # True=activate, False=deactivate


# ── Turn events ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TurnCompleteEvent(Event):
    turn_index: int
    stop_reason: str
    tool_call_count: int


# ── EventBus ──────────────────────────────────────────────────────────────────

_ALL_EVENT_TYPES = [
    ProviderRequestEvent, ProviderResponseEvent,
    ToolCallEvent, ToolResultEvent,
    CompressCheckEvent, CompressDoneEvent, CompressFallbackEvent,
    MemoryExtractEvent, SkillChangeEvent, TurnCompleteEvent,
]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[Handler]] = {}

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        self._handlers.setdefault(event_type, []).append(handler)  # type: ignore[arg-type]

    def unsubscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        handlers = self._handlers.get(event_type, [])
        try:
            handlers.remove(handler)  # type: ignore[arg-type]
        except ValueError:
            pass

    def emit(self, event: Event) -> None:
        """同步广播。handler 抛异常时记录日志但不中断其他 handler。"""
        for handler in list(self._handlers.get(type(event), [])):
            try:
                handler(event)
            except Exception:
                logger.exception("EventBus handler error for %s", type(event).__name__)

    def subscribe_all(self, handler: Handler) -> None:
        """订阅所有事件类型（用于 debug logging）。"""
        for event_type in _ALL_EVENT_TYPES:
            self.subscribe(event_type, handler)
