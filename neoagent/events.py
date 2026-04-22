from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Callable, Literal, TypeVar

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
    input_data: dict  # MappingProxyType at runtime (from ToolExecutor); dict for test/compat construction
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


# ── Tool result lifecycle events ──────────────────────────────────────────────

@dataclass(frozen=True)
class ToolResultFreedEvent(Event):
    """Emitted when a tool_result is auto-freed (collapsed to placeholder)."""
    tool_use_id: str
    tool_name: str
    size: int
    preview: str
    reason: str   # "manual" | "global_compression"


@dataclass(frozen=True)
class ToolResultRecalledEvent(Event):
    """Emitted when a freed tool_result is recalled for the current turn."""
    tool_use_id: str
    tool_name: str


# ── Session events ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SessionResumeWarningEvent(Event):
    """Emitted when resume(validate=True) detects a potential issue with the session."""
    session_id: str
    reason: str   # "workspace_missing" | "stale_session"
    details: str


# ── Multi-agent events ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkerEvent(Event):
    """Wraps an event emitted by a worker agent, bubbled up to the orchestrator."""
    worker_name: str
    task_id: str
    depth: int
    inner: Event


@dataclass(frozen=True)
class TaskDispatchEvent(Event):
    """Emitted when a task is dispatched to a worker."""
    task_id: str
    worker_name: str
    instruction: str
    depth: int


@dataclass(frozen=True)
class TaskCompleteEvent(Event):
    """Emitted when a worker completes (or fails) a task."""
    task_id: str
    worker_name: str
    status: str
    turns_completed: int
    usage: "TokenUsage | None"


# ── v2 Events ─────────────────────────────────────────────────────────────────
# spec § 8.3, lines 1343-1392; § 10.3a (CompressionFailedEvent)
# BatchMember imported lazily via TYPE_CHECKING to avoid circular imports at
# module level; the field type is annotated as `list` per C3 contract.

@dataclass(frozen=True)
class MessageCreatedEvent(Event):
    """spec § 8.3, lines 1343-1392"""
    session_id: str
    msg_id: str
    turn: int
    role: str
    source_type: Literal[
        "user_input",
        "assistant_reply",
        "tool_result",
        "system_injected_compression",
        "system_injected_memory",
        "system_injected_recall",
    ]
    content: list


@dataclass(frozen=True)
class ToolResultPersistedEvent(Event):
    """spec § 8.3, lines 1343-1392"""
    session_id: str
    tool_use_id: str
    turn: int
    tool_name: str
    output: str
    size_bytes: int
    is_error: bool


@dataclass(frozen=True)
class BatchCreatedEvent(Event):
    """spec § 8.3, lines 1343-1392"""
    session_id: str
    batch_id: str
    turns_from: int
    turns_to: int
    summary: str
    members: list  # list[BatchMember] — from neoagent.v2.schema


@dataclass(frozen=True)
class WorkingMemoryUpdatedEvent(Event):
    """spec § 8.3, lines 1343-1392"""
    session_id: str
    version: int
    at_turn: int
    wm_json: dict
    updated_by: str


@dataclass(frozen=True)
class CompressionFailedEvent(Event):
    """spec § 10.3a — fallback strategy a"""
    session_id: str
    reason: str
    retry_count: int


# ── EventBus ──────────────────────────────────────────────────────────────────

_ALL_EVENT_TYPES = [
    ProviderRequestEvent, ProviderResponseEvent,
    ToolCallEvent, ToolResultEvent,
    CompressCheckEvent, CompressDoneEvent, CompressFallbackEvent,
    MemoryExtractEvent, SkillChangeEvent, TurnCompleteEvent,
    ToolResultFreedEvent, ToolResultRecalledEvent,
    WorkerEvent, TaskDispatchEvent, TaskCompleteEvent,
    SessionResumeWarningEvent,
    # v2 events
    MessageCreatedEvent, ToolResultPersistedEvent, BatchCreatedEvent,
    WorkingMemoryUpdatedEvent, CompressionFailedEvent,
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
