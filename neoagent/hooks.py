from __future__ import annotations
import dataclasses
import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Literal

from neoagent.providers.base import Response

logger = logging.getLogger(__name__)

# ── HookResult ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HookResult:
    """Return value from a hook handler.

    Three mutually exclusive states:
    - allow:  proceed normally
    - deny:   abort the operation (pre hooks only; ignored in post)
    - modify: proceed with modified_data applied to the target object
    """
    action: Literal["allow", "deny", "modify"]
    reason: str | None = None
    modified_data: dict | None = None

    @classmethod
    def allow(cls) -> "HookResult":
        return cls(action="allow")

    @classmethod
    def deny(cls, reason: str = "") -> "HookResult":
        return cls(action="deny", reason=reason)

    @classmethod
    def modify(cls, data: dict, reason: str | None = None) -> "HookResult":
        return cls(action="modify", modified_data=data, reason=reason)


# ── Hook event types ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreToolCallEvent:
    """Fired before a tool call, after PermissionChecker passes.

    modify: set modified_data = {"tool_input": {...}} to replace input dict.
    deny:   return error ToolResult with reason, skip execution.

    ``tool_input`` is stored as a ``MappingProxyType`` so in-place mutation by
    handlers raises ``TypeError``, enforcing the HookResult.modify protocol.
    """
    tool_name: str
    tool_input: MappingProxyType
    call_id: str

    def __post_init__(self) -> None:
        # Ensure tool_input is always an immutable MappingProxyType even when
        # callers pass a plain dict (frozen dataclass requires object.__setattr__).
        if not isinstance(self.tool_input, MappingProxyType):
            object.__setattr__(self, "tool_input", MappingProxyType(dict(self.tool_input)))


@dataclass(frozen=True)
class PostToolCallEvent:
    """Fired after a tool call completes (including errors).

    modify: set modified_data = {"result": "..."} to replace output string.
    deny:   ignored (execution already happened).

    ``tool_input`` is stored as a ``MappingProxyType`` — see ``PreToolCallEvent``.
    """
    tool_name: str
    tool_input: MappingProxyType
    call_id: str
    result: str
    is_error: bool

    def __post_init__(self) -> None:
        if not isinstance(self.tool_input, MappingProxyType):
            object.__setattr__(self, "tool_input", MappingProxyType(dict(self.tool_input)))


@dataclass(frozen=True)
class PreProviderCallEvent:
    """Fired before Provider.create() — can modify messages/system/tools.

    modify: set modified_data with any subset of:
            {"system": "...", "messages": [...], "tools": [...]}
    deny:   skip LLM call; QueryLoop injects "[Hook denied: reason]" assistant message.

    ``messages`` and ``tools`` are stored as immutable ``tuple``s to prevent
    handlers from mutating the sequences in-place, bypassing HookResult.modify.
    """
    system: str
    messages: tuple
    tools: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple):
            object.__setattr__(self, "messages", tuple(self.messages))
        if not isinstance(self.tools, tuple):
            object.__setattr__(self, "tools", tuple(self.tools))


@dataclass(frozen=True)
class PostProviderCallEvent:
    """Fired after Provider.create() returns.

    modify: set modified_data = {"response": Response(...)} to replace response.
    deny:   ignored (LLM call already happened).
    """
    response: Response
    input_tokens: int
    output_tokens: int


# ── Type aliases (used by HookManager below) ──────────────────────────────────

HookType = Literal["pre_tool_call", "post_tool_call", "pre_provider_call", "post_provider_call"]
HookHandler = Callable[..., Awaitable[HookResult]]


# ── HookManager ───────────────────────────────────────────────────────────────

@dataclass
class _HookEntry:
    priority: int
    seq: int
    handler: HookHandler


class HookManager:
    """Manages hook registration and execution for pre/post hook lifecycle events.

    Handlers are called in ascending priority order; same-priority handlers run
    in registration order (stable via a monotone sequence counter).
    """

    def __init__(self) -> None:
        self._hooks: dict[HookType, list[_HookEntry]] = {}
        self._seq: int = 0

    def register(self, hook_type: HookType, handler: HookHandler, priority: int = 0) -> None:
        """Register *handler* for *hook_type* at the given *priority* (lower = earlier)."""
        entry = _HookEntry(priority=priority, seq=self._seq, handler=handler)
        self._seq += 1
        if hook_type not in self._hooks:
            self._hooks[hook_type] = []
        self._hooks[hook_type].append(entry)
        self._hooks[hook_type].sort(key=lambda e: (e.priority, e.seq))

    def unregister(self, hook_type: HookType, handler: HookHandler) -> None:
        """Remove *handler* from *hook_type*. No-op if not registered."""
        if hook_type not in self._hooks:
            return
        self._hooks[hook_type] = [e for e in self._hooks[hook_type] if e.handler is not handler]

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _apply_modify(event: Any, modified_data: dict) -> Any:
        """Return a new event with fields in *modified_data* replaced."""
        field_names = {f.name for f in dataclasses.fields(event)}
        updates = {k: v for k, v in modified_data.items() if k in field_names}
        if not updates:
            return event
        return dataclasses.replace(event, **updates)

    @staticmethod
    def _extract_modified_data(original: Any, final: Any) -> dict:
        """Return a dict of fields that differ between *original* and *final*."""
        diff: dict = {}
        for f in dataclasses.fields(original):
            orig_val = getattr(original, f.name)
            final_val = getattr(final, f.name)
            if orig_val != final_val:
                diff[f.name] = final_val
        return diff

    # ── execution ─────────────────────────────────────────────────────────────

    async def run_pre(self, hook_type: HookType, event: Any) -> HookResult:
        """Run pre-hook handlers in priority order.

        - deny:   short-circuits, remaining handlers are NOT called.
        - modify: updates the working event via dataclasses.replace(); next handler sees new event.
        - allow:  continue.
        - exception: log WARNING, skip handler, continue chain.

        Returns the first deny result, or a modify result accumulating all changes,
        or HookResult.allow() if nothing significant happened.
        """
        entries = self._hooks.get(hook_type, [])
        if not entries:
            return HookResult.allow()

        original_event = event
        current_event = event

        for entry in entries:
            try:
                result = await entry.handler(current_event)
            except Exception:
                logger.warning(
                    "Hook handler %r raised an exception for %r; skipping.",
                    entry.handler,
                    hook_type,
                    exc_info=True,
                )
                continue

            if result.action == "deny":
                return result
            elif result.action == "modify" and result.modified_data:
                current_event = self._apply_modify(current_event, result.modified_data)
            # allow → continue

        # Build final result
        if current_event is not original_event:
            diff = self._extract_modified_data(original_event, current_event)
            return HookResult.modify(diff)
        return HookResult.allow()

    async def run_post(self, hook_type: HookType, event: Any) -> HookResult:
        """Run post-hook handlers in priority order.

        - deny:   ignored (execution already happened); handler is still called.
        - modify: accumulates changes; next handler sees updated event.
        - allow:  continue.
        - exception: log WARNING, skip handler, continue chain.

        Returns a modify result if any handler modified the event, otherwise allow.
        """
        entries = self._hooks.get(hook_type, [])
        if not entries:
            return HookResult.allow()

        original_event = event
        current_event = event

        for entry in entries:
            try:
                result = await entry.handler(current_event)
            except Exception:
                logger.warning(
                    "Hook handler %r raised an exception for %r; skipping.",
                    entry.handler,
                    hook_type,
                    exc_info=True,
                )
                continue

            if result.action == "deny":
                # Ignored in post; continue chain without modifying event
                continue
            elif result.action == "modify" and result.modified_data:
                current_event = self._apply_modify(current_event, result.modified_data)
            # allow → continue

        if current_event is not original_event:
            diff = self._extract_modified_data(original_event, current_event)
            return HookResult.modify(diff)
        return HookResult.allow()
