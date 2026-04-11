from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from neoagent.core.types import Message
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
    """
    tool_name: str
    tool_input: dict
    call_id: str


@dataclass(frozen=True)
class PostToolCallEvent:
    """Fired after a tool call completes (including errors).

    modify: set modified_data = {"result": "..."} to replace output string.
    deny:   ignored (execution already happened).
    """
    tool_name: str
    tool_input: dict
    call_id: str
    result: str
    is_error: bool


@dataclass(frozen=True)
class PreProviderCallEvent:
    """Fired before Provider.create() — can modify messages/system/tools.

    modify: set modified_data with any subset of:
            {"system": "...", "messages": [...], "tools": [...]}
    deny:   skip LLM call; QueryLoop injects "[Hook denied: reason]" assistant message.
    """
    system: str
    messages: list[Message]
    tools: list[dict]


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
