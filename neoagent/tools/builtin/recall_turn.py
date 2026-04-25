# neoagent/tools/builtin/recall_turn.py
"""Built-in tool: recall_turn.

Spec refs: § 14.2, § 15.4
Contract ref: C6

permission = "auto" — LLM can call freely, no user confirmation needed.
is_concurrent_safe = True — read-only, does not mutate session state.

Returns the full original Message content (not BatchMember.preview) for each
requested msg_id. Content is fetched by the `id` field that QueryLoop stamps
onto every Message.

Scope:
- Queries both `session.messages` (live window) and
  `session_state.compressed_messages` (trimmed via compression). Live copies
  win on id collision.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from neoagent.core.types import Message, TextBlock, ToolResultBlock, ToolUseBlock, ToolResult
from neoagent.tools.base import BaseTool

if TYPE_CHECKING:
    from neoagent.v2.compressed_store import CompressedMessageStore


class RecallTurnInput(BaseModel):
    msg_ids: list[str]


def _serialize_content(content: Any) -> Any:
    """Convert a Message.content to a JSON-safe value."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return repr(content)
    out = []
    for block in content:
        if isinstance(block, TextBlock):
            out.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            out.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        elif isinstance(block, ToolResultBlock):
            out.append({
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
                "is_error": block.is_error,
            })
        elif isinstance(block, BaseModel):
            out.append(block.model_dump())
        elif isinstance(block, dict):
            out.append(block)
        else:
            out.append(repr(block))
    return out


class RecallTurnTool(BaseTool):
    name = "recall_turn"
    description = (
        "Recall the full original content of previous messages by msg_id. "
        "Use this to retrieve messages referenced in <compressed_history>'s "
        "<recoverable> block (<turn n=...><msg id=... />). You may pass several "
        "ids from the same <turn n=N> group in one call to fetch a complete "
        "conversational round. Returns JSON "
        '[{"id", "role", "content"}, ...]. Ids not found (never existed, or '
        "already evicted) are reported in the 'missing' field instead of raising."
    )
    input_model = RecallTurnInput
    permission = "auto"
    is_concurrent_safe = True
    returns_external_content = False

    def __init__(self, session_ref, store: "CompressedMessageStore | None" = None):
        """
        session_ref: zero-argument callable returning the current Session
        (or SessionState — both resolve to the same messages source here).
        Using a callable avoids holding a stale reference across turn
        boundaries / session resumes.

        store: optional CompressedMessageStore for fetching compressed message
        bodies. When provided, compressed lookup goes through the store instead
        of reading session_state.compressed_messages directly.
        """
        self._get = session_ref
        self._store = store

    async def execute(self, input: RecallTurnInput) -> ToolResult:  # type: ignore[override]
        obj = self._get()
        # Accept either a Session (with .messages) or a SessionState (no
        # .messages directly, but exposes them via a parent Session). Since
        # SessionState doesn't own the messages list, callers typically pass
        # a lambda returning Session or session_state — handle both.
        messages: list[Message]
        if hasattr(obj, "messages"):
            messages = obj.messages
        elif hasattr(obj, "session") and hasattr(obj.session, "messages"):
            messages = obj.session.messages
        else:
            return ToolResult(
                call_id="",
                output="recall_turn: session reference missing .messages attribute",
                is_error=True,
            )

        # Live messages index — always built from the live window.
        live_index: dict[str, Message] = {m.id: m for m in messages if m.id}

        # Compressed messages — through store if available, else legacy fallback.
        # Determine session_id for store lookup.
        session_id: str | None = getattr(obj, "id", None)
        if session_id is None and hasattr(obj, "state") and obj.state is not None:
            wm = getattr(obj.state, "_current_wm", None)
            if wm is not None:
                session_id = getattr(wm, "session_id", None)

        compressed_index: dict[str, Message] = {}
        if self._store is not None and session_id and input.msg_ids:
            compressed_index = await self._store.get_many(session_id, input.msg_ids)
        elif hasattr(obj, "state") and hasattr(obj.state, "compressed_messages"):
            # Legacy fallback when no store wired
            compressed_index = {m.id: m for m in obj.state.compressed_messages if m.id}
        elif hasattr(obj, "compressed_messages"):
            compressed_index = {m.id: m for m in obj.compressed_messages if m.id}

        # Build id → Message index. Compressed list goes in first; live list
        # overwrites on collision so live copy always wins (per spec).
        index: dict[str, Message] = compressed_index.copy()
        index.update(live_index)

        if not input.msg_ids:
            return ToolResult(
                call_id="",
                output=json.dumps({"recalled": [], "missing": []}, ensure_ascii=False),
                is_error=False,
            )

        recalled: list[dict] = []
        missing: list[str] = []
        for tid in input.msg_ids:
            msg = index.get(tid)
            if msg is None:
                missing.append(tid)
                continue
            recalled.append({
                "id": tid,
                "role": msg.role,
                "content": _serialize_content(msg.content),
            })

        payload = {"recalled": recalled}
        if missing:
            payload["missing"] = missing
        return ToolResult(
            call_id="",
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            is_error=False,
        )
