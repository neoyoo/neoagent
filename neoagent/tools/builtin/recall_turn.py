# neoagent/tools/builtin/recall_turn.py
"""Built-in tool: recall_turn.

Spec refs: § 14.2, § 15.4
Contract ref: C6

permission = "auto" — LLM can call freely, no user confirmation needed.
is_concurrent_safe = True — read-only, does not mutate session state.

Returns the full original Message content (not BatchMember.preview) for each
requested msg_id. Content is fetched from SessionState.messages by the `id`
field that QueryLoop stamps onto every Message.

Scope:
- Currently queries only session.messages. Once compression actually trims
  the messages list (Phase 2 Batch B), compressed-out Message originals must
  be retained elsewhere (e.g. SessionState.compressed_messages per spec § 2)
  and merged into the lookup here.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from neoagent.core.types import Message, TextBlock, ToolResultBlock, ToolUseBlock, ToolResult
from neoagent.tools.base import BaseTool


class RecallTurnInput(BaseModel):
    turn_ids: list[str]


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

    def __init__(self, session_ref):
        """
        session_ref: zero-argument callable returning the current Session
        (or SessionState — both resolve to the same messages source here).
        Using a callable avoids holding a stale reference across turn
        boundaries / session resumes.
        """
        self._get = session_ref

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

        # Build id → Message index (skip messages without id, which are legacy
        # or framework-injected without id tracking).
        index: dict[str, Message] = {m.id: m for m in messages if m.id}

        if not input.turn_ids:
            return ToolResult(
                call_id="",
                output=json.dumps({"recalled": [], "missing": []}, ensure_ascii=False),
                is_error=False,
            )

        recalled: list[dict] = []
        missing: list[str] = []
        for tid in input.turn_ids:
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
