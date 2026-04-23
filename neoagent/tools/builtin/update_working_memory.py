# neoagent/tools/builtin/update_working_memory.py
"""Built-in tool: update_working_memory.

Spec ref: § 15.9 (lines 2665+)
Contract ref: C6, C1

permission = "auto" — LLM can call freely, no user confirmation needed.
is_concurrent_safe = False — mutates shared WorkingMemory state.

Design notes:
- session_state_ref is a callable (lambda/closure) returning the current
  SessionState, to avoid holding a stale reference across turn boundaries.
- Does NOT bump wm.version — the Phase 4 Batch 4 snapshot path owns version
  increment at turn-end.
- Scalar fields (progress, critical_context) only support op='set'.
- List fields support set / append / remove with prefix-id validation.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from neoagent.core.types import ToolResult
from neoagent.tools.base import BaseTool

# ── field classification ──────────────────────────────────────────────────────

_LIST_FIELDS: frozenset[str] = frozenset({
    "constraints_and_preferences",
    "key_decisions",
    "relevant_files",
    "next_steps",
})

_SCALAR_FIELDS: frozenset[str] = frozenset({
    "progress",
    "critical_context",
})

# Required id-prefix letter per list field (e.g. 'c' → "c01: …")
_PREFIX_MAP: dict[str, str] = {
    "constraints_and_preferences": "c",
    "key_decisions": "d",
    "relevant_files": "f",
    "next_steps": "n",
}


# ── Pydantic input model ───────────────────────────────────────────────────────

class UpdateWorkingMemoryInput(BaseModel):
    field: Literal[
        "constraints_and_preferences",
        "progress",
        "key_decisions",
        "relevant_files",
        "next_steps",
        "critical_context",
    ]
    value: str | None = None
    op: Literal["set", "append", "remove"] = "set"
    item_id: str | None = None


# ── Tool ───────────────────────────────────────────────────────────────────────

class UpdateWorkingMemoryTool(BaseTool):
    name = "update_working_memory"
    description = (
        "Record facts that the USER has explicitly stated — constraints, "
        "decisions, progress, next steps. Do NOT use this to record your own "
        "assumptions, speculations, or default guesses; only user-confirmed "
        "facts belong here. Prefer ONE call per assistant turn; batching many "
        "fabricated entries is a sign you should stop and ask the user instead. "
        "Scalar fields (progress, critical_context) only support op=set. "
        "List fields (constraints_and_preferences, key_decisions, relevant_files, "
        "next_steps) support set/append/remove with prefix-id validation "
        "(e.g. 'c01: …' for constraints_and_preferences)."
    )
    input_model = UpdateWorkingMemoryInput
    permission = "auto"
    is_concurrent_safe = False
    returns_external_content = False

    def __init__(self, session_state_ref):
        """
        session_state_ref: zero-argument callable that returns the current
        SessionState object. Using a callable avoids holding a stale reference
        when SessionState is replaced between turns.
        """
        self._get_state = session_state_ref

    async def execute(self, input: UpdateWorkingMemoryInput) -> ToolResult:  # type: ignore[override]
        state = self._get_state()
        wm = state._current_wm

        # Guard: WM not yet initialized
        if wm is None:
            return ToolResult(
                call_id="",
                output="WorkingMemory not initialized (framework_init has not run yet)",
                is_error=True,
            )

        # Rule: scalar fields only support op=set
        if input.field in _SCALAR_FIELDS and input.op != "set":
            return ToolResult(
                call_id="",
                output=(
                    f"Scalar field '{input.field}' only supports op=set "
                    f"(got op={input.op!r})"
                ),
                is_error=True,
            )

        # Rule: list fields value must have correct prefix-id format for set/append
        if input.field in _LIST_FIELDS and input.op in ("set", "append"):
            prefix = _PREFIX_MAP[input.field]
            # Accepted format: "<prefix><digits>: <anything>"  e.g. "c01: keep it short"
            if input.value is None or not re.match(rf"^{prefix}\d+: ", input.value):
                return ToolResult(
                    call_id="",
                    output=(
                        f"List field '{input.field}' value must start with "
                        f"'{prefix}NN: ' prefix (e.g. '{prefix}01: your text'). "
                        f"Got: {input.value!r}"
                    ),
                    is_error=True,
                )

        # Rule: op=remove requires item_id
        if input.op == "remove" and input.item_id is None:
            return ToolResult(
                call_id="",
                output="op=remove requires item_id to identify which list entry to remove",
                is_error=True,
            )

        # ── Apply the operation ────────────────────────────────────────────────

        if input.op == "set":
            if input.field in _SCALAR_FIELDS:
                setattr(wm, input.field, input.value or "")
            else:
                # list field: op=set replaces the entire list with a single-item list
                setattr(wm, input.field, [input.value])

        elif input.op == "append":
            lst: list = getattr(wm, input.field)
            lst.append(input.value)

        elif input.op == "remove":
            lst = getattr(wm, input.field)
            # Match by prefix: item_id="c02" removes entries starting with "c02: "
            new_lst = [item for item in lst if not item.startswith(f"{input.item_id}: ")]
            if len(new_lst) == len(lst):
                return ToolResult(
                    call_id="",
                    output=(
                        f"item_id '{input.item_id}' not found in field "
                        f"'{input.field}'. Current entries: {lst!r}"
                    ),
                    is_error=True,
                )
            setattr(wm, input.field, new_lst)

        # Bump WM metadata (version bump is deferred to turn-end snapshot path)
        wm.updated_by = "llm_tool"
        wm.updated_at = datetime.now()

        return ToolResult(
            call_id="",
            output=f"WM updated: field={input.field!r} op={input.op!r}",
            is_error=False,
        )
