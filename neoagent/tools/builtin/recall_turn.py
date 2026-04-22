# neoagent/tools/builtin/recall_turn.py
"""Built-in tool: recall_turn.

Spec refs: § 14.2, § 15.4
Contract ref: C6

permission = "auto" — LLM can call freely, no user confirmation needed.
is_concurrent_safe = True — read-only, does not mutate session state.

SIMPLIFICATION (deferred to future phase):
  Ideally this tool should return the *full original message content* for each
  requested turn_id. However, the current messages list does not store a
  msg_id index (Phase 4 Batch 3 emits MessageCreatedEvent with the id, but
  does not embed it back into the message dict). Implementing a full-fidelity
  recall would require adding a msg_id → message index structure to Session,
  which is a separate architectural change deferred to Phase 8+.

  For now this tool returns BatchMember.preview as the "recalled content".
  The LLM already sees the batch summary in compressed_history; preview gives
  a slightly deeper 20-40 char excerpt per message. This is sufficient for
  the current use-case and clearly labelled as a simplification.
"""
from __future__ import annotations

import json
from pydantic import BaseModel
from neoagent.core.types import ToolResult
from neoagent.tools.base import BaseTool


class RecallTurnInput(BaseModel):
    turn_ids: list[str]


class RecallTurnTool(BaseTool):
    name = "recall_turn"
    description = (
        "Recall original messages for turn_ids listed in the <recoverable> clauses "
        "of compressed_history. Returns content as a tool_result (temporary; the "
        "messages list is not modified). turn_ids must all exist in the recoverable "
        "clause of at least one compression batch."
    )
    input_model = RecallTurnInput
    permission = "auto"
    is_concurrent_safe = True
    returns_external_content = False

    def __init__(self, session_state_ref):
        """
        session_state_ref: zero-argument callable returning current SessionState.
        """
        self._get_state = session_state_ref

    async def execute(self, input: RecallTurnInput) -> ToolResult:  # type: ignore[override]
        state = self._get_state()

        # Collect all recoverable member ids across all batches.
        # Uses getattr(..., []) so the tool works even if SessionState does not
        # yet have a 'batches' field (forward-compat / backward-compat).
        batches = getattr(state, "batches", [])

        # Build index: member_id → BatchMember (for fast lookup & preview access)
        member_index: dict[str, object] = {}
        for batch in batches:
            for member in batch.members:
                member_index[member.id] = member

        # Empty turn_ids → return empty recalled list (not an error)
        if not input.turn_ids:
            return ToolResult(
                call_id="",
                output=json.dumps({"recalled": []}, ensure_ascii=False),
                is_error=False,
            )

        # Validate: all requested turn_ids must be in the recoverable set
        missing = [tid for tid in input.turn_ids if tid not in member_index]
        if missing:
            return ToolResult(
                call_id="",
                output=(
                    f"turn_ids not in recoverable clauses: {missing}. "
                    f"Available recoverable ids: {sorted(member_index.keys())}"
                ),
                is_error=True,
            )

        # Build recalled list preserving request order
        # NOTE: returns BatchMember.preview, not full original content.
        # See module docstring for explanation of this simplification.
        recalled = []
        for tid in input.turn_ids:
            member = member_index[tid]
            recalled.append({
                "id": member.id,
                "role": member.role,
                "preview": member.preview,
                # _simplification: full content recall deferred (no msg_id index yet)
            })

        return ToolResult(
            call_id="",
            output=json.dumps({"recalled": recalled}, ensure_ascii=False, indent=2),
            is_error=False,
        )
