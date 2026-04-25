from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING
import tiktoken
from neoagent.core.types import Message, TextBlock, ToolResultBlock, ToolUseBlock

if TYPE_CHECKING:
    from neoagent.events import EventBus
    from neoagent.providers.base import Provider
    from neoagent.session import SessionState
    from neoagent.v2.abc import CompressionStrategy
    from neoagent.v2.compressed_store import CompressedMessageStore

logger = logging.getLogger(__name__)
_KEEP_RECENT = 6
_SUMMARIZE_SYSTEM_TEMPLATE = """\
You are a context compression assistant. Compress the conversation below using this structure:

PROGRESS: <what has been accomplished>
DECISIONS: <key decisions made>
FILES: <files created or modified, if any>
NEXT STEPS: <what still needs to happen>
KEY CONTEXT: <other important facts>

{previous_section}Output only the structured summary. No explanation."""


def _build_summarize_system(previous_summary: str | None) -> str:
    if previous_summary:
        prev = f"Previous summary (update incrementally — do not discard):\n{previous_summary}\n\n"
    else:
        prev = ""
    return _SUMMARIZE_SYSTEM_TEMPLATE.format(previous_section=prev)

def _message_to_text(msg: Message) -> str:
    if isinstance(msg.content, str):
        return f"{msg.role}: {msg.content}"
    parts = [f"{msg.role}:"]
    for block in msg.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            parts.append(f"[tool_use id={block.id} name={block.name} input={json.dumps(block.input)}]")
        elif isinstance(block, ToolResultBlock):
            parts.append(f"[tool_result id={block.tool_use_id} content={block.content}]")
    return " ".join(parts)

class ContextCompressor:
    def __init__(
        self,
        provider: "Provider",
        max_failures: int = 3,
        tokenizer=None,
        strategy: "CompressionStrategy | None" = None,
        event_bus: "EventBus | None" = None,
        compressed_store: "CompressedMessageStore | None" = None,
    ) -> None:
        self._provider = provider
        self._max_failures = max_failures
        if tokenizer is not None:
            self._enc = tokenizer
        else:
            self._enc = tiktoken.get_encoding("cl100k_base")
        self._strategy = strategy
        self._bus = event_bus
        self._compressed_store = compressed_store

    def estimate_tokens(self, messages: list[Message]) -> int:
        total = 0
        for msg in messages:
            total += len(self._enc.encode(_message_to_text(msg)))
        return total

    def estimate_tools_tokens(self, schemas: list[dict]) -> int:
        if not schemas:
            return 0
        return len(self._enc.encode(json.dumps(schemas)))

    def should_compress(
        self,
        messages: list[Message],
        schemas: list[dict],
        context_budget: int,
        turns_since_last_compression: int = 0,
    ) -> tuple[bool, str]:
        """Returns (should_compress, reason). reason in {"turn_count", "token_threshold", ""}."""
        if turns_since_last_compression >= 10:
            return (True, "turn_count")
        if (self.estimate_tokens(messages) + self.estimate_tools_tokens(schemas)) > context_budget * 0.7:
            return (True, "token_threshold")
        return (False, "")

    async def compress(
        self,
        messages: list[Message],
        context_budget: int,
        session_state: "SessionState | None" = None,
        current_turn: int = 0,
        compress_reason: str = "token_threshold",
        session_id: str | None = None,
    ) -> list[Message] | None:
        """Compress messages. Returns:
        - list[Message]: the to_keep list (caller should replace session.messages with it)
        - None: nothing to compress (skip clear+extend)

        Strategy path uses _split_messages_for_compression to determine to_compress/to_keep.
        Legacy path returns a list (always).
        """
        # ── Strategy path (Task 4.6) ──────────────────────────────────────────
        if self._strategy is not None:
            return await self._compress_with_strategy(
                messages, session_state=session_state, current_turn=current_turn,
                compress_reason=compress_reason, session_id=session_id,
            )

        # ── Legacy path (no strategy) ─────────────────────────────────────────
        if len(messages) <= 1:
            return messages
        failures = session_state.compression_failures if session_state else 0
        if failures >= self._max_failures:
            return self._truncate_oldest(messages)
        try:
            compressed = await self._llm_compress(messages, session_state=session_state)
            if session_state:
                session_state.compression_failures = 0
            return compressed
        except Exception:
            if session_state:
                session_state.compression_failures += 1
            return self._truncate_oldest(messages)

    # ── Strategy path helpers ─────────────────────────────────────────────────

    async def _compress_with_strategy(
        self,
        messages: list[Message],
        session_state: "SessionState | None" = None,
        current_turn: int = 0,
        compress_reason: str = "token_threshold",
        session_id: str | None = None,
    ) -> list[Message] | None:
        """Use injected CompressionStrategy to compress. Emits events as appropriate.

        Returns to_keep list on success, or None if nothing to compress.
        On circuit-break (CompressionError / orphan id), returns messages unchanged.
        """
        from neoagent.v2.errors import CompressionError
        from neoagent.v2.schema import CompressionContext

        # Try to get session_id from WM, caller arg, or fall back to "unknown"
        if session_id is None:
            if session_state and session_state._current_wm:
                session_id = session_state._current_wm.session_id
            else:
                session_id = "unknown"

        previous_wm: dict = {}
        if session_state and session_state._current_wm:
            from neoagent.session import _wm_to_dict
            previous_wm = _wm_to_dict(session_state._current_wm)

        previous_batches: list = session_state.batches if session_state else []

        # ── Physical trim: keep latest 5 user turns ───────────────────────────
        to_compress, to_keep = _split_messages_for_compression(messages, keep_recent_user_turns=5)

        # If nothing to compress, skip LLM call entirely
        if not to_compress:
            return None

        ctx = CompressionContext(
            session_id=session_id,
            messages=to_compress,
            previous_batches=previous_batches,
            previous_wm=previous_wm,
            trigger=compress_reason,
        )

        try:
            delta = await self._strategy.compress(ctx)  # type: ignore[union-attr]
        except CompressionError as exc:
            from neoagent.events import CompressionFailedEvent
            if self._bus is not None:
                self._bus.emit(CompressionFailedEvent(
                    session_id=session_id,
                    reason=str(exc),
                    retry_count=0,
                ))
            # Circuit-break: return messages unchanged
            return messages

        # ── Deterministic Executor — orphan id validation (§ 10.8) ───────────
        # Validate against to_compress (the portion the LLM saw), not all messages.
        if delta.batch_members:
            orphan_id = self._find_orphan_id(delta.batch_members, to_compress, session_state)
            if orphan_id is not None:
                from neoagent.events import CompressionFailedEvent
                if self._bus is not None:
                    self._bus.emit(CompressionFailedEvent(
                        session_id=session_id,
                        reason=f"orphan_id:{orphan_id}",
                        retry_count=0,
                    ))
                return messages

        # ── Apply delta ───────────────────────────────────────────────────────
        self._apply_delta(delta, session_state, session_id, current_turn, compress_reason)

        # ── Move to_compress → compressed store (or legacy direct mutation) ────
        if session_state is not None:
            if self._compressed_store is not None and session_id is not None:
                await self._compressed_store.put_many(session_id, to_compress)
            else:
                # Legacy fallback: direct mutation. Kept for tests that don't wire a store.
                session_state.compressed_messages.extend(to_compress)
            session_state.turns_since_last_compression = 0

        return to_keep

    def _find_orphan_id(
        self,
        batch_members: list,
        messages: list[Message],
        session_state: "SessionState | None",
    ) -> str | None:
        """Return first orphan id in batch_members, or None if all are valid.

        A member id is orphaned when the session has id-tracked messages
        (id_gen._msg_counter > 0) and the id is a "mN" format id with N
        outside the tracked range.

        When _msg_counter == 0 (no messages have been id-tracked yet), skip
        validation — there is no basis to reject any id.
        """
        if session_state is None:
            # No id tracking available — can't validate, pass through
            return None

        max_counter = session_state.id_gen._msg_counter
        if max_counter == 0:
            # No id-tracked messages — no basis to reject
            return None

        # Build valid id set from id_gen counter (m1..m<counter>)
        valid_ids: set[str] = set()
        for i in range(1, max_counter + 1):
            valid_ids.add(f"m{i}")

        for member in batch_members:
            member_id = member.id if hasattr(member, "id") else str(member)
            # Only validate if it looks like a generated id (mNNN format)
            if member_id.startswith("m") and member_id[1:].isdigit():
                if member_id not in valid_ids:
                    return member_id
            else:
                # Non-standard id format — treat as unknown and skip
                pass

        return None

    def _apply_delta(
        self,
        delta,
        session_state: "SessionState | None",
        session_id: str,
        current_turn: int,
        compress_reason: str = "token_threshold",
    ) -> None:
        """Write Batch + apply WM delta + emit BatchCreatedEvent."""
        from datetime import datetime, timezone
        from neoagent.v2.schema import Batch

        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # Determine batch id
        if session_state:
            batch_id = session_state.id_gen.next_batch_id()
        else:
            batch_id = f"cm_{uuid.uuid4().hex[:8]}"

        # Determine turn range from batch_members
        turns_from = current_turn
        turns_to = current_turn

        # Create and store Batch
        batch = Batch(
            session_id=session_id,
            batch_id=batch_id,
            turns_from=turns_from,
            turns_to=turns_to,
            time_from=now,
            time_to=now,
            summary=delta.batch_summary,
            members=list(delta.batch_members),
            trigger=compress_reason,
            created_at=now,
        )

        # Store batch on session_state (batches is now a proper SessionState field)
        if session_state is not None:
            session_state.batches.append(batch)

        # Emit BatchCreatedEvent
        if self._bus is not None:
            from neoagent.events import BatchCreatedEvent
            self._bus.emit(BatchCreatedEvent(
                session_id=session_id,
                batch_id=batch_id,
                turns_from=turns_from,
                turns_to=turns_to,
                summary=delta.batch_summary,
                members=list(delta.batch_members),
            ))

        # Apply working_memory_delta to _current_wm
        if session_state and session_state._current_wm is not None:
            wm = session_state._current_wm
            for op_dict in delta.working_memory_delta:
                _apply_wm_op(wm, op_dict)
            # Bump version + metadata
            wm.version += 1
            wm.at_turn = current_turn
            wm.updated_by = "framework_compression"
            wm.updated_at = now

    # ── Legacy path helpers ───────────────────────────────────────────────────

    async def _llm_compress(
        self,
        messages: list[Message],
        session_state: "SessionState | None" = None,
    ) -> list[Message]:
        if len(messages) < 3:
            return messages
        anchor = messages[0]
        keep = min(_KEEP_RECENT, len(messages) - 2)
        middle = messages[1:len(messages) - keep]
        recent = messages[len(messages) - keep:]
        if not middle:
            middle = messages[1:-1] if len(messages) > 2 else []
            recent = [messages[-1]]
        middle_text = "\n".join(_message_to_text(m) for m in middle)
        previous_summary = session_state.previous_summary if session_state is not None else None
        response = await self._provider.create(
            system=_build_summarize_system(previous_summary),
            messages=[Message(role="user", content=middle_text)],
            tools=[],
        )
        summary = response.text_content.strip() or "(no summary)"
        if session_state is not None:
            session_state.previous_summary = summary
        summary_msg = Message(role="user", content=f"[Context summary from earlier in the conversation]\n{summary}")
        return self._sanitize_tool_pairs([anchor, summary_msg, *recent])

    def _truncate_oldest(self, messages: list[Message]) -> list[Message]:
        if len(messages) < 3:
            return messages
        anchor = messages[0]
        keep = min(_KEEP_RECENT, len(messages) - 1)
        recent = messages[len(messages) - keep:]
        return self._sanitize_tool_pairs([anchor, *recent])

    def _sanitize_tool_pairs(self, messages: list[Message]) -> list[Message]:
        use_ids: set[str] = set()
        result_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg.content, list):
                for b in msg.content:
                    if isinstance(b, ToolUseBlock):
                        use_ids.add(b.id)
                    elif isinstance(b, ToolResultBlock):
                        result_ids.add(b.tool_use_id)
        matched = use_ids & result_ids
        sanitized: list[Message] = []
        for msg in messages:
            if isinstance(msg.content, str):
                sanitized.append(msg)
                continue
            clean = []
            for b in msg.content:
                if isinstance(b, ToolUseBlock) and b.id not in matched:
                    continue
                if isinstance(b, ToolResultBlock) and b.tool_use_id not in matched:
                    continue
                clean.append(b)
            if clean:
                sanitized.append(Message(role=msg.role, content=clean))
        # Ensure role alternation — insert placeholder if needed
        result: list[Message] = []
        for msg in sanitized:
            if result and msg.role == result[-1].role:
                placeholder_role = "user" if msg.role == "assistant" else "assistant"
                result.append(Message(role=placeholder_role, content="(context removed during compression)"))
            result.append(msg)
        return result


def _apply_wm_op(wm, op_dict: dict) -> None:
    """Apply a single working_memory_delta operation to a WorkingMemory object.

    Supported ops:
      - scalar fields (progress, critical_context): op="set"
      - list fields (constraints_and_preferences, key_decisions, relevant_files,
        next_steps): op="set"|"append"|"remove"

    This layer trusts that the contract rules have already been validated
    by the strategy (per spec — Deterministic Executor validates orphan ids,
    not wm_delta content).
    """
    field = op_dict.get("field", "")
    op = op_dict.get("op", "")
    value = op_dict.get("value")
    item_id = op_dict.get("item_id")

    _LIST_FIELDS = {"constraints_and_preferences", "key_decisions", "relevant_files", "next_steps"}
    _SCALAR_FIELDS = {"progress", "critical_context"}

    if field in _SCALAR_FIELDS:
        if op == "set":
            setattr(wm, field, value)
    elif field in _LIST_FIELDS:
        current: list = getattr(wm, field, [])
        if op == "set":
            # Accept either a list (replace whole) or a str (single-item list, back-compat)
            if isinstance(value, list):
                setattr(wm, field, list(value))
            else:
                setattr(wm, field, [value])
        elif op == "append":
            current.append(value)
        elif op == "remove" and item_id:
            setattr(wm, field, [item for item in current if not item.startswith(f"{item_id}:")])
    # Unknown field / op: silently skip (validated upstream by strategy)


def _split_messages_for_compression(
    messages: list[Message],
    keep_recent_user_turns: int = 5,
) -> tuple[list[Message], list[Message]]:
    """Split messages into (to_compress, to_keep).

    to_keep = messages belonging to the latest N user turns (identified by
    distinct Message.turn values for user-role messages with turn != None).
    to_compress = everything older than that boundary.

    Edge cases:
    - If fewer than N user turns exist total, to_compress = [], to_keep = messages.
    - Messages with turn=None (legacy) are always kept (in to_keep, at their
      original position via the partition boundary logic).
    - Preserve original order in both lists.
    """
    if not messages:
        return [], []

    # Collect distinct user-turn values from messages that have a non-None turn.
    # We only count user-role messages for "user-initiated turns".
    seen_user_turns: list[int] = []
    seen_set: set[int] = set()
    for msg in messages:
        if msg.role == "user" and msg.turn is not None and msg.turn not in seen_set:
            seen_user_turns.append(msg.turn)
            seen_set.add(msg.turn)

    # Not enough user turns to compress anything
    if len(seen_user_turns) <= keep_recent_user_turns:
        return [], list(messages)

    # The N most-recent distinct user turns to keep
    turns_to_keep: set[int] = set(seen_user_turns[-keep_recent_user_turns:])

    # Find the index of the first message that belongs to a kept user turn.
    # Messages with turn=None are treated as belonging to "latest" (always kept).
    split_idx = len(messages)  # default: nothing to compress
    for i, msg in enumerate(messages):
        if msg.turn is None:
            # Legacy message — once we hit a None-turn we note it but keep going;
            # they'll end up in to_keep because the split point is computed from
            # the first "kept" user-turn message.
            continue
        if msg.turn in turns_to_keep:
            # First message that belongs to a kept user turn — split here
            split_idx = i
            break

    to_compress = list(messages[:split_idx])
    to_keep = list(messages[split_idx:])

    # Move any None-turn messages from to_compress into to_keep, preserving
    # relative order: prepend them at the start of to_keep.
    none_turn_from_compress = [m for m in to_compress if m.turn is None]
    if none_turn_from_compress:
        real_compress = [m for m in to_compress if m.turn is not None]
        # Re-insert None-turn msgs at their original positions in the full list
        # by simply rebuilding: to_compress = non-None-turn portion before split,
        # to_keep = None-turn messages originally before split + rest of to_keep.
        # We preserve overall order: none_turn_from_compress + to_keep (already ordered).
        to_compress = real_compress
        to_keep = none_turn_from_compress + to_keep

    return to_compress, to_keep
