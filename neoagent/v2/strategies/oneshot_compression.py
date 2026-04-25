# neoagent/v2/strategies/oneshot_compression.py
"""OneShotCompressionStrategy — single LLM call compressor.

Spec refs:
  § 10.2  (lines 1589-1635) — CompressionStrategy ABC + OneShot signature
  § 10.3a (lines 1710-1766) — Compressor Output Contract (8 hard rules)
  § 10.4  (lines 1770-1800) — 7-section structured template format

Contract refs:
  C2 — CompressionStrategy ABC
  C4 — Output Contract (JSON schema + 8 rules + degrade strategy a)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from anthropic import AsyncAnthropic

from neoagent.v2.abc import CompressionStrategy
from neoagent.v2.errors import CompressionError
from neoagent.v2.schema import BatchMember, CompressionContext, CompressionDelta
from neoagent.v2.strategies._compressor_prompt import COMPRESSOR_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# ── Contract constants (§ 10.3a) ──────────────────────────────────────────────

_LIST_FIELDS = frozenset({
    "constraints_and_preferences",
    "key_decisions",
    "relevant_files",
    "next_steps",
})
_SCALAR_FIELDS = frozenset({
    "progress",
    "critical_context",
})
_VALID_ROLES = frozenset({"user", "assistant", "tool"})
_VALID_LIST_OPS = frozenset({"set", "append", "remove"})
_SCALAR_OP_ONLY = frozenset({"set"})

# list-segment value must start with a letter prefix + digits + colon
_PREFIX_PATTERN = re.compile(r"^(c|d|f|n)\d+: ")


class OneShotCompressionStrategy(CompressionStrategy):
    """Single LLM call: input full context, output complete delta JSON.

    Cheapest option; suitable for short sessions or budget-constrained deployments.

    spec § 10.3 + § 10.3a

    Degrade strategy (a):
    - Rules 1-4 violated → drop working_memory_delta (set to []),
      retain batch_members, log warning, do NOT raise.
    - JSON parse failure / top-level structure error → retry up to max_retries;
      exhausted → raise CompressionError.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 3,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_url = base_url
        # base_url=None falls back to the Anthropic default; pass a custom URL
        # for Anthropic-protocol-compatible providers (e.g. DashScope's Qwen endpoint).
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url) if base_url \
            else AsyncAnthropic(api_key=api_key)

    async def compress(self, context: CompressionContext) -> CompressionDelta:
        """Compress context via single LLM call. § 10.2, § 10.3a."""
        prompt = self._build_prompt(context)

        for attempt in range(self.max_retries):
            try:
                raw = await self._call_llm(prompt)
                parsed = json.loads(raw)
                # Validate top-level structure (required keys must exist)
                for required_key in ("batch_members", "working_memory_delta"):
                    if required_key not in parsed:
                        raise KeyError(f"Missing top-level key: {required_key}")
                return self._validate_and_build(parsed, context)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Compressor attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                if attempt == self.max_retries - 1:
                    raise CompressionError(
                        f"Exhausted {self.max_retries} retries: last error: {exc}"
                    ) from exc

        # unreachable, but satisfies type checkers
        raise CompressionError("Exhausted retries")  # pragma: no cover

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_prompt(self, ctx: CompressionContext) -> str:
        """Build compressor prompt from context, embedding all 8 hard rules."""
        # Format previous_batches
        if ctx.previous_batches:
            batches_text = "\n".join(
                f"  - id={b.get('id', '?')}: {b.get('summary', '')}"
                if isinstance(b, dict)
                else f"  - {b}"
                for b in ctx.previous_batches
            )
        else:
            batches_text = "  （暂无）"

        # Format previous working memory
        previous_wm_text = json.dumps(ctx.previous_wm, ensure_ascii=False, indent=2)

        # Format messages grouped by turn
        messages_text, msg_id_range_hint = self._format_messages_by_turn(ctx.messages)

        return COMPRESSOR_PROMPT_TEMPLATE.format(
            previous_batches=batches_text,
            previous_wm=previous_wm_text,
            messages_to_compress=messages_text,
            msg_id_range_hint=msg_id_range_hint,
            trigger=ctx.trigger,
        )

    def _format_messages_by_turn(
        self, messages: list
    ) -> tuple[str, str]:
        """Format messages grouped by turn label and compute msg_id_range_hint.

        Returns (messages_text, msg_id_range_hint).

        Turn grouping:
          - Messages with turn=None → "Turn (unknown):" group at the top
          - Others → "Turn N:" groups in ascending turn order

        Each message line: "  [msg_id] (role): content"
        Content truncated to 200 chars.
        """
        if not messages:
            return "  （暂无消息）", ""

        # Collect msg_ids for range hint
        all_msg_ids: list[str] = []
        # Group messages by turn value
        # turn_groups: ordered dict turn_key -> list of formatted lines
        # turn_key: int | None (None = unknown)
        from collections import defaultdict
        turn_groups: dict[int | None, list[str]] = defaultdict(list)
        turn_order: list[int | None] = []  # preserves first-seen order
        seen_turns: set[int | None] = set()

        # Also track distinct user-turn values (int only, not None) for K count
        distinct_user_turns: set[int] = set()

        for msg in messages:
            if isinstance(msg, dict):
                msg_id = msg.get("id") or msg.get("msg_id") or "?"
                role = msg.get("role", "?")
                content = msg.get("content", "")
                turn_val = msg.get("turn")
            else:
                # Message dataclass/Pydantic object
                msg_id = getattr(msg, "id", None) or "?"
                role = getattr(msg, "role", "?")
                content = getattr(msg, "content", "")
                turn_val = getattr(msg, "turn", None)

            if isinstance(msg_id, str) and msg_id != "?":
                all_msg_ids.append(msg_id)

            # Truncate content
            if isinstance(content, str) and len(content) > 200:
                content = content[:200] + "..."
            elif isinstance(content, list):
                content = str(content)[:200] + "..."

            line = f"  [{msg_id}] ({role}): {content}"

            if turn_val not in seen_turns:
                seen_turns.add(turn_val)
                turn_order.append(turn_val)
            turn_groups[turn_val].append(line)

            if isinstance(turn_val, int):
                distinct_user_turns.add(turn_val)

        # Build output: None-turn group first, then ascending int-turn groups
        # Sort: None first, then int turns in ascending order
        sorted_turns: list[int | None] = []
        if None in turn_order:
            sorted_turns.append(None)
        int_turns = sorted(t for t in turn_order if t is not None)
        sorted_turns.extend(int_turns)

        blocks: list[str] = []
        for t in sorted_turns:
            label = "Turn (unknown):" if t is None else f"Turn {t}:"
            blocks.append(label)
            blocks.extend(turn_groups[t])

        messages_text = "\n".join(blocks) if blocks else "  （暂无消息）"

        # Compute msg_id_range_hint
        msg_id_range_hint = self._compute_msg_id_range_hint(
            all_msg_ids, len(messages), len(distinct_user_turns)
        )

        return messages_text, msg_id_range_hint

    def _compute_msg_id_range_hint(
        self, msg_ids: list[str], n_messages: int, k_user_turns: int
    ) -> str:
        """Compute the msg_id_range hint string.

        Extracts numeric ids (mN format) to find first/last.
        Falls back to first/last in list order if no numeric ids found.
        """
        if not msg_ids:
            return ""

        # Extract mN-format ids for numeric range
        numeric_ids: list[int] = []
        for mid in msg_ids:
            if mid.startswith("m") and mid[1:].isdigit():
                numeric_ids.append(int(mid[1:]))

        if numeric_ids:
            first_n = min(numeric_ids)
            last_n = max(numeric_ids)
            first_id = f"m{first_n}"
            last_id = f"m{last_n}"
        else:
            first_id = msg_ids[0]
            last_id = msg_ids[-1]

        return (
            f"Available msg_id range: {first_id}..{last_id} "
            f"({n_messages} messages across {k_user_turns} user turns).\n"
            "You MUST reference only msg_ids from this range in <recoverable>. "
            "Fabricating ids will cause retrieval failures."
        )

    async def _call_llm(self, prompt: str) -> str:
        """Single AsyncAnthropic messages.create call, returns text content."""
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        # Skip non-text blocks (thinking / redacted_thinking) and return the
        # first text block's content. Anthropic-protocol providers (e.g. Qwen
        # via DashScope) may prepend ThinkingBlocks.
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        raise CompressionError("Provider returned no text block in compressor response")

    def _validate_and_build(
        self,
        parsed: dict[str, Any],
        ctx: CompressionContext,
    ) -> CompressionDelta:
        """Validate 8-rule contract and apply degrade strategy (a).

        Rules 1/2 apply to batch_members.
        Rules 3/4/5 apply to working_memory_delta.
        A single violation in batch_members also triggers delta drop (hard violation).
        """
        # batch_summary: deprecated — silently ignore if present (legacy LLM output)
        batch_members_raw = parsed["batch_members"]
        wm_delta_raw = parsed["working_memory_delta"]

        # Build existing msg_id → Message mapping for rule 1 validation + turn lookup
        existing_ids: set[str] = set()
        msg_id_to_turn: dict[str, int | None] = {}
        for msg in ctx.messages:
            if isinstance(msg, dict):
                msg_id = msg.get("id") or msg.get("msg_id")
                if msg_id:
                    existing_ids.add(str(msg_id))
                    msg_id_to_turn[str(msg_id)] = msg.get("turn")
            else:
                msg_id = getattr(msg, "id", None)
                if msg_id:
                    existing_ids.add(str(msg_id))
                    msg_id_to_turn[str(msg_id)] = getattr(msg, "turn", None)

        # ── Validate batch_members (rules 1 & 2) ─────────────────────────────
        batch_members: list[BatchMember] = []
        member_violation: str | None = None

        for item in batch_members_raw:
            msg_id = item.get("id", "")
            role = item.get("role", "")
            preview = item.get("preview", "")

            # Rule 1: id must exist in context.messages
            if existing_ids and msg_id not in existing_ids:
                member_violation = (
                    f"Rule 1 violation: batch_member id '{msg_id}' "
                    f"not found in context messages {sorted(existing_ids)}"
                )
                break

            # Rule 2: role ∈ {"user", "assistant", "tool"}
            if role not in _VALID_ROLES:
                member_violation = (
                    f"Rule 2 violation: batch_member role '{role}' "
                    f"not in {sorted(_VALID_ROLES)}"
                )
                break

            # Populate turn: use LLM output if present, else fall back to Message.turn
            llm_turn = item.get("turn")
            if llm_turn is not None:
                member_turn: int | None = int(llm_turn)
            elif msg_id in msg_id_to_turn:
                member_turn = msg_id_to_turn[msg_id]
            else:
                # Orphan msg_id — graceful degradation: set turn=None, log warning
                logger.warning(
                    "BatchMember id '%s' not found in context for turn lookup; "
                    "setting turn=None (graceful degradation)",
                    msg_id,
                )
                member_turn = None

            batch_members.append(BatchMember(id=msg_id, role=role, preview=preview, turn=member_turn))

        if member_violation:
            logger.warning(
                "CompressionDelta hard violation — dropping working_memory_delta. "
                "Reason: %s",
                member_violation,
            )
            # Degrade strategy (a): keep batch_members parsed so far,
            # but drop wm_delta entirely. For member violations, still include
            # the members parsed before the violation (or none if first item failed).
            # Per spec: retain batch_members — include what was valid.
            # However since the LLM output is untrusted at this point, we include
            # the fully built batch_members list (items before violation).
            return CompressionDelta(
                batch_members=batch_members,  # valid members before violation
                working_memory_delta=[],
            )

        # ── Validate working_memory_delta (rules 3, 4, 5) ────────────────────
        validated_delta: list[dict] = []
        delta_violation: str | None = None

        for item in wm_delta_raw:
            field = item.get("field", "")
            op = item.get("op", "")
            value = item.get("value")
            item_id = item.get("item_id")

            if field in _SCALAR_FIELDS:
                # Rule 3: scalar fields only allow op="set"
                if op not in _SCALAR_OP_ONLY:
                    delta_violation = (
                        f"Rule 3 violation: scalar field '{field}' "
                        f"only allows op='set', got op='{op}'"
                    )
                    break
            elif field in _LIST_FIELDS:
                # Rule 3: list fields allow set/append/remove
                if op not in _VALID_LIST_OPS:
                    delta_violation = (
                        f"Rule 3 violation: list field '{field}' "
                        f"op must be one of {sorted(_VALID_LIST_OPS)}, got '{op}'"
                    )
                    break

                # Rule 4a: for set/append, value must have prefix id
                if op in ("set", "append"):
                    if not isinstance(value, str) or not _PREFIX_PATTERN.match(value):
                        delta_violation = (
                            f"Rule 4 violation: list field '{field}' op='{op}' "
                            f"value must match ^(c|d|f|n)\\d+: prefix, got: {value!r}"
                        )
                        break

                # Rule 4b: remove must include item_id
                if op == "remove":
                    if not item_id:
                        delta_violation = (
                            f"Rule 4 violation: list field '{field}' op='remove' "
                            f"must provide item_id, got: {item_id!r}"
                        )
                        break
            else:
                # Unknown field — treat as violation to be safe
                delta_violation = (
                    f"Rule 3 violation: unknown field '{field}' — "
                    f"must be one of {sorted(_SCALAR_FIELDS | _LIST_FIELDS)}"
                )
                break

            validated_delta.append(item)

        if delta_violation:
            logger.warning(
                "CompressionDelta hard violation — dropping working_memory_delta. "
                "Reason: %s",
                delta_violation,
            )
            # Degrade strategy (a): drop entire working_memory_delta
            return CompressionDelta(
                batch_members=batch_members,
                working_memory_delta=[],
            )

        return CompressionDelta(
            batch_members=batch_members,
            working_memory_delta=validated_delta,
        )
