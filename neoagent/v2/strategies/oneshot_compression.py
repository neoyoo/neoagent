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
    "goal",
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
    - Rules 1/2/3/4/5 violated → drop working_memory_delta (set to []),
      retain batch_summary + batch_members, log warning, do NOT raise.
    - JSON parse failure / top-level structure error → retry up to max_retries;
      exhausted → raise CompressionError.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self._client = AsyncAnthropic(api_key=api_key)

    async def compress(self, context: CompressionContext) -> CompressionDelta:
        """Compress context via single LLM call. § 10.2, § 10.3a."""
        prompt = self._build_prompt(context)

        for attempt in range(self.max_retries):
            try:
                raw = await self._call_llm(prompt)
                parsed = json.loads(raw)
                # Validate top-level structure (required keys must exist)
                for required_key in ("batch_summary", "batch_members", "working_memory_delta"):
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
        session_goal = ctx.previous_wm.get("goal", "(no goal set)")

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

        # Format messages to compress
        if ctx.messages:
            msgs_lines = []
            for msg in ctx.messages:
                if isinstance(msg, dict):
                    msg_id = msg.get("id", msg.get("msg_id", "?"))
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    # Truncate long content for prompt readability
                    if isinstance(content, str) and len(content) > 200:
                        content = content[:200] + "..."
                    elif isinstance(content, list):
                        content = str(content)[:200] + "..."
                    msgs_lines.append(f"  [{msg_id}] ({role}): {content}")
                else:
                    msgs_lines.append(f"  {msg}")
            messages_text = "\n".join(msgs_lines)
        else:
            messages_text = "  （暂无消息）"

        return COMPRESSOR_PROMPT_TEMPLATE.format(
            session_goal=session_goal,
            previous_batches=batches_text,
            previous_wm=previous_wm_text,
            messages_to_compress=messages_text,
            trigger=ctx.trigger,
        )

    async def _call_llm(self, prompt: str) -> str:
        """Single AsyncAnthropic messages.create call, returns text content."""
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text from first content block
        return response.content[0].text

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
        batch_summary_raw = parsed["batch_summary"]
        batch_members_raw = parsed["batch_members"]
        wm_delta_raw = parsed["working_memory_delta"]

        # Build existing msg_id set for rule 1 validation
        existing_ids: set[str] = set()
        for msg in ctx.messages:
            if isinstance(msg, dict):
                msg_id = msg.get("id") or msg.get("msg_id")
                if msg_id:
                    existing_ids.add(str(msg_id))

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

            batch_members.append(BatchMember(id=msg_id, role=role, preview=preview))

        if member_violation:
            logger.warning(
                "CompressionDelta hard violation — dropping working_memory_delta. "
                "Reason: %s",
                member_violation,
            )
            # Degrade strategy (a): keep batch_summary + batch_members parsed so far,
            # but drop wm_delta entirely. For member violations, still include
            # the members parsed before the violation (or none if first item failed).
            # Per spec: "保留 batch_summary + batch_members" — include what was valid.
            # However since the LLM output is untrusted at this point, we include
            # the fully built batch_members list (items before violation).
            return CompressionDelta(
                batch_summary=batch_summary_raw,
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

            # Rule 5: goal immutable — field="goal" forbidden in working_memory_delta
            if field == "goal":
                delta_violation = (
                    f"Rule 5 violation: goal is immutable — "
                    f"field='goal' found in working_memory_delta"
                )
                break

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
                batch_summary=batch_summary_raw,
                batch_members=batch_members,
                working_memory_delta=[],
            )

        return CompressionDelta(
            batch_summary=batch_summary_raw,
            batch_members=batch_members,
            working_memory_delta=validated_delta,
        )
