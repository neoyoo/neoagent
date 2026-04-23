# neoagent/v2/strategies/oneshot_memory_review.py
"""OneShotMemoryReviewStrategy — single LLM call memory reviewer.

Spec refs:
  § 11.2  (lines 1884-1896) — MemoryReviewStrategy ABC
  § 11.3  (lines 1899-1964) — OneShotMemoryReviewStrategy implementation

Contract refs:
  C2 — MemoryReviewStrategy ABC

Soft-failure semantics (differs from CompressionStrategy):
  - JSON parse failure / top-level non-list → retry; exhausted → return []
  - Individual entry validation failure → skip entry, log warning, continue
  - Memory review is NOT on the critical path; losing a few entries is acceptable.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from datetime import datetime

from anthropic import AsyncAnthropic

from neoagent.v2.abc import MemoryReviewStrategy
from neoagent.v2.schema import MemoryEntry, WorkingMemory
from neoagent.v2.strategies._memory_review_prompt import MEMORY_REVIEW_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Valid memory types per spec § 11.3
_ALLOWED_TYPES = frozenset({"preference", "fact", "goal", "constraint", "habit"})


class OneShotMemoryReviewStrategy(MemoryReviewStrategy):
    """Single LLM call: input messages + WM, output list of MemoryEntry objects.

    spec § 11.3 (lines 1899-1964)

    Degrade strategy (soft-failure):
    - JSON parse error / non-list response → retry up to max_retries;
      exhausted → return [] (do NOT raise — memory review is non-critical).
    - Individual entry with invalid fields → skip that entry + warning log.
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

    async def review(
        self,
        session_id: str,
        user_id: str,
        messages: list,
        wm: WorkingMemory,
    ) -> list[MemoryEntry]:
        """Extract learnable MemoryEntry objects from recent messages.

        Returns an empty list on exhausted retries (soft failure).
        """
        prompt = self._build_prompt(user_id, messages, wm)

        for attempt in range(self.max_retries):
            try:
                raw = await self._call_llm(prompt)
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError(
                        f"Expected JSON array at top level, got {type(parsed).__name__}"
                    )
                return self._build_entries(parsed, user_id)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "MemoryReview attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                if attempt == self.max_retries - 1:
                    logger.error(
                        "MemoryReview exhausted %d retries for session=%s user=%s; returning []",
                        self.max_retries,
                        session_id,
                        user_id,
                    )
                    return []

        return []  # unreachable, satisfies type checkers

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_prompt(
        self,
        user_id: str,
        messages: list,
        wm: WorkingMemory,
    ) -> str:
        """Build the memory review prompt, embedding user_id, messages and WM."""
        # Format messages for prompt
        if messages:
            msgs_lines = []
            for msg in messages:
                if isinstance(msg, dict):
                    msg_id = msg.get("id", msg.get("msg_id", "?"))
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    if isinstance(content, str) and len(content) > 300:
                        content = content[:300] + "..."
                    elif isinstance(content, list):
                        content = str(content)[:300] + "..."
                    msgs_lines.append(f"  [{msg_id}] ({role}): {content}")
                else:
                    msgs_lines.append(f"  {msg}")
            messages_text = "\n".join(msgs_lines)
        else:
            messages_text = "  （暂无消息）"

        # Serialize WorkingMemory to JSON
        try:
            wm_dict = asdict(wm)
            # Convert datetime objects to ISO strings for JSON serialization
            for key, value in wm_dict.items():
                if isinstance(value, datetime):
                    wm_dict[key] = value.isoformat()
            working_memory_json = json.dumps(wm_dict, ensure_ascii=False, indent=2)
        except Exception:
            # Fallback: minimal representation
            working_memory_json = json.dumps(
                {"session_id": wm.session_id},
                ensure_ascii=False,
            )

        return MEMORY_REVIEW_PROMPT_TEMPLATE.format(
            user_id=user_id,
            messages_text=messages_text,
            working_memory_json=working_memory_json,
        )

    async def _call_llm(self, prompt: str) -> str:
        """Single AsyncAnthropic messages.create call; returns text content."""
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        # Skip non-text blocks (thinking / redacted_thinking); return first text.
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        # Non-critical: memory review degrades silently; return empty JSON list.
        return "[]"

    def _build_entries(
        self,
        parsed_list: list[dict],
        user_id: str,
    ) -> list[MemoryEntry]:
        """Validate each raw dict and construct MemoryEntry objects.

        Invalid entries are skipped with a warning; valid entries are returned.
        Auto-fills: user_id, memory_id (uuid hex 12 chars), created_at (now).
        """
        entries: list[MemoryEntry] = []

        for i, item in enumerate(parsed_list):
            if not isinstance(item, dict):
                logger.warning(
                    "MemoryReview: entry[%d] is not a dict (got %s), skipping",
                    i,
                    type(item).__name__,
                )
                continue

            # Validate type
            entry_type = item.get("type")
            if entry_type not in _ALLOWED_TYPES:
                logger.warning(
                    "MemoryReview: entry[%d] has invalid type=%r "
                    "(must be one of %s), skipping",
                    i,
                    entry_type,
                    sorted(_ALLOWED_TYPES),
                )
                continue

            # Validate content
            content = item.get("content")
            if not content or not isinstance(content, str) or not content.strip():
                logger.warning(
                    "MemoryReview: entry[%d] has missing or empty content, skipping",
                    i,
                )
                continue

            # Validate confidence
            confidence = item.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                logger.warning(
                    "MemoryReview: entry[%d] has out-of-range confidence=%r "
                    "(must be in [0, 1]), skipping",
                    i,
                    confidence,
                )
                continue

            # Extract optional fields
            category = item.get("category")  # may be None
            evidence = item.get("evidence")  # may be None

            entries.append(
                MemoryEntry(
                    user_id=user_id,
                    memory_id=uuid.uuid4().hex[:12],
                    type=entry_type,
                    category=category if isinstance(category, str) else None,
                    content=content.strip(),
                    confidence=float(confidence),
                    evidence=evidence if isinstance(evidence, dict) else None,
                    created_at=datetime.now(),
                )
            )

        return entries
