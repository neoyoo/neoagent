from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING
from neoagent.core.types import Message, TextBlock, ToolUseBlock, ToolResultBlock

if TYPE_CHECKING:
    from neoagent.memory.store import MemoryStore
    from neoagent.providers.base import Provider

logger = logging.getLogger(__name__)

_TOOL_CALLS_THRESHOLD = 5
_TOKEN_DELTA_THRESHOLD = 4000
_MAX_CONVERSATION_CHARS = 8000

_EXTRACT_SYSTEM = """\
You are a memory extraction assistant for an AI agent.
Given a conversation, extract long-term information worth remembering across sessions.

Output a JSON array. Each element has exactly these keys:
  "filename"    — short snake_case name ending in .md, e.g. "user_prefs.md"
  "description" — one sentence ≤160 chars, used as retrieval index
  "content"     — full markdown content to store

Only extract information that is:
- Stable across sessions (not ephemeral task details)
- Useful to recall in future conversations
- Not derivable from reading the code or documentation

If nothing is worth extracting, return an empty array: []

Return ONLY valid JSON. No explanation, no markdown fences."""


def _should_extract(tool_calls_count: int, token_delta: int) -> bool:
    return tool_calls_count >= _TOOL_CALLS_THRESHOLD or token_delta >= _TOKEN_DELTA_THRESHOLD


def _messages_to_text(messages: list[Message]) -> str:
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg.content, str):
            parts.append(f"{msg.role}: {msg.content}")
        elif isinstance(msg.content, list):
            for b in msg.content:
                if isinstance(b, TextBlock):
                    parts.append(f"{msg.role}: {b.text}")
                elif isinstance(b, ToolUseBlock):
                    args = json.dumps(b.input)[:200]
                    parts.append(f"[tool_use: {b.name}({args})]")
                elif isinstance(b, ToolResultBlock):
                    preview = (b.content or "")[:300]
                    parts.append(f"[tool_result: {preview}]")
    return "\n".join(parts)


class MemoryExtractor:
    """LLM-based extractor: decides what to remember and writes to MemoryStore."""

    def __init__(self, provider: "Provider", store: "MemoryStore") -> None:
        self._provider = provider
        self._store = store

    async def extract(
        self,
        messages: list[Message],
        tool_calls_count: int = 0,
        token_delta: int = 0,
    ) -> int:
        """Extract memories from messages and persist them.

        Returns the number of memory items stored.
        Only runs if trigger conditions are met.
        """
        if not _should_extract(tool_calls_count, token_delta):
            return 0

        conversation = _messages_to_text(messages)[-_MAX_CONVERSATION_CHARS:]
        try:
            response = await self._provider.create(
                system=_EXTRACT_SYSTEM,
                messages=[Message(role="user", content=conversation)],
                tools=[],
            )
            raw = response.text_content.strip()
            if not raw:
                return 0
            # Strip markdown code fences if present (```json ... ```)
            if raw.startswith("```"):
                lines = raw.splitlines()
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw = "\n".join(lines).strip()
            items = json.loads(raw)
            if not isinstance(items, list):
                return 0
        except Exception as exc:
            logger.warning("Memory extraction failed: %s", exc)
            return 0

        import re as _re
        stored = 0
        for item in items:
            filename = item.get("filename", "").strip()
            description = item.get("description", "").strip()
            content = item.get("content", "").strip()
            if not filename or not description or not content:
                continue
            # Sanitize: reject path traversal, keep only safe characters
            if "/" in filename or ".." in filename or "\\" in filename:
                continue
            if not _re.match(r'^[a-zA-Z0-9_\-]+\.md$', filename):
                continue
            self._store.write_topic(filename, content)
            stored += 1

        if stored > 0:
            self._rebuild_index()

        return stored

    def _rebuild_index(self) -> None:
        """Regenerate MEMORY.md from all .md files in the store directory."""
        entries: list[str] = []
        for path in sorted(self._store.directory.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            first_line = ""
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.lstrip("# ").strip()
                if stripped:
                    first_line = stripped[:160]
                    break
            entries.append(f"- [{first_line}]({path.name})")
        content = "# Memory Index\n\n" + "\n".join(entries) + "\n"
        self._store.write_index(content)
