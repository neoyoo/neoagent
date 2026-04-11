from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING
import tiktoken
from neoagent.core.types import Message, TextBlock, ToolResultBlock, ToolUseBlock

if TYPE_CHECKING:
    from neoagent.providers.base import Provider

logger = logging.getLogger(__name__)
_KEEP_RECENT = 6
_ENCODING = tiktoken.get_encoding("cl100k_base")
_SUMMARIZE_SYSTEM_TEMPLATE = """\
You are a context compression assistant. Compress the conversation below using this structure:

GOAL: <the original task or conversation goal — preserve from previous summary if provided>
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
    def __init__(self, provider: Provider, max_failures: int = 3) -> None:
        self._provider = provider
        self._max_failures = max_failures
        self._consecutive_failures: int = 0
        self._previous_summary: str | None = None

    def reset_session_state(self) -> None:
        """Clear iterative summary state. Call when starting a new task/session."""
        self._previous_summary = None
        self._consecutive_failures = 0

    def estimate_tokens(self, messages: list[Message]) -> int:
        total = 0
        for msg in messages:
            total += len(_ENCODING.encode(_message_to_text(msg)))
        return total

    def estimate_tools_tokens(self, schemas: list[dict]) -> int:
        if not schemas:
            return 0
        return len(_ENCODING.encode(json.dumps(schemas)))

    def should_compress(self, messages: list[Message], schemas: list[dict], context_budget: int) -> bool:
        return (self.estimate_tokens(messages) + self.estimate_tools_tokens(schemas)) > context_budget * 0.7

    async def compress(self, messages: list[Message], context_budget: int) -> list[Message]:
        if len(messages) <= 1:
            return messages
        if self._consecutive_failures >= self._max_failures:
            return self._truncate_oldest(messages)
        try:
            compressed = await self._llm_compress(messages)
            self._consecutive_failures = 0
            return compressed
        except Exception:
            self._consecutive_failures += 1
            return self._truncate_oldest(messages)

    async def _llm_compress(self, messages: list[Message]) -> list[Message]:
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
        response = await self._provider.create(
            system=_build_summarize_system(self._previous_summary),
            messages=[Message(role="user", content=middle_text)],
            tools=[],
        )
        summary = response.text_content.strip() or "(no summary)"
        self._previous_summary = summary
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
