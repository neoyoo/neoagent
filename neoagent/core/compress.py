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
_SUMMARIZE_SYSTEM = (
    "You are a context compression assistant. "
    "Summarize the following conversation history concisely, "
    "preserving all key facts, decisions, tool outputs, and context "
    "that would be needed to continue the conversation seamlessly. "
    "Output only the summary."
)

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
            system=_SUMMARIZE_SYSTEM,
            messages=[Message(role="user", content=middle_text)],
            tools=[],
        )
        summary = response.text_content.strip() or "(no summary)"
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
        return sanitized
