from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from neoagent.core.types import Message
from neoagent.memory.extractor import MemoryExtractor
from neoagent.memory.retriever import MemoryRetriever

if TYPE_CHECKING:
    from neoagent.memory.store import MemoryStore
    from neoagent.providers.base import Provider

logger = logging.getLogger(__name__)


class MemoryManager:
    """Orchestrates memory: triggers extraction and builds prompt sections."""

    def __init__(self, store: "MemoryStore", provider: "Provider") -> None:
        self._store = store
        self._retriever = MemoryRetriever(store)
        self._extractor = MemoryExtractor(provider, store)
        self._tool_calls_count: int = 0
        self._initial_token_estimate: int = 0

    def record_tool_calls(self, count: int) -> None:
        self._tool_calls_count += count

    async def maybe_extract(self, messages: list[Message], current_tokens: int) -> None:
        """Extract memories if trigger conditions are met.

        First call only sets the baseline; subsequent calls check the delta.
        """
        if self._initial_token_estimate == 0:
            self._initial_token_estimate = current_tokens
            return

        token_delta = max(0, current_tokens - self._initial_token_estimate)
        extracted = await self._extractor.extract(
            messages,
            tool_calls_count=self._tool_calls_count,
            token_delta=token_delta,
        )
        if extracted > 0:
            logger.info("Stored %d memory item(s)", extracted)
            self._tool_calls_count = 0

    def build_prompt_section(self, query: str | None = None) -> str:
        return self._retriever.retrieve(query)
