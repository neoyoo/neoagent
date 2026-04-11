from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from neoagent.core.types import Message
from neoagent.memory.extractor import MemoryExtractor
from neoagent.memory.retriever import MemoryRetriever

if TYPE_CHECKING:
    from neoagent.memory.store import MemoryStore
    from neoagent.providers.base import Provider
    from neoagent.session import SessionState

logger = logging.getLogger(__name__)


class MemoryManager:
    """Orchestrates memory: triggers extraction and builds prompt sections."""

    def __init__(self, store: "MemoryStore", provider: "Provider") -> None:
        self._store = store
        self._retriever = MemoryRetriever(store)
        self._extractor = MemoryExtractor(provider, store)

    def record_tool_calls(self, count: int, session_state: "SessionState | None" = None) -> None:
        if session_state is not None:
            session_state.memory_tool_calls += count

    async def maybe_extract(
        self,
        messages: list[Message],
        current_tokens: int,
        session_state: "SessionState | None" = None,
    ) -> None:
        """Extract memories if trigger conditions are met.

        First call sets the token baseline. Extraction can still fire on the
        first call if tool_calls_count already meets the threshold.
        """
        if session_state is not None:
            if session_state.memory_token_baseline == 0:
                session_state.memory_token_baseline = current_tokens

            token_delta = max(0, current_tokens - session_state.memory_token_baseline)
            extracted = await self._extractor.extract(
                messages,
                tool_calls_count=session_state.memory_tool_calls,
                token_delta=token_delta,
            )
            if extracted > 0:
                logger.info("Stored %d memory item(s)", extracted)
                session_state.memory_tool_calls = 0
                session_state.memory_token_baseline = current_tokens
        else:
            # Fallback: no session_state — no-op (state tracking requires session)
            pass

    def build_prompt_section(self, query: str | None = None) -> str:
        return self._retriever.retrieve(query)
