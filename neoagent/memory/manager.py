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
    from neoagent.v2.abc import MemoryProvider
    from neoagent.v2.schema import MemoryEntry

logger = logging.getLogger(__name__)


class MemoryManager:
    """Orchestrates memory: triggers extraction and builds prompt sections.

    When *memory_provider* is supplied, extracted items are converted to
    ``MemoryEntry`` objects and persisted via ``provider.upsert()`` instead of
    the file-based ``MemoryStore``.  All other behaviour (trigger logic, token
    baseline tracking, etc.) is unchanged — back-compat is preserved when
    *memory_provider* is ``None`` (the default).
    """

    def __init__(
        self,
        store: "MemoryStore",
        provider: "Provider",
        *,
        memory_provider: "MemoryProvider | None" = None,
    ) -> None:
        self._store = store
        self._retriever = MemoryRetriever(store)
        self._extractor = MemoryExtractor(provider, store)
        self._memory_provider: "MemoryProvider | None" = memory_provider

    def record_tool_calls(self, count: int, session_state: "SessionState | None" = None) -> None:
        if session_state is not None:
            session_state.memory_tool_calls += count

    async def maybe_extract(
        self,
        messages: list[Message],
        current_tokens: int,
        session_state: "SessionState | None" = None,
    ) -> tuple[bool, int]:
        """Extract memories if trigger conditions are met.

        First call sets the token baseline. Extraction can still fire on the
        first call if tool_calls_count already meets the threshold.

        When *memory_provider* is set, extracted items are also persisted via
        ``provider.upsert()`` as ``MemoryEntry`` objects (in addition to the
        file-based ``MemoryStore`` write).

        Returns:
            (triggered, items_stored) — triggered is True when extraction ran,
            items_stored is the number of memory items written (0 if not triggered).
        """
        if session_state is not None:
            if session_state.memory_token_baseline == 0:
                session_state.memory_token_baseline = current_tokens

            token_delta = max(0, current_tokens - session_state.memory_token_baseline)

            # Snapshot topic list before extraction to detect new writes.
            topics_before = set(fn for fn, _ in self._store.list_topics())

            extracted = await self._extractor.extract(
                messages,
                tool_calls_count=session_state.memory_tool_calls,
                token_delta=token_delta,
            )
            if extracted > 0:
                logger.info("Stored %d memory item(s)", extracted)
                session_state.memory_tool_calls = 0
                session_state.memory_token_baseline = current_tokens

                # ── Task 7.3: mirror to MemoryProvider when configured ────────
                if self._memory_provider is not None:
                    await self._upsert_new_topics(topics_before)

                return (True, extracted)
            return (False, 0)
        else:
            # Fallback: no session_state — no-op (state tracking requires session)
            return (False, 0)

    async def _upsert_new_topics(self, topics_before: set[str]) -> None:
        """Convert newly written MemoryStore topics to MemoryEntry and upsert to provider."""
        from datetime import datetime
        from neoagent.v2.schema import MemoryEntry

        topics_after = {fn: desc for fn, desc in self._store.list_topics()}
        new_filenames = set(topics_after.keys()) - topics_before

        entries: list[MemoryEntry] = []
        for filename in new_filenames:
            content = self._store.read_topic(filename)
            description = topics_after.get(filename, filename)
            entries.append(MemoryEntry(
                user_id="default",
                memory_id=filename,
                type="fact",
                category=None,
                content=content or description,
                confidence=0.7,
                created_at=datetime.now(),
            ))

        if entries and self._memory_provider is not None:
            await self._memory_provider.upsert(entries)

    async def search_with_provider(
        self,
        user_id: str,
        query: str,
        k: int = 5,
    ) -> "list[MemoryEntry]":
        """Search via the injected MemoryProvider.

        Returns empty list when no provider is configured.
        """
        if self._memory_provider is None:
            return []
        return await self._memory_provider.search(user_id, query, k=k)

    def build_prompt_section(self, query: str | None = None) -> str:
        return self._retriever.retrieve(query)
