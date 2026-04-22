# neoagent/v2/abc.py
"""Abstract base classes for neoagent v2 pluggable interfaces.

Spec refs:
  § 2.3a  (lines 270-296)  — WorkingMemoryStore
  § 10.2  (lines 1610-1616) — CompressionStrategy
  § 11.2  (lines 1896+)    — MemoryReviewStrategy
  § 12.2  (lines 2019+)    — MemoryProvider
  § 15.8  (lines 2619+)    — SessionStorage

Contract: C2 (Shared Contracts 2026-04-22)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from neoagent.v2.schema import (
    CompressionContext,
    CompressionDelta,
    MemoryEntry,
    WorkingMemory,
)


class WorkingMemoryStore(ABC):
    """spec § 2.3a, lines 270-296"""

    @abstractmethod
    async def get_current(self, session_id: str) -> WorkingMemory | None: ...

    @abstractmethod
    async def save(self, session_id: str, wm: WorkingMemory) -> None: ...

    @abstractmethod
    async def get_version(self, session_id: str, version: int) -> WorkingMemory | None: ...

    @abstractmethod
    async def list_versions(self, session_id: str) -> list[int]: ...


class SessionStorage(ABC):
    """spec § 15.8, lines 2619+"""

    @abstractmethod
    async def save_session(self, session_id: str, state_dict: dict) -> None: ...

    @abstractmethod
    async def load_session(self, session_id: str) -> dict | None: ...

    @abstractmethod
    async def save_messages(self, session_id: str, messages: list) -> None: ...

    @abstractmethod
    async def load_messages(self, session_id: str) -> list: ...


class CompressionStrategy(ABC):
    """spec § 10.2, lines 1610-1616"""

    @abstractmethod
    async def compress(self, context: CompressionContext) -> CompressionDelta: ...


class MemoryReviewStrategy(ABC):
    """spec § 11.2, lines 1896+"""

    @abstractmethod
    async def review(self, session_id: str, user_id: str, messages: list, wm: WorkingMemory) -> list[MemoryEntry]: ...


class MemoryProvider(ABC):
    """spec § 12.2, lines 2019+"""

    @abstractmethod
    async def search(self, user_id: str, query: str, k: int = 5) -> list[MemoryEntry]: ...

    @abstractmethod
    async def upsert(self, entries: list[MemoryEntry]) -> None: ...

    @abstractmethod
    async def delete(self, user_id: str, memory_ids: list[str]) -> None: ...

    @abstractmethod
    async def reinforce(self, user_id: str, memory_id: str) -> None: ...
