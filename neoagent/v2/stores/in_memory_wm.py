# neoagent/v2/stores/in_memory_wm.py
"""In-memory implementation of WorkingMemoryStore.

Spec refs: § 2.3a (lines 288-296)
Contract: C2 (WorkingMemoryStore)
"""
from __future__ import annotations

from neoagent.v2.abc import WorkingMemoryStore
from neoagent.v2.schema import WorkingMemory


class InMemoryWorkingMemoryStore(WorkingMemoryStore):
    """Append-only in-memory store keyed by session_id.

    Data structure: ``_data: dict[str, list[WorkingMemory]]``

    Version assignment is the caller's responsibility via ``wm.version``.
    The store preserves append-only semantics and never re-assigns versions.
    """

    def __init__(self) -> None:
        self._data: dict[str, list[WorkingMemory]] = {}

    async def save(self, session_id: str, wm: WorkingMemory) -> None:
        """Append *wm* to the version list for *session_id*."""
        if session_id not in self._data:
            self._data[session_id] = []
        self._data[session_id].append(wm)

    async def get_current(self, session_id: str) -> WorkingMemory | None:
        """Return the most-recently saved WorkingMemory, or None."""
        versions = self._data.get(session_id)
        if not versions:
            return None
        return versions[-1]

    async def get_version(self, session_id: str, version: int) -> WorkingMemory | None:
        """Return the WorkingMemory whose ``version`` field matches, or None.

        O(n) scan — acceptable for an in-memory store.
        """
        versions = self._data.get(session_id, [])
        for wm in versions:
            if wm.version == version:
                return wm
        return None

    async def list_versions(self, session_id: str) -> list[int]:
        """Return sorted list of version numbers saved for *session_id*."""
        versions = self._data.get(session_id, [])
        return sorted(wm.version for wm in versions)
