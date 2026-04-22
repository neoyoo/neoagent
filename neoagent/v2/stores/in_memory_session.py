# neoagent/v2/stores/in_memory_session.py
"""In-memory implementation of SessionStorage.

Spec refs: § 15.8 (lines 2619+)
Contract: C2 (SessionStorage)
"""
from __future__ import annotations

import copy

from neoagent.v2.abc import SessionStorage


class InMemorySessionStorage(SessionStorage):
    """Pure in-memory session + message store.

    All public methods return copies so that callers cannot mutate
    internal storage by modifying the returned objects.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}
        self._messages: dict[str, list] = {}

    async def save_session(self, session_id: str, state_dict: dict) -> None:
        """Deep-copy *state_dict* into internal storage."""
        self._sessions[session_id] = copy.deepcopy(state_dict)

    async def load_session(self, session_id: str) -> dict | None:
        """Return a deep-copy of the stored state, or None if absent."""
        stored = self._sessions.get(session_id)
        if stored is None:
            return None
        return copy.deepcopy(stored)

    async def save_messages(self, session_id: str, messages: list) -> None:
        """Store a shallow copy of *messages*."""
        self._messages[session_id] = list(messages)

    async def load_messages(self, session_id: str) -> list:
        """Return a shallow copy of stored messages, or [] if absent."""
        return list(self._messages.get(session_id, []))
