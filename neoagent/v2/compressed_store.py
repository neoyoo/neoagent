"""CompressedMessageStore — pluggable backend for compressed message originals.

Compression writes message bodies here when trimming session.messages.
recall_turn fetches them back by msg_id.

Default impl (InMemoryCompressedMessageStore) wraps session_state.compressed_messages
so SessionStorage (e.g. JsonFileStorage) continues to round-trip bodies inside the
session record. To relocate bodies independently from the session record, provide
a custom CompressedMessageStore (e.g. ES, Qdrant, Postgres).
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable

from neoagent.core.types import Message


@runtime_checkable
class CompressedMessageStore(Protocol):
    """Pluggable storage for the bodies of messages that compression has trimmed
    out of the live messages window.

    All operations are async to permit network/IO-bound implementations. The
    default InMemory impl is async-compatible (no actual await), so callers
    must always `await` regardless of backend.

    session_id partitions data; an instance is meant to serve multiple sessions.
    msg_id uniquely identifies a message within a session.
    """

    async def put_many(self, session_id: str, messages: list[Message]) -> None:
        """Persist N message bodies. Idempotent on (session_id, msg.id) collision —
        duplicates are silently skipped, never raise. Order preserved across
        successive put_many calls when the backend supports it."""
        ...

    async def get_many(self, session_id: str, msg_ids: list[str]) -> dict[str, Message]:
        """Fetch message bodies by id. Returns dict[msg_id → Message] containing
        only ids that exist. Missing ids are silently absent — caller checks and
        reports them in the recall result's `missing` field."""
        ...

    async def list_ids(self, session_id: str) -> list[str]:
        """All msg_ids stored for this session, in insertion order if the backend
        preserves it."""
        ...

    async def delete_session(self, session_id: str) -> None:
        """Remove all entries for the given session. Idempotent — never raises
        if session is unknown."""
        ...


class InMemoryCompressedMessageStore:
    """Default implementation. Wraps each session's `state.compressed_messages`
    list as the backing store. The list reference is shared, so:
    - `put_many` extends the same list that SessionStorage will serialize.
    - JsonFileStorage save/load round-trips bodies via the existing
      session JSON — no schema change required.

    A session must be `bind_session(...)`-ed before put/get/list calls, so the
    store knows which list to use. Agent.new_session() / Agent.resume() do this
    automatically when the configured store is an InMemoryCompressedMessageStore.

    bind_session is NOT part of the Protocol — it's an implementation detail
    of this in-memory backend. Remote backends (ES, Postgres) don't need it.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, list[Message]] = {}

    def bind_session(self, session_id: str, compressed_messages_list: list[Message]) -> None:
        """Bind a SessionState's compressed_messages list as the backing store
        for this session. Call once per session lifetime (on new_session/resume).
        Re-binding the same id replaces the previous reference."""
        self._buckets[session_id] = compressed_messages_list

    async def put_many(self, session_id: str, messages: list[Message]) -> None:
        bucket = self._buckets.get(session_id)
        if bucket is None:
            raise RuntimeError(
                f"InMemoryCompressedMessageStore: session {session_id!r} not bound. "
                f"Call bind_session() before put_many()."
            )
        existing_ids = {m.id for m in bucket if m.id}
        for msg in messages:
            if msg.id and msg.id in existing_ids:
                continue  # idempotent skip
            bucket.append(msg)
            if msg.id:
                existing_ids.add(msg.id)

    async def get_many(self, session_id: str, msg_ids: list[str]) -> dict[str, Message]:
        bucket = self._buckets.get(session_id, [])
        index = {m.id: m for m in bucket if m.id}
        return {mid: index[mid] for mid in msg_ids if mid in index}

    async def list_ids(self, session_id: str) -> list[str]:
        return [m.id for m in self._buckets.get(session_id, []) if m.id]

    async def delete_session(self, session_id: str) -> None:
        # Note: clears the store's reference, but does NOT clear the underlying
        # SessionState.compressed_messages list (which may still be in use).
        # Remote backends would actually delete data here.
        self._buckets.pop(session_id, None)
