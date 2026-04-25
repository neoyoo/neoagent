# tests/v2/test_compressed_store.py
"""Unit tests for InMemoryCompressedMessageStore and CompressedMessageStore Protocol.

Spec refs: compressed_store.py contract
"""
from __future__ import annotations

import pytest

from neoagent.core.types import Message
from neoagent.v2.compressed_store import CompressedMessageStore, InMemoryCompressedMessageStore


def _msg(id: str, content: str = "body", role: str = "user") -> Message:
    return Message(id=id, role=role, content=content)


# ── 1. put → get round-trip ───────────────────────────────────────────────────


class TestPutGetRoundtrip:
    @pytest.mark.asyncio
    async def test_put_get_roundtrip(self):
        """put 3 messages, get all 3 back."""
        store = InMemoryCompressedMessageStore()
        bucket: list[Message] = []
        store.bind_session("s1", bucket)

        msgs = [_msg("m1", "first"), _msg("m2", "second"), _msg("m3", "third")]
        await store.put_many("s1", msgs)

        result = await store.get_many("s1", ["m1", "m2", "m3"])
        assert len(result) == 3
        assert result["m1"].content == "first"
        assert result["m2"].content == "second"
        assert result["m3"].content == "third"


# ── 2. get with missing ids returns only found entries ────────────────────────


class TestGetMissingIdsReturnsPartial:
    @pytest.mark.asyncio
    async def test_get_missing_ids_returns_partial(self):
        """Request 5 ids, only 2 stored, returns dict with 2 entries."""
        store = InMemoryCompressedMessageStore()
        bucket: list[Message] = []
        store.bind_session("s1", bucket)

        await store.put_many("s1", [_msg("m1"), _msg("m2")])

        result = await store.get_many("s1", ["m1", "m2", "m3", "m4", "m5"])
        assert set(result.keys()) == {"m1", "m2"}


# ── 3. put idempotent on duplicate id ─────────────────────────────────────────


class TestPutIdempotentOnDuplicateId:
    @pytest.mark.asyncio
    async def test_put_idempotent_on_duplicate_id(self):
        """put same msg twice, list_ids returns 1 entry."""
        store = InMemoryCompressedMessageStore()
        bucket: list[Message] = []
        store.bind_session("s1", bucket)

        msg = _msg("m1", "original")
        await store.put_many("s1", [msg])
        await store.put_many("s1", [msg])  # duplicate

        ids = await store.list_ids("s1")
        assert ids.count("m1") == 1
        assert len(ids) == 1


# ── 4. put preserves order within call ────────────────────────────────────────


class TestPutPreservesOrderWithinCall:
    @pytest.mark.asyncio
    async def test_put_preserves_order_within_call(self):
        """put [m3, m1, m2] returns same order from list_ids."""
        store = InMemoryCompressedMessageStore()
        bucket: list[Message] = []
        store.bind_session("s1", bucket)

        await store.put_many("s1", [_msg("m3"), _msg("m1"), _msg("m2")])

        ids = await store.list_ids("s1")
        assert ids == ["m3", "m1", "m2"]


# ── 5. put raises when session not bound ─────────────────────────────────────


class TestPutRaisesWhenSessionNotBound:
    @pytest.mark.asyncio
    async def test_put_raises_when_session_not_bound(self):
        """RuntimeError on unbound session_id."""
        store = InMemoryCompressedMessageStore()

        with pytest.raises(RuntimeError, match="not bound"):
            await store.put_many("unbound-session", [_msg("m1")])


# ── 6. bind_session replaces previous reference ───────────────────────────────


class TestBindSessionReplacesPreviousReference:
    @pytest.mark.asyncio
    async def test_bind_session_replaces_previous_reference(self):
        """Re-bind with new list, get reflects new list."""
        store = InMemoryCompressedMessageStore()

        old_bucket: list[Message] = [_msg("m1", "old")]
        store.bind_session("s1", old_bucket)

        new_bucket: list[Message] = [_msg("m2", "new")]
        store.bind_session("s1", new_bucket)  # replace

        result = await store.get_many("s1", ["m1", "m2"])
        # m1 was in old bucket — no longer accessible
        assert "m1" not in result
        # m2 is in new bucket
        assert result["m2"].content == "new"


# ── 7. delete_session clears internal buckets ─────────────────────────────────


class TestDeleteSessionClearsInternalBuckets:
    @pytest.mark.asyncio
    async def test_delete_session_clears_internal_buckets(self):
        """After delete, list_ids returns []."""
        store = InMemoryCompressedMessageStore()
        bucket: list[Message] = []
        store.bind_session("s1", bucket)
        await store.put_many("s1", [_msg("m1"), _msg("m2")])

        await store.delete_session("s1")

        ids = await store.list_ids("s1")
        assert ids == []

    @pytest.mark.asyncio
    async def test_delete_session_unknown_is_idempotent(self):
        """delete_session on unknown session_id does not raise."""
        store = InMemoryCompressedMessageStore()
        await store.delete_session("does-not-exist")  # should not raise


# ── 8. multiple sessions are isolated ────────────────────────────────────────


class TestMultipleSessionsIsolated:
    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self):
        """put for sid1 doesn't affect get for sid2."""
        store = InMemoryCompressedMessageStore()
        bucket1: list[Message] = []
        bucket2: list[Message] = []
        store.bind_session("s1", bucket1)
        store.bind_session("s2", bucket2)

        await store.put_many("s1", [_msg("m1", "from-s1")])
        await store.put_many("s2", [_msg("m2", "from-s2")])

        result_s1 = await store.get_many("s1", ["m1", "m2"])
        result_s2 = await store.get_many("s2", ["m1", "m2"])

        assert set(result_s1.keys()) == {"m1"}
        assert result_s1["m1"].content == "from-s1"

        assert set(result_s2.keys()) == {"m2"}
        assert result_s2["m2"].content == "from-s2"


# ── 9. Protocol runtime_checkable ────────────────────────────────────────────


class TestProtocolRuntimeCheckable:
    def test_protocol_runtime_checkable(self):
        """isinstance(InMemory(), CompressedMessageStore) is True."""
        store = InMemoryCompressedMessageStore()
        assert isinstance(store, CompressedMessageStore)
