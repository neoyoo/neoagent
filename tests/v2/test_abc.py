"""Tests for neoagent/v2/abc.py — strict TDD per Shared Contracts C2.

Spec refs:
  § 2.3a  (lines 270-296)  — WorkingMemoryStore
  § 10.2  (lines 1610-1616) — CompressionStrategy
  § 11.2  (lines 1896+)    — MemoryReviewStrategy
  § 12.2  (lines 2019+)    — MemoryProvider
  § 15.8  (lines 2619+)    — SessionStorage
"""
from __future__ import annotations

import inspect
import pytest
from datetime import datetime


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_wm(session_id: str = "s1") -> "WorkingMemory":
    from neoagent.v2.schema import WorkingMemory
    return WorkingMemory(
        session_id=session_id,
        version=1,
        at_turn=0,
        constraints_and_preferences=[],
        progress="",
        key_decisions=[],
        relevant_files=[],
        next_steps=[],
        critical_context="",
        updated_by="framework_init",
        updated_at=datetime(2026, 4, 22),
    )


def _make_entry(user_id: str = "u1") -> "MemoryEntry":
    from neoagent.v2.schema import MemoryEntry
    return MemoryEntry(
        user_id=user_id,
        memory_id="mem_1",
        type="preference",
        category=None,
        content="test content",
    )


def _make_compression_context() -> "CompressionContext":
    from neoagent.v2.schema import CompressionContext
    return CompressionContext(
        session_id="s1",
        messages=[],
        previous_batches=[],
        previous_wm={},
        trigger="token_threshold",
    )


def _make_compression_delta() -> "CompressionDelta":
    from neoagent.v2.schema import CompressionDelta
    return CompressionDelta(
        batch_members=[],
        working_memory_delta=[],
    )


# ── WorkingMemoryStore ────────────────────────────────────────────────────────

class TestWorkingMemoryStore:
    """spec § 2.3a, lines 270-296"""

    def test_direct_instantiation_raises_type_error(self):
        from neoagent.v2.abc import WorkingMemoryStore
        with pytest.raises(TypeError):
            WorkingMemoryStore()  # type: ignore

    def test_partial_subclass_raises_type_error(self):
        from neoagent.v2.abc import WorkingMemoryStore

        class Partial(WorkingMemoryStore):
            async def get_current(self, session_id: str):
                return None
            # missing: save, get_version, list_versions

        with pytest.raises(TypeError):
            Partial()

    def test_full_subclass_instantiates(self):
        from neoagent.v2.abc import WorkingMemoryStore
        from neoagent.v2.schema import WorkingMemory

        class Concrete(WorkingMemoryStore):
            async def get_current(self, session_id: str) -> WorkingMemory | None:
                return None
            async def save(self, session_id: str, wm: WorkingMemory) -> None:
                pass
            async def get_version(self, session_id: str, version: int) -> WorkingMemory | None:
                return None
            async def list_versions(self, session_id: str) -> list[int]:
                return []

        obj = Concrete()
        assert isinstance(obj, WorkingMemoryStore)

    def test_get_current_signature(self):
        from neoagent.v2.abc import WorkingMemoryStore
        sig = inspect.signature(WorkingMemoryStore.get_current)
        params = list(sig.parameters.keys())
        assert params == ["self", "session_id"]

    def test_save_signature(self):
        from neoagent.v2.abc import WorkingMemoryStore
        sig = inspect.signature(WorkingMemoryStore.save)
        params = list(sig.parameters.keys())
        assert params == ["self", "session_id", "wm"]

    def test_get_version_signature(self):
        from neoagent.v2.abc import WorkingMemoryStore
        sig = inspect.signature(WorkingMemoryStore.get_version)
        params = list(sig.parameters.keys())
        assert params == ["self", "session_id", "version"]

    def test_list_versions_signature(self):
        from neoagent.v2.abc import WorkingMemoryStore
        sig = inspect.signature(WorkingMemoryStore.list_versions)
        params = list(sig.parameters.keys())
        assert params == ["self", "session_id"]

    def test_all_methods_are_abstract(self):
        from neoagent.v2.abc import WorkingMemoryStore
        for name in ("get_current", "save", "get_version", "list_versions"):
            method = getattr(WorkingMemoryStore, name)
            assert getattr(method, "__isabstractmethod__", False), f"{name} should be abstract"


# ── SessionStorage ────────────────────────────────────────────────────────────

class TestSessionStorage:
    """spec § 15.8, lines 2619+"""

    def test_direct_instantiation_raises_type_error(self):
        from neoagent.v2.abc import SessionStorage
        with pytest.raises(TypeError):
            SessionStorage()  # type: ignore

    def test_partial_subclass_raises_type_error(self):
        from neoagent.v2.abc import SessionStorage

        class Partial(SessionStorage):
            async def save_session(self, session_id: str, state_dict: dict) -> None:
                pass
            async def load_session(self, session_id: str) -> dict | None:
                return None
            # missing: save_messages, load_messages

        with pytest.raises(TypeError):
            Partial()

    def test_full_subclass_instantiates(self):
        from neoagent.v2.abc import SessionStorage

        class Concrete(SessionStorage):
            async def save_session(self, session_id: str, state_dict: dict) -> None:
                pass
            async def load_session(self, session_id: str) -> dict | None:
                return None
            async def save_messages(self, session_id: str, messages: list) -> None:
                pass
            async def load_messages(self, session_id: str) -> list:
                return []

        obj = Concrete()
        assert isinstance(obj, SessionStorage)

    def test_save_session_signature(self):
        from neoagent.v2.abc import SessionStorage
        sig = inspect.signature(SessionStorage.save_session)
        params = list(sig.parameters.keys())
        assert params == ["self", "session_id", "state_dict"]

    def test_load_session_signature(self):
        from neoagent.v2.abc import SessionStorage
        sig = inspect.signature(SessionStorage.load_session)
        params = list(sig.parameters.keys())
        assert params == ["self", "session_id"]

    def test_save_messages_signature(self):
        from neoagent.v2.abc import SessionStorage
        sig = inspect.signature(SessionStorage.save_messages)
        params = list(sig.parameters.keys())
        assert params == ["self", "session_id", "messages"]

    def test_load_messages_signature(self):
        from neoagent.v2.abc import SessionStorage
        sig = inspect.signature(SessionStorage.load_messages)
        params = list(sig.parameters.keys())
        assert params == ["self", "session_id"]

    def test_all_methods_are_abstract(self):
        from neoagent.v2.abc import SessionStorage
        for name in ("save_session", "load_session", "save_messages", "load_messages"):
            method = getattr(SessionStorage, name)
            assert getattr(method, "__isabstractmethod__", False), f"{name} should be abstract"


# ── CompressionStrategy ───────────────────────────────────────────────────────

class TestCompressionStrategy:
    """spec § 10.2, lines 1610-1616"""

    def test_direct_instantiation_raises_type_error(self):
        from neoagent.v2.abc import CompressionStrategy
        with pytest.raises(TypeError):
            CompressionStrategy()  # type: ignore

    def test_partial_subclass_without_compress_raises(self):
        from neoagent.v2.abc import CompressionStrategy

        class Partial(CompressionStrategy):
            pass  # compress not overridden

        with pytest.raises(TypeError):
            Partial()

    def test_full_subclass_instantiates(self):
        from neoagent.v2.abc import CompressionStrategy
        from neoagent.v2.schema import CompressionContext, CompressionDelta

        class Concrete(CompressionStrategy):
            async def compress(self, context: CompressionContext) -> CompressionDelta:
                return _make_compression_delta()

        obj = Concrete()
        assert isinstance(obj, CompressionStrategy)

    def test_compress_signature(self):
        from neoagent.v2.abc import CompressionStrategy
        sig = inspect.signature(CompressionStrategy.compress)
        params = list(sig.parameters.keys())
        assert params == ["self", "context"]

    def test_compress_is_abstract(self):
        from neoagent.v2.abc import CompressionStrategy
        method = getattr(CompressionStrategy, "compress")
        assert getattr(method, "__isabstractmethod__", False)


# ── MemoryReviewStrategy ──────────────────────────────────────────────────────

class TestMemoryReviewStrategy:
    """spec § 11.2, lines 1896+"""

    def test_direct_instantiation_raises_type_error(self):
        from neoagent.v2.abc import MemoryReviewStrategy
        with pytest.raises(TypeError):
            MemoryReviewStrategy()  # type: ignore

    def test_partial_subclass_without_review_raises(self):
        from neoagent.v2.abc import MemoryReviewStrategy

        class Partial(MemoryReviewStrategy):
            pass  # review not overridden

        with pytest.raises(TypeError):
            Partial()

    def test_full_subclass_instantiates(self):
        from neoagent.v2.abc import MemoryReviewStrategy
        from neoagent.v2.schema import WorkingMemory, MemoryEntry

        class Concrete(MemoryReviewStrategy):
            async def review(
                self, session_id: str, user_id: str, messages: list, wm: WorkingMemory
            ) -> list[MemoryEntry]:
                return []

        obj = Concrete()
        assert isinstance(obj, MemoryReviewStrategy)

    def test_review_signature(self):
        from neoagent.v2.abc import MemoryReviewStrategy
        sig = inspect.signature(MemoryReviewStrategy.review)
        params = list(sig.parameters.keys())
        assert params == ["self", "session_id", "user_id", "messages", "wm"]

    def test_review_is_abstract(self):
        from neoagent.v2.abc import MemoryReviewStrategy
        method = getattr(MemoryReviewStrategy, "review")
        assert getattr(method, "__isabstractmethod__", False)


# ── MemoryProvider ────────────────────────────────────────────────────────────

class TestMemoryProvider:
    """spec § 12.2, lines 2019+"""

    def test_direct_instantiation_raises_type_error(self):
        from neoagent.v2.abc import MemoryProvider
        with pytest.raises(TypeError):
            MemoryProvider()  # type: ignore

    def test_partial_subclass_raises_type_error(self):
        from neoagent.v2.abc import MemoryProvider
        from neoagent.v2.schema import MemoryEntry

        class Partial(MemoryProvider):
            async def search(self, user_id: str, query: str, k: int = 5) -> list[MemoryEntry]:
                return []
            async def upsert(self, entries: list[MemoryEntry]) -> None:
                pass
            # missing: delete, reinforce

        with pytest.raises(TypeError):
            Partial()

    def test_full_subclass_instantiates(self):
        from neoagent.v2.abc import MemoryProvider
        from neoagent.v2.schema import MemoryEntry

        class Concrete(MemoryProvider):
            async def search(self, user_id: str, query: str, k: int = 5) -> list[MemoryEntry]:
                return []
            async def upsert(self, entries: list[MemoryEntry]) -> None:
                pass
            async def delete(self, user_id: str, memory_ids: list[str]) -> None:
                pass
            async def reinforce(self, user_id: str, memory_id: str) -> None:
                pass

        obj = Concrete()
        assert isinstance(obj, MemoryProvider)

    def test_search_signature(self):
        from neoagent.v2.abc import MemoryProvider
        sig = inspect.signature(MemoryProvider.search)
        params = list(sig.parameters.keys())
        assert params == ["self", "user_id", "query", "k"]

    def test_search_k_default_is_5(self):
        from neoagent.v2.abc import MemoryProvider
        sig = inspect.signature(MemoryProvider.search)
        assert sig.parameters["k"].default == 5

    def test_upsert_signature(self):
        from neoagent.v2.abc import MemoryProvider
        sig = inspect.signature(MemoryProvider.upsert)
        params = list(sig.parameters.keys())
        assert params == ["self", "entries"]

    def test_delete_signature(self):
        from neoagent.v2.abc import MemoryProvider
        sig = inspect.signature(MemoryProvider.delete)
        params = list(sig.parameters.keys())
        assert params == ["self", "user_id", "memory_ids"]

    def test_reinforce_signature(self):
        from neoagent.v2.abc import MemoryProvider
        sig = inspect.signature(MemoryProvider.reinforce)
        params = list(sig.parameters.keys())
        assert params == ["self", "user_id", "memory_id"]

    def test_all_methods_are_abstract(self):
        from neoagent.v2.abc import MemoryProvider
        for name in ("search", "upsert", "delete", "reinforce"):
            method = getattr(MemoryProvider, name)
            assert getattr(method, "__isabstractmethod__", False), f"{name} should be abstract"
