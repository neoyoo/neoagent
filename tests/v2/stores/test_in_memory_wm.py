# tests/v2/stores/test_in_memory_wm.py
"""TDD tests for InMemoryWorkingMemoryStore.

Spec refs: § 2.3a (lines 288-296)
"""
from __future__ import annotations

import pytest
from datetime import datetime

from neoagent.v2.schema import WorkingMemory
from neoagent.v2.stores.in_memory_wm import InMemoryWorkingMemoryStore


def _make_wm(session_id: str, version: int, at_turn: int = 1) -> WorkingMemory:
    return WorkingMemory(
        session_id=session_id,
        version=version,
        at_turn=at_turn,
        goal="test goal",
        constraints_and_preferences=[],
        progress="",
        key_decisions=[],
        relevant_files=[],
        next_steps=[],
        critical_context="",
        updated_by="framework_init",
        updated_at=datetime(2026, 1, 1),
    )


@pytest.mark.asyncio
async def test_save_and_get_current():
    store = InMemoryWorkingMemoryStore()
    sid = "session-a"
    wm1 = _make_wm(sid, version=1)
    wm2 = _make_wm(sid, version=2, at_turn=2)

    await store.save(sid, wm1)
    await store.save(sid, wm2)

    result = await store.get_current(sid)
    assert result is not None
    assert result.version == 2


@pytest.mark.asyncio
async def test_get_current_empty():
    store = InMemoryWorkingMemoryStore()
    result = await store.get_current("nonexistent-session")
    assert result is None


@pytest.mark.asyncio
async def test_get_version():
    store = InMemoryWorkingMemoryStore()
    sid = "session-b"
    for v in [1, 2, 3]:
        await store.save(sid, _make_wm(sid, version=v))

    result = await store.get_version(sid, 2)
    assert result is not None
    assert result.version == 2


@pytest.mark.asyncio
async def test_get_version_missing():
    store = InMemoryWorkingMemoryStore()
    sid = "session-c"
    await store.save(sid, _make_wm(sid, version=1))

    result = await store.get_version(sid, 99)
    assert result is None


@pytest.mark.asyncio
async def test_list_versions_sorted():
    store = InMemoryWorkingMemoryStore()
    sid = "session-d"
    for v in [3, 1, 2]:
        await store.save(sid, _make_wm(sid, version=v))

    versions = await store.list_versions(sid)
    assert versions == [1, 2, 3]


@pytest.mark.asyncio
async def test_multiple_sessions_isolated():
    store = InMemoryWorkingMemoryStore()
    wm_a = _make_wm("session-a", version=1)
    wm_b = _make_wm("session-b", version=42)

    await store.save("session-a", wm_a)
    await store.save("session-b", wm_b)

    result_a = await store.get_current("session-a")
    result_b = await store.get_current("session-b")

    assert result_a is not None and result_a.version == 1
    assert result_b is not None and result_b.version == 42

    versions_a = await store.list_versions("session-a")
    versions_b = await store.list_versions("session-b")
    assert versions_a == [1]
    assert versions_b == [42]
