# tests/v2/stores/test_in_memory_session.py
"""TDD tests for InMemorySessionStorage.

Spec refs: § 15.8 (lines 2619+)
"""
from __future__ import annotations

import pytest

from neoagent.v2.stores.in_memory_session import InMemorySessionStorage


@pytest.mark.asyncio
async def test_save_load_session():
    store = InMemorySessionStorage()
    state = {"status": "active", "turn": 5}
    await store.save_session("s1", state)

    loaded = await store.load_session("s1")
    assert loaded == {"status": "active", "turn": 5}


@pytest.mark.asyncio
async def test_load_session_missing():
    store = InMemorySessionStorage()
    result = await store.load_session("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_save_load_messages():
    store = InMemorySessionStorage()
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    await store.save_messages("s1", messages)

    loaded = await store.load_messages("s1")
    assert loaded == messages


@pytest.mark.asyncio
async def test_load_messages_missing():
    store = InMemorySessionStorage()
    result = await store.load_messages("nonexistent")
    assert result == []


@pytest.mark.asyncio
async def test_saved_data_independent():
    """Mutations to returned data must not affect stored data."""
    store = InMemorySessionStorage()

    # Test session isolation
    original_state = {"key": "value", "nested": {"x": 1}}
    await store.save_session("s1", original_state)

    # Mutate original dict after saving — should not affect store
    original_state["key"] = "MUTATED"

    loaded = await store.load_session("s1")
    assert loaded["key"] == "value"

    # Mutate the returned value — should not affect store
    loaded["key"] = "MUTATED_RETURN"
    reloaded = await store.load_session("s1")
    assert reloaded["key"] == "value"

    # Test messages isolation
    original_msgs = [{"role": "user", "content": "hello"}]
    await store.save_messages("s1", original_msgs)

    # Mutate original list after saving
    original_msgs.append({"role": "assistant", "content": "intruder"})

    loaded_msgs = await store.load_messages("s1")
    assert len(loaded_msgs) == 1

    # Mutate returned list
    loaded_msgs.append({"role": "assistant", "content": "intruder2"})
    reloaded_msgs = await store.load_messages("s1")
    assert len(reloaded_msgs) == 1
