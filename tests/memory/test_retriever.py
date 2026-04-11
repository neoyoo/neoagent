from __future__ import annotations
import pytest
from pathlib import Path
from neoagent.memory.store import MemoryStore
from neoagent.memory.retriever import MemoryRetriever


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path)


def test_retrieve_empty_store(store: MemoryStore) -> None:
    r = MemoryRetriever(store)
    assert r.retrieve() == ""


def test_retrieve_no_query_returns_index_only(store: MemoryStore) -> None:
    store.write_index("# Memory\n- [goals](goals.md)\n")
    store.write_topic("goals.md", "# Goals\n- Build stuff\n")
    r = MemoryRetriever(store)
    result = r.retrieve(query=None)
    assert "goals" in result
    assert "Build stuff" not in result


def test_retrieve_with_matching_query(store: MemoryStore) -> None:
    store.write_index("# Memory\n- [user preferences](prefs.md)\n")
    store.write_topic("prefs.md", "# User Preferences\n- Prefers Python\n")
    r = MemoryRetriever(store)
    result = r.retrieve(query="user preferences")
    assert "Prefers Python" in result


def test_retrieve_no_match_returns_index(store: MemoryStore) -> None:
    store.write_index("# Memory\n- [coding style](style.md)\n")
    store.write_topic("style.md", "# Coding Style\n- Use black\n")
    r = MemoryRetriever(store)
    result = r.retrieve(query="completely unrelated xyz123")
    assert result != ""


def test_retrieve_content_truncated(store: MemoryStore) -> None:
    store.write_index("# Memory\n- [big file](big.md)\n")
    long_content = "x" * 5000
    store.write_topic("big.md", long_content)
    r = MemoryRetriever(store, max_content_chars=2000)
    result = r.retrieve(query="big")
    assert len(result) < 5000 + 200
