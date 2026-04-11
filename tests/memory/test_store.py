from __future__ import annotations
import pytest
from pathlib import Path
from neoagent.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path)


def test_empty_index(store: MemoryStore) -> None:
    assert store.read_index() == ""


def test_write_read_index(store: MemoryStore) -> None:
    store.write_index("# Memory\n- [goals](goals.md)\n")
    assert "goals" in store.read_index()


def test_list_topics_empty(store: MemoryStore) -> None:
    assert store.list_topics() == []


def test_list_topics_parses_links(store: MemoryStore) -> None:
    store.write_index("# Memory\n- [user preferences](prefs.md)\n- [project state](state.md)\n")
    topics = store.list_topics()
    assert topics == [("prefs.md", "user preferences"), ("state.md", "project state")]


def test_write_read_topic(store: MemoryStore) -> None:
    store.write_topic("goals.md", "# Goals\n- Build a great agent\n")
    assert "Build a great agent" in store.read_topic("goals.md")


def test_read_missing_topic(store: MemoryStore) -> None:
    assert store.read_topic("nonexistent.md") == ""


def test_delete_topic(store: MemoryStore) -> None:
    store.write_topic("temp.md", "content")
    store.delete_topic("temp.md")
    assert store.read_topic("temp.md") == ""


def test_delete_missing_topic_no_error(store: MemoryStore) -> None:
    store.delete_topic("nonexistent.md")  # should not raise


def test_read_topic_path_traversal_blocked(store: MemoryStore) -> None:
    """Path traversal via filename must raise ValueError (not silently escape the dir)."""
    with pytest.raises(ValueError, match="escapes memory directory"):
        store.read_topic("../../etc/passwd")


def test_write_topic_path_traversal_blocked(store: MemoryStore) -> None:
    with pytest.raises(ValueError, match="escapes memory directory"):
        store.write_topic("../../tmp/evil.md", "bad content")


def test_delete_topic_path_traversal_blocked(store: MemoryStore) -> None:
    with pytest.raises(ValueError, match="escapes memory directory"):
        store.delete_topic("../../tmp/evil.md")
