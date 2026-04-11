from __future__ import annotations
import json
import pytest
from pathlib import Path
from datetime import datetime
from neoagent.session import Session, SessionState, JsonFileStorage, _session_to_dict, _session_from_dict
from neoagent.core.types import Message


def test_session_state_defaults():
    s = SessionState()
    assert s.previous_summary is None
    assert s.memory_tool_calls == 0
    assert s.memory_token_baseline == 0
    assert s.total_input_tokens == 0
    assert s.total_output_tokens == 0


def test_session_create_generates_uuid():
    s = Session.create()
    assert len(s.id) == 36
    assert s.messages == []
    assert isinstance(s.state, SessionState)
    assert isinstance(s.created_at, datetime)
    assert isinstance(s.updated_at, datetime)


def test_session_create_with_explicit_id():
    s = Session.create(session_id="my-session")
    assert s.id == "my-session"


def test_session_create_two_different_ids():
    a = Session.create()
    b = Session.create()
    assert a.id != b.id


def test_session_fork_independent_messages():
    s = Session.create()
    s.messages.append(Message(role="user", content="hello"))
    forked = s.fork()
    forked.messages.append(Message(role="assistant", content="hi"))
    assert len(s.messages) == 1
    assert len(forked.messages) == 2


def test_session_fork_independent_state():
    s = Session.create()
    s.state.total_input_tokens = 100
    forked = s.fork()
    forked.state.total_input_tokens = 999
    assert s.state.total_input_tokens == 100


def test_session_fork_new_id():
    s = Session.create()
    forked = s.fork()
    assert forked.id != s.id


def test_session_fork_explicit_id():
    s = Session.create()
    forked = s.fork(new_id="fork-1")
    assert forked.id == "fork-1"


@pytest.fixture
def storage(tmp_path: Path) -> JsonFileStorage:
    return JsonFileStorage(tmp_path)


def test_storage_save_creates_file(storage: JsonFileStorage, tmp_path: Path):
    s = Session.create(session_id="abc")
    storage.save(s)
    assert (tmp_path / "abc.json").exists()


def test_storage_load_roundtrip(storage: JsonFileStorage):
    s = Session.create(session_id="xyz")
    s.messages.append(Message(role="user", content="test message"))
    s.state.previous_summary = "some summary"
    s.state.total_input_tokens = 42
    storage.save(s)
    loaded = storage.load("xyz")
    assert loaded.id == "xyz"
    assert len(loaded.messages) == 1
    assert loaded.messages[0].role == "user"
    assert loaded.messages[0].content == "test message"
    assert loaded.state.previous_summary == "some summary"
    assert loaded.state.total_input_tokens == 42


def test_storage_load_missing_raises_key_error(storage: JsonFileStorage):
    with pytest.raises(KeyError, match="Session not found"):
        storage.load("nonexistent")


def test_storage_list_ids(storage: JsonFileStorage):
    storage.save(Session.create(session_id="a"))
    storage.save(Session.create(session_id="b"))
    ids = storage.list_ids()
    assert set(ids) == {"a", "b"}


def test_storage_delete(storage: JsonFileStorage, tmp_path: Path):
    s = Session.create(session_id="del-me")
    storage.save(s)
    storage.delete("del-me")
    assert not (tmp_path / "del-me.json").exists()


def test_storage_delete_missing_no_error(storage: JsonFileStorage):
    storage.delete("nonexistent")


def test_session_resume(storage: JsonFileStorage):
    s = Session.create(session_id="resume-me")
    s.state.total_output_tokens = 77
    storage.save(s)
    loaded = Session.resume("resume-me", storage)
    assert loaded.state.total_output_tokens == 77


def test_session_resume_missing(storage: JsonFileStorage):
    with pytest.raises(KeyError):
        Session.resume("ghost", storage)


def test_session_save_updates_updated_at(storage: JsonFileStorage):
    s = Session.create(session_id="save-me")
    old_ts = s.updated_at
    s.save(storage)
    loaded = storage.load("save-me")
    assert loaded.updated_at >= old_ts


def test_session_to_dict_round_trip():
    s = Session.create(session_id="roundtrip")
    s.messages.append(Message(role="user", content="hello"))
    s.state.memory_tool_calls = 3
    d = _session_to_dict(s)
    assert isinstance(d, dict)
    s2 = _session_from_dict(d)
    assert s2.id == "roundtrip"
    assert s2.messages[0].content == "hello"
    assert s2.state.memory_tool_calls == 3


def test_session_to_dict_is_json_serializable():
    s = Session.create()
    s.messages.append(Message(role="assistant", content="ok"))
    d = _session_to_dict(s)
    json.dumps(d)


# ── SessionState.promoted_tools (Fix 6: session-level deferred promote) ───────

def test_session_state_promoted_tools_default_empty():
    """promoted_tools must default to an empty set."""
    state = SessionState()
    assert isinstance(state.promoted_tools, set)
    assert len(state.promoted_tools) == 0


def test_session_state_promoted_tools_independent_per_instance():
    """Each SessionState must have its own promoted_tools set (not shared)."""
    s1 = SessionState()
    s2 = SessionState()
    s1.promoted_tools.add("github__create_issue")
    assert "github__create_issue" not in s2.promoted_tools


def test_session_fork_independent_promoted_tools():
    """Forked session must have an independent copy of promoted_tools."""
    s = Session.create()
    s.state.promoted_tools.add("github__create_issue")
    forked = s.fork()
    forked.state.promoted_tools.add("filesystem__read_file")
    # Original must not see changes from fork
    assert "filesystem__read_file" not in s.state.promoted_tools
    # Fork must have original's promoted tools
    assert "github__create_issue" in forked.state.promoted_tools


def test_two_sessions_have_independent_promote_state():
    """Two sessions sharing a DeferredToolRegistry must have independent promote sets."""
    from neoagent.tools.deferred import DeferredToolRegistry
    deferred = DeferredToolRegistry()
    deferred.register("github__create_issue", "Create issue")

    session_a = Session.create()
    session_b = Session.create()

    # Session A promotes the tool
    session_a.state.promoted_tools.add("github__create_issue")

    # Session B must NOT see Session A's promote
    assert "github__create_issue" not in session_b.state.promoted_tools

    # Verify global registry is still consistent
    assert deferred.is_deferred("github__create_issue")


def test_session_promoted_tools_serialized_and_deserialized():
    """promoted_tools must survive a save/load round-trip."""
    s = Session.create(session_id="promote-rt")
    s.state.promoted_tools = {"github__create_issue", "filesystem__read_file"}
    d = _session_to_dict(s)
    s2 = _session_from_dict(d)
    assert s2.state.promoted_tools == {"github__create_issue", "filesystem__read_file"}


def test_session_promoted_tools_empty_serializes_as_empty_list():
    """Empty promoted_tools serializes as [] in the dict."""
    s = Session.create(session_id="empty-pt")
    d = _session_to_dict(s)
    assert d["state"]["promoted_tools"] == []


def test_session_promoted_tools_deserialized_missing_key_defaults_to_empty():
    """If promoted_tools is missing from stored dict (old format), defaults to empty."""
    d = {
        "id": "old-session",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "state": {
            "previous_summary": None,
            "compression_failures": 0,
            "memory_tool_calls": 0,
            "memory_token_baseline": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            # No "promoted_tools" key — simulates old persisted session
        },
        "messages": [],
    }
    s = _session_from_dict(d)
    assert s.state.promoted_tools == set()
