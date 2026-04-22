# tests/v2/test_session_state_v2.py
"""TDD tests for Task 4.2 — SessionState v2 extension.

Tests cover:
  A. New field existence and defaults
  B. Serialization round-trips (to_dict / from_dict)
  C. Backward compatibility with old session dicts
  D. Full Session save/load round-trip
  E. WorkingMemory datetime serialization

Spec refs: § 2.3a, § 2.7, § 8.2, § 11.4, § 15.3
"""
from __future__ import annotations

import pytest
from datetime import datetime
from pathlib import Path

from neoagent.session import (
    Session,
    SessionState,
    JsonFileStorage,
    _session_to_dict,
    _session_from_dict,
)
from neoagent.v2.schema import WorkingMemory
from neoagent.v2.id_gen import SessionIdGenerator
from neoagent.v2.nudge import NudgeCounter


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_wm(session_id: str = "s1", version: int = 1, at_turn: int = 0) -> WorkingMemory:
    return WorkingMemory(
        session_id=session_id,
        version=version,
        at_turn=at_turn,
        goal="test goal",
        constraints_and_preferences=["c01: keep it short"],
        progress="in progress",
        key_decisions=["d01: use TDD"],
        relevant_files=["session.py"],
        next_steps=["n01: write tests"],
        critical_context="none",
        updated_by="framework_init",
        updated_at=datetime(2026, 4, 22, 10, 0, 0),
    )


# ── A. Field existence and defaults ──────────────────────────────────────────

class TestSessionStateNewFieldDefaults:
    def test_current_wm_defaults_to_none(self):
        """New SessionState._current_wm must be None by default."""
        state = SessionState()
        assert state._current_wm is None

    def test_id_gen_defaults_to_session_id_generator_instance(self):
        """New SessionState.id_gen must be a SessionIdGenerator instance."""
        state = SessionState()
        assert isinstance(state.id_gen, SessionIdGenerator)

    def test_nudge_counter_defaults_to_nudge_counter_instance(self):
        """New SessionState.nudge_counter must be a NudgeCounter instance."""
        state = SessionState()
        assert isinstance(state.nudge_counter, NudgeCounter)

    def test_id_gen_fresh_per_instance(self):
        """Each SessionState must have its own independent id_gen."""
        s1 = SessionState()
        s2 = SessionState()
        s1.id_gen.next_msg_id()  # advance s1 counter
        assert s2.id_gen._msg_counter == 0  # s2 unaffected

    def test_nudge_counter_fresh_per_instance(self):
        """Each SessionState must have its own independent nudge_counter."""
        s1 = SessionState()
        s2 = SessionState()
        s1.nudge_counter.tick()
        assert s2.nudge_counter._count == 0

    def test_nudge_counter_default_threshold(self):
        """Default nudge_counter threshold must be 10."""
        state = SessionState()
        assert state.nudge_counter.threshold == 10


# ── B. Serialization round-trips ─────────────────────────────────────────────

class TestSessionStateRoundTrip:
    def test_default_session_state_round_trip(self):
        """Default SessionState: to_dict → from_dict must recover equivalent state."""
        session = Session.create(session_id="rt-default")
        d = _session_to_dict(session)
        recovered = _session_from_dict(d)

        assert recovered.state._current_wm is None
        assert isinstance(recovered.state.id_gen, SessionIdGenerator)
        assert recovered.state.id_gen._msg_counter == 0
        assert isinstance(recovered.state.nudge_counter, NudgeCounter)
        assert recovered.state.nudge_counter._count == 0
        assert recovered.state.nudge_counter.threshold == 10

    def test_wm_filled_round_trip(self):
        """WorkingMemory assigned to _current_wm must survive round-trip."""
        session = Session.create(session_id="rt-wm")
        wm = _make_wm(session_id="rt-wm", version=3, at_turn=12)
        session.state._current_wm = wm

        d = _session_to_dict(session)
        recovered = _session_from_dict(d)

        rwm = recovered.state._current_wm
        assert rwm is not None
        assert rwm.session_id == "rt-wm"
        assert rwm.version == 3
        assert rwm.at_turn == 12
        assert rwm.goal == "test goal"
        assert rwm.constraints_and_preferences == ["c01: keep it short"]
        assert rwm.progress == "in progress"
        assert rwm.key_decisions == ["d01: use TDD"]
        assert rwm.relevant_files == ["session.py"]
        assert rwm.next_steps == ["n01: write tests"]
        assert rwm.critical_context == "none"
        assert rwm.updated_by == "framework_init"

    def test_id_gen_state_round_trip(self):
        """id_gen counter state must survive round-trip after 5 next_msg_id() calls."""
        session = Session.create(session_id="rt-idgen")
        for _ in range(5):
            session.state.id_gen.next_msg_id()

        d = _session_to_dict(session)
        recovered = _session_from_dict(d)

        assert recovered.state.id_gen._msg_counter == 5
        # Next call should return m6
        assert recovered.state.id_gen.next_msg_id() == "m6"

    def test_id_gen_all_counters_round_trip(self):
        """All three id_gen counters must survive round-trip."""
        session = Session.create(session_id="rt-idgen-all")
        session.state.id_gen.next_msg_id()
        session.state.id_gen.next_batch_id()
        session.state.id_gen.next_batch_id()
        session.state.id_gen.next_decision_id()
        session.state.id_gen.next_decision_id()
        session.state.id_gen.next_decision_id()

        d = _session_to_dict(session)
        recovered = _session_from_dict(d)

        assert recovered.state.id_gen._msg_counter == 1
        assert recovered.state.id_gen._batch_counter == 2
        assert recovered.state.id_gen._decision_counter == 3

    def test_nudge_counter_count_round_trip(self):
        """nudge_counter._count must survive round-trip after 3 ticks."""
        session = Session.create(session_id="rt-nudge")
        session.state.nudge_counter.tick()
        session.state.nudge_counter.tick()
        session.state.nudge_counter.tick()

        d = _session_to_dict(session)
        recovered = _session_from_dict(d)

        assert recovered.state.nudge_counter._count == 3

    def test_nudge_counter_non_default_threshold_round_trip(self):
        """Non-default nudge_counter threshold must survive round-trip."""
        session = Session.create(session_id="rt-nudge-thresh")
        session.state.nudge_counter = NudgeCounter(threshold=5)
        session.state.nudge_counter.tick()
        session.state.nudge_counter.tick()

        d = _session_to_dict(session)
        recovered = _session_from_dict(d)

        assert recovered.state.nudge_counter.threshold == 5
        assert recovered.state.nudge_counter._count == 2

    def test_dict_is_json_serializable(self):
        """to_dict result must be JSON-serializable (no datetime objects etc.)."""
        import json
        session = Session.create(session_id="json-safe")
        session.state._current_wm = _make_wm()
        d = _session_to_dict(session)
        json.dumps(d)  # must not raise


# ── C. Backward compatibility ─────────────────────────────────────────────────

class TestBackwardCompatibility:
    def _old_session_dict(self) -> dict:
        """Minimal old-format dict without new fields."""
        return {
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
                "promoted_tools": [],
                "freed_tool_results": {},
                "tool_use_to_tool_name": {},
                # No _current_wm, id_gen, nudge_counter
            },
            "messages": [],
        }

    def test_old_session_dict_does_not_raise(self):
        """from_dict with old format (missing new fields) must not raise."""
        d = self._old_session_dict()
        session = _session_from_dict(d)  # must not raise
        assert session.id == "old-session"

    def test_old_session_wm_defaults_to_none(self):
        """_current_wm missing in old dict → None."""
        d = self._old_session_dict()
        session = _session_from_dict(d)
        assert session.state._current_wm is None

    def test_old_session_id_gen_defaults_to_fresh(self):
        """id_gen missing in old dict → fresh SessionIdGenerator (counter=0)."""
        d = self._old_session_dict()
        session = _session_from_dict(d)
        assert isinstance(session.state.id_gen, SessionIdGenerator)
        assert session.state.id_gen._msg_counter == 0
        assert session.state.id_gen._batch_counter == 0
        assert session.state.id_gen._decision_counter == 0

    def test_old_session_nudge_counter_defaults_to_fresh(self):
        """nudge_counter missing in old dict → fresh NudgeCounter (count=0, threshold=10)."""
        d = self._old_session_dict()
        session = _session_from_dict(d)
        assert isinstance(session.state.nudge_counter, NudgeCounter)
        assert session.state.nudge_counter._count == 0
        assert session.state.nudge_counter.threshold == 10

    def test_existing_fields_unaffected(self):
        """Adding new fields must not break deserialization of existing fields."""
        d = self._old_session_dict()
        d["state"]["previous_summary"] = "old summary"
        d["state"]["total_input_tokens"] = 42
        session = _session_from_dict(d)
        assert session.state.previous_summary == "old summary"
        assert session.state.total_input_tokens == 42


# ── D. Full Session save/load round-trip ─────────────────────────────────────

class TestFullSessionRoundTrip:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> JsonFileStorage:
        return JsonFileStorage(tmp_path)

    def test_session_with_wm_save_load(self, storage: JsonFileStorage):
        """Full Session: create, assign WM, save, load → fields match."""
        session = Session.create(session_id="full-rt")
        session.state._current_wm = _make_wm(session_id="full-rt", version=2, at_turn=5)
        session.state.id_gen.next_msg_id()
        session.state.id_gen.next_msg_id()
        session.state.nudge_counter.tick()

        storage.save(session)
        loaded = storage.load("full-rt")

        assert loaded.id == "full-rt"
        rwm = loaded.state._current_wm
        assert rwm is not None
        assert rwm.version == 2
        assert rwm.at_turn == 5
        assert loaded.state.id_gen._msg_counter == 2
        assert loaded.state.nudge_counter._count == 1

    def test_session_without_wm_save_load(self, storage: JsonFileStorage):
        """Full Session with no WM: save/load → _current_wm is None."""
        session = Session.create(session_id="full-no-wm")
        storage.save(session)
        loaded = storage.load("full-no-wm")
        assert loaded.state._current_wm is None


# ── E. WorkingMemory datetime serialization ───────────────────────────────────

class TestWorkingMemoryDatetimeSerialization:
    def test_updated_at_serialized_as_iso8601_string(self):
        """WM.updated_at must be serialized as ISO 8601 string in the dict."""
        dt = datetime(2026, 4, 22, 10, 30, 0)
        session = Session.create(session_id="dt-serial")
        session.state._current_wm = WorkingMemory(
            session_id="dt-serial",
            version=1,
            at_turn=0,
            goal="check datetime",
            constraints_and_preferences=[],
            progress="",
            key_decisions=[],
            relevant_files=[],
            next_steps=[],
            critical_context="",
            updated_by="llm_tool",
            updated_at=dt,
        )
        d = _session_to_dict(session)
        wm_dict = d["state"]["_current_wm"]
        assert isinstance(wm_dict["updated_at"], str)
        assert wm_dict["updated_at"] == dt.isoformat()

    def test_updated_at_deserialized_from_iso8601_string(self):
        """WM.updated_at must be deserialized back to datetime from ISO 8601 string."""
        dt = datetime(2026, 4, 22, 10, 30, 0)
        session = Session.create(session_id="dt-deseial")
        session.state._current_wm = WorkingMemory(
            session_id="dt-deseial",
            version=1,
            at_turn=0,
            goal="check datetime",
            constraints_and_preferences=[],
            progress="",
            key_decisions=[],
            relevant_files=[],
            next_steps=[],
            critical_context="",
            updated_by="llm_tool",
            updated_at=dt,
        )
        d = _session_to_dict(session)
        recovered = _session_from_dict(d)
        assert isinstance(recovered.state._current_wm.updated_at, datetime)
        assert recovered.state._current_wm.updated_at == dt

    def test_updated_at_full_round_trip(self):
        """WM datetime round-trip: original datetime is exactly preserved."""
        dt = datetime(2026, 1, 15, 8, 45, 30, 123456)
        wm = WorkingMemory(
            session_id="s1",
            version=1,
            at_turn=3,
            goal="g",
            constraints_and_preferences=[],
            progress="",
            key_decisions=[],
            relevant_files=[],
            next_steps=[],
            critical_context="",
            updated_by="framework_compression",
            updated_at=dt,
        )
        session = Session.create(session_id="dt-full-rt")
        session.state._current_wm = wm

        d = _session_to_dict(session)
        recovered = _session_from_dict(d)
        assert recovered.state._current_wm.updated_at == dt
