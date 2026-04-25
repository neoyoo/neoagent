# tests/v2/test_session_state_fields.py
"""Batch B1 — SessionState new fields: compressed_messages, batches, turns_since_last_compression.

Tests cover:
  A. Field defaults
  B. Roundtrip serialization (to_dict / from_dict)
"""
from __future__ import annotations

import dataclasses
import pytest
from datetime import datetime
from pathlib import Path

from neoagent.core.types import Message
from neoagent.session import (
    Session,
    SessionState,
    JsonFileStorage,
    _session_to_dict,
    _session_from_dict,
)
from neoagent.v2.schema import Batch, BatchMember


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_batch(session_id: str = "s1", batch_id: str = "cm_1") -> Batch:
    now = datetime(2026, 4, 24, 10, 0, 0)
    return Batch(
        session_id=session_id,
        batch_id=batch_id,
        turns_from=0,
        turns_to=2,
        time_from=now,
        time_to=now,
        summary="PROGRESS: done\nDECISIONS:\nFILES:\nNEXT STEPS:\nKEY CONTEXT:",
        members=[BatchMember(id="m1", role="user", preview="hello")],
        trigger="token_threshold",
        created_at=now,
    )


# ── A. Field defaults ─────────────────────────────────────────────────────────

class TestSessionStateNewFieldDefaults:
    def test_compressed_messages_default_empty_list(self):
        """SessionState.compressed_messages must default to []."""
        state = SessionState()
        assert state.compressed_messages == []
        assert isinstance(state.compressed_messages, list)

    def test_batches_default_empty_list(self):
        """SessionState.batches must default to []."""
        state = SessionState()
        assert state.batches == []
        assert isinstance(state.batches, list)

    def test_turns_since_last_compression_default_zero(self):
        """SessionState.turns_since_last_compression must default to 0."""
        state = SessionState()
        assert state.turns_since_last_compression == 0

    def test_compressed_messages_independent_per_instance(self):
        """Each SessionState must have its own compressed_messages list."""
        s1 = SessionState()
        s2 = SessionState()
        s1.compressed_messages.append(Message(role="user", content="x"))
        assert s2.compressed_messages == []

    def test_batches_independent_per_instance(self):
        """Each SessionState must have its own batches list."""
        s1 = SessionState()
        s2 = SessionState()
        s1.batches.append(_make_batch())
        assert s2.batches == []


# ── B. Roundtrip serialization ────────────────────────────────────────────────

class TestSessionStateRoundtrip:
    def test_compressed_messages_roundtrip(self):
        """compressed_messages survive to_dict / from_dict."""
        session = Session.create(session_id="cm-rt")
        session.state.compressed_messages.append(
            Message(id="m1", role="user", content="compressed turn", turn=0)
        )
        d = _session_to_dict(session)
        recovered = _session_from_dict(d)

        assert len(recovered.state.compressed_messages) == 1
        msg = recovered.state.compressed_messages[0]
        assert msg.id == "m1"
        assert msg.role == "user"
        assert msg.content == "compressed turn"
        assert msg.turn == 0

    def test_batches_roundtrip(self):
        """batches survive to_dict / from_dict."""
        session = Session.create(session_id="batch-rt")
        batch = _make_batch(session_id="batch-rt", batch_id="cm_1")
        session.state.batches.append(batch)

        d = _session_to_dict(session)
        recovered = _session_from_dict(d)

        assert len(recovered.state.batches) == 1
        rb = recovered.state.batches[0]
        assert rb.batch_id == "cm_1"
        assert rb.session_id == "batch-rt"
        assert rb.turns_from == 0
        assert rb.turns_to == 2
        assert rb.summary.startswith("PROGRESS:")
        assert len(rb.members) == 1
        assert rb.members[0].id == "m1"

    def test_turns_since_last_compression_roundtrip(self):
        """turns_since_last_compression survives to_dict / from_dict."""
        session = Session.create(session_id="turns-rt")
        session.state.turns_since_last_compression = 7

        d = _session_to_dict(session)
        recovered = _session_from_dict(d)

        assert recovered.state.turns_since_last_compression == 7

    def test_empty_new_fields_roundtrip(self):
        """Empty compressed_messages / batches / zero turns survive round-trip."""
        session = Session.create(session_id="empty-rt")
        d = _session_to_dict(session)
        recovered = _session_from_dict(d)

        assert recovered.state.compressed_messages == []
        assert recovered.state.batches == []
        assert recovered.state.turns_since_last_compression == 0

    def test_dict_is_json_serializable_with_new_fields(self):
        """to_dict result with new fields must be JSON-serializable."""
        import json
        session = Session.create(session_id="json-safe-b1")
        session.state.compressed_messages.append(Message(role="user", content="old", turn=0))
        session.state.batches.append(_make_batch())
        session.state.turns_since_last_compression = 3
        d = _session_to_dict(session)
        json.dumps(d)  # must not raise

    def test_full_save_load_roundtrip(self, tmp_path: Path):
        """Full save/load via JsonFileStorage preserves all three new fields."""
        storage = JsonFileStorage(tmp_path)
        session = Session.create(session_id="full-b1-rt")
        session.state.compressed_messages.append(
            Message(id="m1", role="user", content="archived", turn=1)
        )
        session.state.batches.append(_make_batch(session_id="full-b1-rt"))
        session.state.turns_since_last_compression = 4

        storage.save(session)
        loaded = storage.load("full-b1-rt")

        assert len(loaded.state.compressed_messages) == 1
        assert loaded.state.compressed_messages[0].content == "archived"
        assert len(loaded.state.batches) == 1
        assert loaded.state.batches[0].batch_id == "cm_1"
        assert loaded.state.turns_since_last_compression == 4

    def test_backward_compat_old_dict_missing_new_fields(self):
        """Old session dict without new B1 fields deserializes to defaults."""
        old_dict = {
            "id": "old-b1",
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
                # No compressed_messages, batches, turns_since_last_compression
            },
            "messages": [],
        }
        session = _session_from_dict(old_dict)
        assert session.state.compressed_messages == []
        assert session.state.batches == []
        assert session.state.turns_since_last_compression == 0
