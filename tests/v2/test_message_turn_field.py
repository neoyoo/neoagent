# tests/v2/test_message_turn_field.py
"""Batch B1 — Message.turn field.

Tests cover:
  A. Message accepts turn field and serializes it
  B. Message roundtrip via session save/load preserves turn
"""
from __future__ import annotations

import pytest
from pathlib import Path

from neoagent.core.types import Message
from neoagent.session import (
    Session,
    JsonFileStorage,
    _message_to_dict,
    _message_from_dict,
    _session_to_dict,
    _session_from_dict,
)


# ── A. turn field basics ──────────────────────────────────────────────────────

class TestMessageTurnField:
    def test_message_defaults_turn_to_none(self):
        """Message without turn= → turn is None."""
        msg = Message(role="user", content="hello")
        assert msg.turn is None

    def test_message_accepts_turn_int(self):
        """Message with turn=3 → turn is 3."""
        msg = Message(role="user", content="hello", turn=3)
        assert msg.turn == 3

    def test_message_turn_zero(self):
        """turn=0 is valid (first turn)."""
        msg = Message(role="assistant", content="hi", turn=0)
        assert msg.turn == 0

    def test_message_to_dict_includes_turn_when_set(self):
        """_message_to_dict writes 'turn' key when turn is not None."""
        msg = Message(id="m1", role="user", content="hello", turn=5)
        d = _message_to_dict(msg)
        assert "turn" in d
        assert d["turn"] == 5

    def test_message_to_dict_omits_turn_when_none(self):
        """_message_to_dict omits 'turn' key when turn is None."""
        msg = Message(id="m1", role="user", content="hello")
        d = _message_to_dict(msg)
        assert "turn" not in d

    def test_message_from_dict_reads_turn(self):
        """_message_from_dict restores turn from dict."""
        d = {"role": "user", "content": "hello", "id": "m2", "turn": 7}
        msg = _message_from_dict(d)
        assert msg.turn == 7

    def test_message_from_dict_turn_none_when_absent(self):
        """_message_from_dict → turn=None if key missing (backward compat)."""
        d = {"role": "assistant", "content": "reply"}
        msg = _message_from_dict(d)
        assert msg.turn is None

    def test_message_to_dict_from_dict_roundtrip_with_turn(self):
        """to_dict → from_dict preserves turn exactly."""
        msg = Message(id="m3", role="assistant", content="reply", turn=2)
        recovered = _message_from_dict(_message_to_dict(msg))
        assert recovered.turn == 2
        assert recovered.role == "assistant"

    def test_message_to_dict_from_dict_roundtrip_without_turn(self):
        """to_dict → from_dict preserves turn=None (backward compat)."""
        msg = Message(id="m4", role="user", content="q")
        recovered = _message_from_dict(_message_to_dict(msg))
        assert recovered.turn is None


# ── B. Session save/load roundtrip ───────────────────────────────────────────

class TestMessageTurnSessionRoundtrip:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> JsonFileStorage:
        return JsonFileStorage(tmp_path)

    def test_session_roundtrip_preserves_turn(self, storage: JsonFileStorage):
        """Messages with turn= survive session save/load."""
        session = Session.create(session_id="turn-rt")
        session.messages.append(Message(id="m1", role="user", content="hello", turn=0))
        session.messages.append(Message(id="m2", role="assistant", content="hi", turn=0))

        storage.save(session)
        loaded = storage.load("turn-rt")

        assert loaded.messages[0].turn == 0
        assert loaded.messages[1].turn == 0

    def test_session_roundtrip_mixed_turn_none(self, storage: JsonFileStorage):
        """Messages without turn= (None) survive alongside messages with turn."""
        session = Session.create(session_id="turn-mixed")
        session.messages.append(Message(role="user", content="old msg"))   # no turn
        session.messages.append(Message(id="m1", role="assistant", content="new", turn=3))

        storage.save(session)
        loaded = storage.load("turn-mixed")

        assert loaded.messages[0].turn is None
        assert loaded.messages[1].turn == 3

    def test_session_to_dict_from_dict_roundtrip_turn(self):
        """to_dict / from_dict preserves turn on messages."""
        session = Session.create(session_id="turn-dict-rt")
        session.messages.append(Message(id="m1", role="user", content="q", turn=1))
        d = _session_to_dict(session)
        recovered = _session_from_dict(d)
        assert recovered.messages[0].turn == 1
