# tests/v2/test_compress_wm_op_list.py
"""Batch B1 — Validator fix: _apply_wm_op op=set on list fields accepts list value.

Tests cover:
  A. op=set with value=list → replaces whole field with that list
  B. op=set with value=str  → single-item list (back-compat)
  C. Other ops (append, remove) unaffected
"""
from __future__ import annotations

import pytest
from datetime import datetime

from neoagent.core.compress import _apply_wm_op
from neoagent.v2.schema import WorkingMemory


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_wm() -> WorkingMemory:
    return WorkingMemory(
        session_id="s1",
        version=1,
        at_turn=0,
        constraints_and_preferences=[],
        progress="",
        key_decisions=[],
        relevant_files=[],
        next_steps=[],
        critical_context="",
        updated_by="framework_init",
        updated_at=datetime(2026, 4, 24, 10, 0, 0),
    )


# ── A. op=set with list value ─────────────────────────────────────────────────

class TestApplyWmOpSetListValue:
    def test_set_constraints_with_list_replaces_whole_field(self):
        """op=set, value=['c01', 'c02'] → field equals ['c01', 'c02']."""
        wm = _make_wm()
        _apply_wm_op(wm, {"field": "constraints_and_preferences", "op": "set", "value": ["c01", "c02"]})
        assert wm.constraints_and_preferences == ["c01", "c02"]

    def test_set_key_decisions_with_list(self):
        """op=set, value list on key_decisions replaces whole field."""
        wm = _make_wm()
        _apply_wm_op(wm, {"field": "key_decisions", "op": "set", "value": ["d01: foo", "d02: bar"]})
        assert wm.key_decisions == ["d01: foo", "d02: bar"]

    def test_set_relevant_files_with_list(self):
        """op=set, value list on relevant_files replaces whole field."""
        wm = _make_wm()
        _apply_wm_op(wm, {"field": "relevant_files", "op": "set", "value": ["a.py", "b.py"]})
        assert wm.relevant_files == ["a.py", "b.py"]

    def test_set_next_steps_with_list(self):
        """op=set, value list on next_steps replaces whole field."""
        wm = _make_wm()
        _apply_wm_op(wm, {"field": "next_steps", "op": "set", "value": ["n01: do x", "n02: do y"]})
        assert wm.next_steps == ["n01: do x", "n02: do y"]

    def test_set_list_value_is_a_copy(self):
        """The list set is a fresh copy — mutating original does not affect wm field."""
        wm = _make_wm()
        items = ["c01: a"]
        _apply_wm_op(wm, {"field": "constraints_and_preferences", "op": "set", "value": items})
        items.append("c02: b")
        assert wm.constraints_and_preferences == ["c01: a"]

    def test_set_empty_list_clears_field(self):
        """op=set, value=[] → field becomes empty list."""
        wm = _make_wm()
        wm.key_decisions = ["d01: existing"]
        _apply_wm_op(wm, {"field": "key_decisions", "op": "set", "value": []})
        assert wm.key_decisions == []


# ── B. op=set with str value (back-compat) ────────────────────────────────────

class TestApplyWmOpSetStrValue:
    def test_set_constraints_with_str_wraps_in_list(self):
        """op=set, value='c01' (str) → field equals ['c01'] (back-compat)."""
        wm = _make_wm()
        _apply_wm_op(wm, {"field": "constraints_and_preferences", "op": "set", "value": "c01"})
        assert wm.constraints_and_preferences == ["c01"]

    def test_set_key_decisions_with_str(self):
        """op=set with str on key_decisions → single-item list."""
        wm = _make_wm()
        _apply_wm_op(wm, {"field": "key_decisions", "op": "set", "value": "d01: decision"})
        assert wm.key_decisions == ["d01: decision"]


# ── C. Other ops unaffected ───────────────────────────────────────────────────

class TestApplyWmOpOtherOps:
    def test_append_still_works(self):
        """op=append is unchanged — appends single value."""
        wm = _make_wm()
        wm.key_decisions = ["d01: first"]
        _apply_wm_op(wm, {"field": "key_decisions", "op": "append", "value": "d02: second"})
        assert wm.key_decisions == ["d01: first", "d02: second"]

    def test_remove_still_works(self):
        """op=remove is unchanged — removes by item_id prefix."""
        wm = _make_wm()
        wm.key_decisions = ["d01: keep", "d02: remove me"]
        _apply_wm_op(wm, {"field": "key_decisions", "op": "remove", "item_id": "d02"})
        assert wm.key_decisions == ["d01: keep"]

    def test_scalar_set_still_works(self):
        """op=set on scalar field (progress) is unchanged."""
        wm = _make_wm()
        _apply_wm_op(wm, {"field": "progress", "op": "set", "value": "50% done"})
        assert wm.progress == "50% done"
