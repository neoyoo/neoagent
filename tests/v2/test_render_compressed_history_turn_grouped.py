# tests/v2/test_render_compressed_history_turn_grouped.py
"""Turn-grouped <recoverable> rendering. Bug #2: <summary> removed.

Contract:
  - Batch with members across turns 0, 1, 2 → 3 <turn> elements in order
  - Batch with all members in turn 0 → single <turn n="0">
  - Members with turn=None → <turn n="unknown">
  - role= attribute preserved
  - <summary> tag is NEVER emitted (WM is the single source of state)
"""
from __future__ import annotations

from datetime import datetime

import pytest

from neoagent.core.prompt import LayeredPromptBuilder
from neoagent.v2.schema import Batch, BatchMember


def _make_builder() -> LayeredPromptBuilder:
    return LayeredPromptBuilder()


def _make_batch(
    members: list[BatchMember],
    summary: str | dict = "test summary",
    batch_id: str = "cm_1",
    turns_from: int = 0,
    turns_to: int = 2,
) -> Batch:
    now = datetime(2026, 4, 24, 12, 0, 0)
    return Batch(
        session_id="s1",
        batch_id=batch_id,
        turns_from=turns_from,
        turns_to=turns_to,
        time_from=now,
        time_to=now,
        summary=summary,
        members=members,
        trigger="token_threshold",
        created_at=now,
    )


# ── Turn-grouped <recoverable> ────────────────────────────────────────────────

class TestTurnGroupedRecoverable:

    def test_three_turns_produce_three_turn_elements(self):
        """Members across turns 0, 1, 2 → 3 <turn> elements."""
        builder = _make_builder()
        members = [
            BatchMember(id="m1", role="user", preview="p1", turn=0),
            BatchMember(id="m2", role="assistant", preview="p2", turn=0),
            BatchMember(id="m3", role="user", preview="p3", turn=1),
            BatchMember(id="m4", role="assistant", preview="p4", turn=1),
            BatchMember(id="m5", role="user", preview="p5", turn=2),
        ]
        batch = _make_batch(members)
        output = builder._render_compressed_history([batch])

        assert '<turn n="0">' in output
        assert '<turn n="1">' in output
        assert '<turn n="2">' in output

    def test_three_turns_in_ascending_order(self):
        """Turn 0 before Turn 1 before Turn 2 in output."""
        builder = _make_builder()
        members = [
            BatchMember(id="m1", role="user", preview="p1", turn=0),
            BatchMember(id="m3", role="user", preview="p3", turn=1),
            BatchMember(id="m5", role="user", preview="p5", turn=2),
        ]
        batch = _make_batch(members)
        output = builder._render_compressed_history([batch])

        pos0 = output.index('<turn n="0">')
        pos1 = output.index('<turn n="1">')
        pos2 = output.index('<turn n="2">')
        assert pos0 < pos1 < pos2

    def test_single_turn_wraps_all_members(self):
        """All members in turn 0 → single <turn n="0"> wrapping all."""
        builder = _make_builder()
        members = [
            BatchMember(id="m1", role="user", preview="p1", turn=0),
            BatchMember(id="m2", role="assistant", preview="p2", turn=0),
            BatchMember(id="m3", role="user", preview="p3", turn=0),
        ]
        batch = _make_batch(members)
        output = builder._render_compressed_history([batch])

        assert '<turn n="0">' in output
        assert '<turn n="1">' not in output
        # All three msgs inside the single turn block
        pos_turn0 = output.index('<turn n="0">')
        pos_close_turn = output.index("</turn>", pos_turn0)
        turn_block = output[pos_turn0:pos_close_turn]
        assert 'id="m1"' in turn_block
        assert 'id="m2"' in turn_block
        assert 'id="m3"' in turn_block

    def test_members_within_turn_preserve_order(self):
        """m1 before m2 within same turn in rendered output."""
        builder = _make_builder()
        members = [
            BatchMember(id="m1", role="user", preview="first", turn=0),
            BatchMember(id="m2", role="assistant", preview="second", turn=0),
        ]
        batch = _make_batch(members)
        output = builder._render_compressed_history([batch])

        pos_m1 = output.index('id="m1"')
        pos_m2 = output.index('id="m2"')
        assert pos_m1 < pos_m2

    def test_role_attribute_preserved_in_msg_tag(self):
        """role= attribute is rendered correctly on <msg> tags."""
        builder = _make_builder()
        members = [
            BatchMember(id="m1", role="user", preview="p", turn=0),
            BatchMember(id="m2", role="assistant", preview="q", turn=0),
        ]
        batch = _make_batch(members)
        output = builder._render_compressed_history([batch])

        assert 'role="user"' in output
        assert 'role="assistant"' in output

    def test_preview_attribute_preserved(self):
        """preview= attribute is rendered on <msg> tags."""
        builder = _make_builder()
        members = [
            BatchMember(id="m1", role="user", preview="my preview text", turn=0),
        ]
        batch = _make_batch(members)
        output = builder._render_compressed_history([batch])

        assert 'preview="my preview text"' in output


# ── turn=None → <turn n="unknown"> ───────────────────────────────────────────

class TestTurnNoneGroup:

    def test_none_turn_members_go_to_unknown_group(self):
        """Members with turn=None render under <turn n="unknown">."""
        builder = _make_builder()
        members = [
            BatchMember(id="m1", role="user", preview="legacy", turn=None),
        ]
        batch = _make_batch(members)
        output = builder._render_compressed_history([batch])

        assert '<turn n="unknown">' in output

    def test_none_turn_group_before_int_turns(self):
        """Unknown group appears before int-turn groups."""
        builder = _make_builder()
        members = [
            BatchMember(id="m1", role="user", preview="legacy", turn=None),
            BatchMember(id="m2", role="user", preview="normal", turn=0),
        ]
        batch = _make_batch(members)
        output = builder._render_compressed_history([batch])

        pos_unknown = output.index('<turn n="unknown">')
        pos_turn0 = output.index('<turn n="0">')
        assert pos_unknown < pos_turn0

    def test_none_turn_message_inside_unknown_block(self):
        """m1 with turn=None appears inside the unknown block."""
        builder = _make_builder()
        members = [
            BatchMember(id="m1", role="user", preview="legacy", turn=None),
            BatchMember(id="m2", role="user", preview="normal", turn=0),
        ]
        batch = _make_batch(members)
        output = builder._render_compressed_history([batch])

        pos_unknown = output.index('<turn n="unknown">')
        pos_turn0 = output.index('<turn n="0">')
        unknown_block = output[pos_unknown:pos_turn0]
        assert 'id="m1"' in unknown_block


# ── Bug #2: <summary> tag is never emitted ───────────────────────────────────

class TestNoSummaryEmitted:
    """Bug #2 fix: <summary> removed from rendered output entirely.

    WM is the single source of state; compressed_history is a pure recall index.
    """

    def test_summary_absent_when_batch_summary_is_none(self):
        """Batch.summary=None → no <summary> tag emitted."""
        builder = _make_builder()
        batch = _make_batch(members=[], summary=None)
        output = builder._render_compressed_history([batch])

        assert "<summary>" not in output
        assert "</summary>" not in output

    def test_summary_absent_when_batch_summary_is_dict(self):
        """Even if batch.summary is a legacy dict, no <summary> is rendered."""
        builder = _make_builder()
        summary = {
            "CONSTRAINTS_AND_PREFERENCES": ["c01: use JWT"],
            "PROGRESS": "login done",
            "KEY_DECISIONS": ["d01: use FastAPI"],
            "RELEVANT_FILES": ["f01: auth.py"],
            "NEXT_STEPS": ["n01: write tests"],
            "CRITICAL_CONTEXT": "HTTPS in prod",
        }
        batch = _make_batch(members=[], summary=summary)
        output = builder._render_compressed_history([batch])

        assert "<summary>" not in output
        assert "</summary>" not in output

    def test_summary_absent_when_batch_summary_is_str(self):
        """Even if batch.summary is a legacy str, no <summary> is rendered."""
        builder = _make_builder()
        batch = _make_batch(members=[], summary="PROGRESS: done\nDECISIONS: none")
        output = builder._render_compressed_history([batch])

        assert "<summary>" not in output
        assert "</summary>" not in output

    def test_no_wm_duplicate_fields_in_output(self):
        """No constraints_and_preferences / progress / key_decisions XML in history output."""
        builder = _make_builder()
        batch = _make_batch(members=[], summary=None)
        output = builder._render_compressed_history([batch])

        assert "<constraints_and_preferences>" not in output
        assert "<progress>" not in output
        assert "<key_decisions>" not in output


# ── Batch XML structure ───────────────────────────────────────────────────────

class TestBatchXMLStructure:

    def test_batch_has_id_and_turns_attributes(self):
        """<batch> tag has id= and turns= attributes."""
        builder = _make_builder()
        batch = _make_batch(members=[], batch_id="cm_42", turns_from=2, turns_to=5)
        output = builder._render_compressed_history([batch])

        assert 'id="cm_42"' in output
        assert 'turns="2-5"' in output

    def test_recoverable_tag_absent_when_no_members(self):
        """No <recoverable> block when batch has no members."""
        builder = _make_builder()
        batch = _make_batch(members=[])
        output = builder._render_compressed_history([batch])

        assert "<recoverable>" not in output

    def test_compressed_history_wrapper(self):
        """Output is wrapped in <compressed_history>."""
        builder = _make_builder()
        output = builder._render_compressed_history([_make_batch(members=[])])

        assert output.startswith("<compressed_history>")
        assert output.endswith("</compressed_history>")

    def test_multiple_batches_all_present(self):
        """Multiple batches all appear in output."""
        builder = _make_builder()
        batch1 = _make_batch(members=[], batch_id="cm_1")
        batch2 = _make_batch(members=[], batch_id="cm_2")
        output = builder._render_compressed_history([batch1, batch2])

        assert 'id="cm_1"' in output
        assert 'id="cm_2"' in output
