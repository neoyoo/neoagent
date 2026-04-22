"""Tests for neoagent/v2/schema.py — strict TDD per Shared Contracts C1."""
from __future__ import annotations
import pytest
from datetime import datetime


# ── WorkingMemory ─────────────────────────────────────────────────────────────

class TestWorkingMemory:
    def _make(self, **overrides):
        from neoagent.v2.schema import WorkingMemory
        defaults = dict(
            session_id="s1",
            version=1,
            at_turn=0,
            goal="write tests",
            constraints_and_preferences=[],
            progress="",
            key_decisions=[],
            relevant_files=[],
            next_steps=[],
            critical_context="",
            updated_by="framework_init",
            updated_at=datetime(2026, 4, 22),
        )
        defaults.update(overrides)
        return WorkingMemory(**defaults)

    def test_construct_basic(self):
        wm = self._make()
        assert wm.session_id == "s1"
        assert wm.version == 1
        assert wm.at_turn == 0
        assert wm.goal == "write tests"

    def test_list_fields_default_to_empty_list(self):
        from neoagent.v2.schema import WorkingMemory
        # WorkingMemory requires explicit args — lists can be passed empty
        wm = self._make()
        assert wm.constraints_and_preferences == []
        assert wm.key_decisions == []
        assert wm.relevant_files == []
        assert wm.next_steps == []

    def test_list_fields_are_independent_instances(self):
        from neoagent.v2.schema import WorkingMemory
        wm1 = self._make()
        wm2 = self._make()
        wm1.constraints_and_preferences.append("c01: x")
        assert wm2.constraints_and_preferences == []

    def test_updated_by_literal_values(self):
        for val in ("llm_tool", "framework_init", "framework_compression"):
            wm = self._make(updated_by=val)
            assert wm.updated_by == val

    def test_updated_at_is_datetime(self):
        wm = self._make()
        assert isinstance(wm.updated_at, datetime)

    def test_goal_field_exists(self):
        wm = self._make(goal="explore Mars")
        assert wm.goal == "explore Mars"


# ── BatchMember ───────────────────────────────────────────────────────────────

class TestBatchMember:
    def test_construct(self):
        from neoagent.v2.schema import BatchMember
        bm = BatchMember(id="m5", role="user", preview="short preview text here")
        assert bm.id == "m5"
        assert bm.role == "user"
        assert bm.preview == "short preview text here"

    def test_role_assistant(self):
        from neoagent.v2.schema import BatchMember
        bm = BatchMember(id="t7", role="assistant", preview="assistant says hi")
        assert bm.role == "assistant"

    def test_role_tool(self):
        from neoagent.v2.schema import BatchMember
        bm = BatchMember(id="t3", role="tool", preview="tool returned data")
        assert bm.role == "tool"


# ── CompressionDelta ──────────────────────────────────────────────────────────

class TestCompressionDelta:
    def test_construct_all_required_fields(self):
        from neoagent.v2.schema import CompressionDelta, BatchMember
        cd = CompressionDelta(
            batch_summary="7-section structured text",
            batch_members=[BatchMember(id="m1", role="user", preview="user said hi")],
            working_memory_delta=[{"field": "progress", "op": "set", "value": "done"}],
        )
        assert cd.batch_summary == "7-section structured text"
        assert len(cd.batch_members) == 1
        assert cd.batch_members[0].id == "m1"
        assert len(cd.working_memory_delta) == 1

    def test_batch_members_required_and_independent(self):
        from neoagent.v2.schema import CompressionDelta
        cd1 = CompressionDelta(batch_summary="s", batch_members=[], working_memory_delta=[])
        cd2 = CompressionDelta(batch_summary="s", batch_members=[], working_memory_delta=[])
        cd1.batch_members.append("x")  # type: ignore
        assert cd2.batch_members == []

    def test_working_memory_delta_required_and_independent(self):
        from neoagent.v2.schema import CompressionDelta
        cd1 = CompressionDelta(batch_summary="s", batch_members=[], working_memory_delta=[])
        cd2 = CompressionDelta(batch_summary="s", batch_members=[], working_memory_delta=[])
        cd1.working_memory_delta.append({"field": "x", "op": "set", "value": "y"})
        assert cd2.working_memory_delta == []


# ── CompressionContext ────────────────────────────────────────────────────────

class TestCompressionContext:
    def test_construct(self):
        from neoagent.v2.schema import CompressionContext
        ctx = CompressionContext(
            session_id="s1",
            messages=[{"role": "user", "content": "hello"}],
            previous_batches=[],
            previous_wm={"goal": "explore"},
            trigger="token_threshold",
        )
        assert ctx.session_id == "s1"
        assert ctx.trigger == "token_threshold"
        assert ctx.previous_batches == []

    def test_trigger_message_count(self):
        from neoagent.v2.schema import CompressionContext
        ctx = CompressionContext(
            session_id="s2",
            messages=[],
            previous_batches=[],
            previous_wm={},
            trigger="message_count",
        )
        assert ctx.trigger == "message_count"

    def test_trigger_window_ratio(self):
        from neoagent.v2.schema import CompressionContext
        ctx = CompressionContext(
            session_id="s3",
            messages=[],
            previous_batches=[],
            previous_wm={},
            trigger="window_ratio",
        )
        assert ctx.trigger == "window_ratio"


# ── Batch ─────────────────────────────────────────────────────────────────────

class TestBatch:
    def test_construct(self):
        from neoagent.v2.schema import Batch, BatchMember
        now = datetime(2026, 4, 22, 12, 0, 0)
        batch = Batch(
            session_id="s1",
            batch_id="cm_1",
            turns_from=0,
            turns_to=5,
            time_from=now,
            time_to=now,
            summary="batch summary",
            members=[BatchMember(id="m1", role="user", preview="hello")],
            trigger="token_threshold",
            created_at=now,
        )
        assert batch.session_id == "s1"
        assert batch.batch_id == "cm_1"
        assert batch.turns_from == 0
        assert batch.turns_to == 5
        assert len(batch.members) == 1

    def test_members_required_and_independent(self):
        from neoagent.v2.schema import Batch
        now = datetime(2026, 4, 22)
        b1 = Batch(session_id="s1", batch_id="cm_1", turns_from=0, turns_to=1,
                   time_from=now, time_to=now, summary="s", members=[],
                   trigger="message_count", created_at=now)
        b2 = Batch(session_id="s1", batch_id="cm_2", turns_from=0, turns_to=1,
                   time_from=now, time_to=now, summary="s", members=[],
                   trigger="message_count", created_at=now)
        b1.members.append("x")  # type: ignore
        assert b2.members == []


# ── MemoryEntry ───────────────────────────────────────────────────────────────

class TestMemoryEntry:
    def test_construct_minimal(self):
        from neoagent.v2.schema import MemoryEntry
        entry = MemoryEntry(
            user_id="u1",
            memory_id="mem_1",
            type="preference",
            category=None,
            content="user prefers short answers",
        )
        assert entry.user_id == "u1"
        assert entry.memory_id == "mem_1"
        assert entry.type == "preference"
        assert entry.category is None
        assert entry.content == "user prefers short answers"

    def test_default_confidence(self):
        from neoagent.v2.schema import MemoryEntry
        entry = MemoryEntry(user_id="u1", memory_id="m1", type="fact", category=None, content="x")
        assert entry.confidence == 0.5

    def test_default_usage_count(self):
        from neoagent.v2.schema import MemoryEntry
        entry = MemoryEntry(user_id="u1", memory_id="m1", type="fact", category=None, content="x")
        assert entry.usage_count == 0

    def test_default_evidence_is_none(self):
        from neoagent.v2.schema import MemoryEntry
        entry = MemoryEntry(user_id="u1", memory_id="m1", type="fact", category=None, content="x")
        assert entry.evidence is None

    def test_default_last_reinforced_is_none(self):
        from neoagent.v2.schema import MemoryEntry
        entry = MemoryEntry(user_id="u1", memory_id="m1", type="fact", category=None, content="x")
        assert entry.last_reinforced_at is None

    def test_created_at_defaults_to_now(self):
        from neoagent.v2.schema import MemoryEntry
        entry = MemoryEntry(user_id="u1", memory_id="m1", type="fact", category=None, content="x")
        assert isinstance(entry.created_at, datetime)

    def test_created_at_independent_instances(self):
        from neoagent.v2.schema import MemoryEntry
        e1 = MemoryEntry(user_id="u1", memory_id="m1", type="fact", category=None, content="x")
        e2 = MemoryEntry(user_id="u1", memory_id="m2", type="fact", category=None, content="y")
        # both should have datetime instances (may be same or different, both valid)
        assert isinstance(e1.created_at, datetime)
        assert isinstance(e2.created_at, datetime)

    def test_can_set_category(self):
        from neoagent.v2.schema import MemoryEntry
        entry = MemoryEntry(user_id="u1", memory_id="m1", type="goal", category="work", content="finish docs")
        assert entry.category == "work"

    def test_can_override_confidence(self):
        from neoagent.v2.schema import MemoryEntry
        entry = MemoryEntry(user_id="u1", memory_id="m1", type="fact", category=None,
                            content="x", confidence=0.9)
        assert entry.confidence == 0.9


# ── Layer Enum ────────────────────────────────────────────────────────────────

class TestLayerEnum:
    def test_layer_has_7_values(self):
        from neoagent.v2.schema import Layer
        assert len(Layer) == 7

    def test_layer_values(self):
        from neoagent.v2.schema import Layer
        assert Layer.IDENTITY == "identity"
        assert Layer.PERSISTENT_MEMORY == "persistent_memory"
        assert Layer.CAPABILITIES == "capabilities"
        assert Layer.SECURITY == "security"
        assert Layer.WORKING_MEMORY == "working_memory"
        assert Layer.COMPRESSED_HISTORY == "compressed_history"
        assert Layer.MEMORY_CONTEXT == "memory_context"

    def test_layer_is_str_enum(self):
        from neoagent.v2.schema import Layer
        assert isinstance(Layer.IDENTITY, str)

    def test_layer_iteration_order(self):
        from neoagent.v2.schema import Layer
        values = [l.value for l in Layer]
        assert values == [
            "identity",
            "persistent_memory",
            "capabilities",
            "security",
            "working_memory",
            "compressed_history",
            "memory_context",
        ]

    def test_layer_members_exist(self):
        from neoagent.v2.schema import Layer
        # Enumerate all 7 members by name
        names = {l.name for l in Layer}
        assert names == {
            "IDENTITY", "PERSISTENT_MEMORY", "CAPABILITIES",
            "SECURITY", "WORKING_MEMORY", "COMPRESSED_HISTORY", "MEMORY_CONTEXT"
        }
