"""Tests for LayeredPromptBuilder (Phase 4, Task 4.1).

Spec refs: § 7.1, § 7.2, § 7.4, § 15.1
"""
from __future__ import annotations

import warnings
from datetime import datetime

import pytest

from neoagent.core.prompt import LayeredPromptBuilder, PromptSection
from neoagent.v2.schema import (
    Batch,
    BatchMember,
    Layer,
    MemoryEntry,
    WorkingMemory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(name: str, content: str, priority: int = 5) -> PromptSection:
    return PromptSection(name=name, content=content, priority=priority)


def _wm(
    *,
    version: int = 3,
    at_turn: int = 12,
    constraints: list[str] | None = None,
    progress: str = "50% done",
    key_decisions: list[str] | None = None,
    relevant_files: list[str] | None = None,
    next_steps: list[str] | None = None,
    critical_context: str = "Important context",
) -> WorkingMemory:
    return WorkingMemory(
        session_id="s1",
        version=version,
        at_turn=at_turn,
        constraints_and_preferences=constraints or [],
        progress=progress,
        key_decisions=key_decisions or [],
        relevant_files=relevant_files or [],
        next_steps=next_steps or [],
        critical_context=critical_context,
        updated_by="framework_init",
        updated_at=datetime(2026, 1, 1),
    )


def _batch(
    batch_id: str = "cm_1",
    turns_from: int = 1,
    turns_to: int = 12,
    summary: str = "GOAL: test\nPROGRESS: done",
    members: list[BatchMember] | None = None,
) -> Batch:
    return Batch(
        session_id="s1",
        batch_id=batch_id,
        turns_from=turns_from,
        turns_to=turns_to,
        time_from=datetime(2026, 1, 1),
        time_to=datetime(2026, 1, 2),
        summary=summary,
        members=members or [],
        trigger="token_threshold",
        created_at=datetime(2026, 1, 2),
    )


def _entry(type_: str, content: str, confidence: float = 0.8) -> MemoryEntry:
    return MemoryEntry(
        user_id="u1",
        memory_id="m1",
        type=type_,
        category=None,
        content=content,
        confidence=confidence,
    )


# ===========================================================================
# A. register / remove / build_system_prompt
# ===========================================================================


class TestRegisterRemoveBuild:
    def test_register_single_layer(self):
        """A1: register 1 IDENTITY section → build_system_prompt contains content."""
        builder = LayeredPromptBuilder()
        builder.register_layer_section(Layer.IDENTITY, _section("id1", "You are Neo."))
        result = builder.build_system_prompt()
        assert "You are Neo." in result

    def test_register_multi_layer_order(self):
        """A2: IDENTITY output precedes CAPABILITIES precedes SECURITY."""
        builder = LayeredPromptBuilder()
        builder.register_layer_section(Layer.SECURITY, _section("sec", "Security rules."))
        builder.register_layer_section(Layer.CAPABILITIES, _section("cap", "Tool capabilities."))
        builder.register_layer_section(Layer.IDENTITY, _section("id1", "Identity content."))
        result = builder.build_system_prompt()
        pos_id = result.index("Identity content.")
        pos_cap = result.index("Tool capabilities.")
        pos_sec = result.index("Security rules.")
        assert pos_id < pos_cap < pos_sec

    def test_same_layer_priority_descending(self):
        """A3: 2 IDENTITY sections with priority 1 and 10 → priority=10 rendered first."""
        builder = LayeredPromptBuilder()
        builder.register_layer_section(Layer.IDENTITY, _section("low", "LOW", priority=1))
        builder.register_layer_section(Layer.IDENTITY, _section("high", "HIGH", priority=10))
        result = builder.build_system_prompt()
        assert result.index("HIGH") < result.index("LOW")

    def test_remove_layer_section(self):
        """A4: remove section → not in build."""
        builder = LayeredPromptBuilder()
        builder.register_layer_section(Layer.IDENTITY, _section("id1", "Remove me."))
        builder.remove_layer_section(Layer.IDENTITY, "id1")
        result = builder.build_system_prompt()
        assert "Remove me." not in result

    def test_remove_not_found_raises_key_error(self):
        """A5: remove nonexistent name → KeyError."""
        builder = LayeredPromptBuilder()
        with pytest.raises(KeyError):
            builder.remove_layer_section(Layer.IDENTITY, "ghost")

    def test_build_system_prompt_excludes_ephemeral_layers(self):
        """A6: WORKING_MEMORY section registered → build_system_prompt excludes it."""
        builder = LayeredPromptBuilder()
        builder.register_layer_section(
            Layer.WORKING_MEMORY, _section("wm", "WM ephemeral content.")
        )
        result = builder.build_system_prompt()
        assert "WM ephemeral content." not in result


# ===========================================================================
# B. skill activation
# ===========================================================================


class TestSkillActivation:
    def test_register_skill_inactive_by_default(self):
        """B7: registered skill not visible in build_system_prompt."""
        builder = LayeredPromptBuilder()
        builder.register_skill("math", _section("math", "Math skill content."))
        result = builder.build_system_prompt()
        assert "Math skill content." not in result

    def test_activate_skill_visible(self):
        """B8: activate_skill → section appears in build_system_prompt."""
        builder = LayeredPromptBuilder()
        builder.register_skill("math", _section("math", "Math skill content."))
        builder.activate_skill("math")
        result = builder.build_system_prompt()
        assert "Math skill content." in result

    def test_deactivate_skill_disappears(self):
        """B9: deactivate_skill → section no longer in build_system_prompt."""
        builder = LayeredPromptBuilder()
        builder.register_skill("math", _section("math", "Math skill content."))
        builder.activate_skill("math")
        builder.deactivate_skill("math")
        result = builder.build_system_prompt()
        assert "Math skill content." not in result

    def test_is_skill_active(self):
        """B10: is_skill_active reflects correct state."""
        builder = LayeredPromptBuilder()
        builder.register_skill("geo", _section("geo", "Geo content."))
        assert not builder.is_skill_active("geo")
        builder.activate_skill("geo")
        assert builder.is_skill_active("geo")
        builder.deactivate_skill("geo")
        assert not builder.is_skill_active("geo")

    def test_skill_routed_to_capabilities_layer(self):
        """B11: skill section appears in CAPABILITIES order position."""
        builder = LayeredPromptBuilder()
        # Register in IDENTITY and SECURITY
        builder.register_layer_section(Layer.IDENTITY, _section("id1", "IDENTITY_CONTENT"))
        builder.register_layer_section(Layer.SECURITY, _section("sec", "SECURITY_CONTENT"))
        # Register and activate a skill (should route to CAPABILITIES)
        builder.register_skill("sk", _section("sk", "SKILL_CONTENT"))
        builder.activate_skill("sk")
        result = builder.build_system_prompt()
        # IDENTITY < CAPABILITIES(skill) < SECURITY
        assert result.index("IDENTITY_CONTENT") < result.index("SKILL_CONTENT")
        assert result.index("SKILL_CONTENT") < result.index("SECURITY_CONTENT")


# ===========================================================================
# C. build_ephemeral
# ===========================================================================


class TestBuildEphemeral:
    def test_all_none_returns_empty(self):
        """C12: wm=None, batches=[], memory_entries=None → empty / no XML blocks."""
        builder = LayeredPromptBuilder()
        result = builder.build_ephemeral(wm=None, batches=[], memory_entries=None)
        assert "<working_memory" not in result
        assert "<compressed_history" not in result
        assert "<memory-context" not in result
        assert result.strip() == ""

    def test_wm_only(self):
        """C13: wm rendered with version and at_turn attributes, 7 sections."""
        builder = LayeredPromptBuilder()
        wm = _wm(version=3, at_turn=12)
        result = builder.build_ephemeral(wm=wm, batches=[], memory_entries=None)
        assert '<working_memory version="3" at_turn="12">' in result
        assert "CONSTRAINTS_AND_PREFERENCES:" in result
        assert "PROGRESS:" in result
        assert "KEY_DECISIONS:" in result
        assert "RELEVANT_FILES:" in result
        assert "NEXT_STEPS:" in result
        assert "CRITICAL_CONTEXT:" in result
        assert "</working_memory>" in result

    def test_wm_list_prefixes_preserved(self):
        """C14: key_decisions with d01/d02 prefixes → preserved in output."""
        builder = LayeredPromptBuilder()
        wm = _wm(
            key_decisions=["d01: 酒店", "d02: 航班"],
            constraints=["c01: 预算 5000"],
        )
        result = builder.build_ephemeral(wm=wm, batches=[])
        assert "d01: 酒店" in result
        assert "d02: 航班" in result
        assert "c01: 预算 5000" in result

    def test_batches_only(self):
        """C15: 1 Batch with 2 BatchMembers → compressed_history with 1 batch and 2 msg elements."""
        builder = LayeredPromptBuilder()
        members = [
            BatchMember(id="m01", role="user", preview="Hello"),
            BatchMember(id="m02", role="assistant", preview="Hi there"),
        ]
        batch = _batch(batch_id="cm_1", turns_from=1, turns_to=12, members=members)
        result = builder.build_ephemeral(wm=None, batches=[batch])
        assert "<compressed_history>" in result
        assert 'id="cm_1"' in result
        assert 'turns="1-12"' in result
        assert 'id="m01"' in result
        assert 'role="user"' in result
        assert 'preview="Hello"' in result
        assert 'id="m02"' in result
        assert 'role="assistant"' in result
        assert "</compressed_history>" in result

    def test_memory_entries_only(self):
        """C16: 2 MemoryEntry → memory-context with 2 entry elements."""
        builder = LayeredPromptBuilder()
        entries = [
            _entry("preference", "偏好宋干节前后出行", 0.8),
            _entry("fact", "用户来自上海", 0.9),
        ]
        result = builder.build_ephemeral(wm=None, batches=[], memory_entries=entries)
        assert "<memory-context>" in result
        assert 'type="preference"' in result
        assert "偏好宋干节前后出行" in result
        assert 'type="fact"' in result
        assert "用户来自上海" in result
        assert "</memory-context>" in result

    def test_all_three_blocks_present(self):
        """C17: wm + batches + memory_entries → all three XML blocks in output."""
        builder = LayeredPromptBuilder()
        wm = _wm()
        batch = _batch(members=[BatchMember(id="m01", role="user", preview="Hi")])
        entries = [_entry("fact", "Shanghai")]
        result = builder.build_ephemeral(wm=wm, batches=[batch], memory_entries=entries)
        assert "<working_memory" in result
        assert "<compressed_history>" in result
        assert "<memory-context>" in result

    def test_empty_wm_fields_renders_gracefully(self):
        """C18: WM with empty list fields → renders without error, section headers present."""
        builder = LayeredPromptBuilder()
        wm = _wm(
            constraints=[],
            key_decisions=[],
            relevant_files=[],
            next_steps=[],
        )
        result = builder.build_ephemeral(wm=wm, batches=[])
        # Should not raise; section headers present
        assert "CONSTRAINTS_AND_PREFERENCES:" in result
        assert "KEY_DECISIONS:" in result
        assert "RELEVANT_FILES:" in result
        assert "NEXT_STEPS:" in result


# ===========================================================================
# D. PromptBuilder backward compat
# ===========================================================================


class TestPromptBuilderBackwardCompat:
    def test_prompt_builder_emits_deprecation_warning(self):
        """D19: PromptBuilder() emits DeprecationWarning."""
        from neoagent.core.prompt import PromptBuilder

        with pytest.warns(DeprecationWarning, match="deprecated"):
            pb = PromptBuilder()

    def test_prompt_builder_methods_still_work(self):
        """D20: PromptBuilder original methods remain functional."""
        from neoagent.core.prompt import PromptBuilder, PromptSection

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pb = PromptBuilder()

        sec = PromptSection(name="s", content="hello", priority=0)
        pb.add_section(sec)
        assert "hello" in pb.build()

        pb.register_skill("sk", PromptSection(name="sk", content="skill_text", priority=1))
        assert not pb.is_skill_active("sk")
        pb.activate_skill("sk")
        assert pb.is_skill_active("sk")
        pb.deactivate_skill("sk")
        assert not pb.is_skill_active("sk")
        assert "skill_text" not in pb.build()
