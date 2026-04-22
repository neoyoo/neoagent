"""Tests for Task 6.3 — security prompt blocks.

spec § 6 Layer 4 SECURITY (lines 770-890)
"""
from __future__ import annotations

import pytest

from neoagent.core.prompt import LayeredPromptBuilder, PromptSection
from neoagent.v2.prompts.security_blocks import (
    HARD_CONSTRAINTS_CONTENT,
    HEURISTIC_GUIDELINES_CONTENT,
    SECURITY_BOUNDARY_CONTENT,
    TAG_CONTRACT_CONTENT,
    build_security_sections,
)
from neoagent.v2.schema import Layer


def test_build_security_sections_returns_four():
    """build_security_sections returns exactly 4 PromptSections."""
    sections = build_security_sections()
    assert len(sections) == 4
    names = {s.name for s in sections}
    assert names == {"hard_constraints", "heuristic_guidelines", "security_boundary", "tag_contract"}


def test_hard_constraints_goal_immutable():
    """HARD_CONSTRAINTS content mentions goal is immutable after framework_init."""
    lower = HARD_CONSTRAINTS_CONTENT.lower()
    assert "goal" in lower and ("immutable" in lower or "不可" in HARD_CONSTRAINTS_CONTENT)


def test_hard_constraints_update_working_memory_five_turns():
    """HARD_CONSTRAINTS mentions update_working_memory and 5 turn cadence."""
    assert "update_working_memory" in HARD_CONSTRAINTS_CONTENT


def test_hard_constraints_framework_rendering_invariant():
    """HARD_CONSTRAINTS mentions framework rendering invariant."""
    lower = HARD_CONSTRAINTS_CONTENT.lower()
    assert "rendering" in lower or "invariant" in lower or "rendering invariant" in lower or "渲染" in HARD_CONSTRAINTS_CONTENT


def test_security_boundary_authoritative_source():
    """SECURITY_BOUNDARY declares authoritative instruction source."""
    content = SECURITY_BOUNDARY_CONTENT
    assert (
        "authoritative" in content.lower()
        or "指令来源" in content
        or "system prompt" in content.lower()
        or "权威" in content
    )


def test_tag_contract_contains_all_seven_tags():
    """TAG_CONTRACT names all 7 framework-injected tags."""
    content = TAG_CONTRACT_CONTENT
    assert "user_profile" in content
    assert "working_memory" in content
    assert "compressed_history" in content
    assert "memory-context" in content
    assert "freed_tool_results" in content
    assert "recoverable" in content
    assert "source" in content


def test_heuristic_guidelines_soft_nature():
    """HEURISTIC_GUIDELINES indicates non-mandatory nature."""
    content = HEURISTIC_GUIDELINES_CONTENT
    assert (
        "建议" in content
        or "推荐" in content
        or "soft" in content.lower()
        or "期望" in content
        or "不会强制" in content
        or "框架不" in content
    )


def test_build_security_sections_integrates_with_layered_prompt_builder():
    """Sections from build_security_sections() can be registered to LayeredPromptBuilder
    under Layer.SECURITY, and build_system_prompt() includes all 4 block tags.
    """
    builder = LayeredPromptBuilder()
    for sec in build_security_sections():
        builder.register_layer_section(Layer.SECURITY, sec)

    prompt = builder.build_system_prompt()
    assert "<HARD_CONSTRAINTS>" in prompt
    assert "<HEURISTIC_GUIDELINES>" in prompt
    assert "<SECURITY_BOUNDARY>" in prompt
    assert "<TAG_CONTRACT>" in prompt
