# tests/v2/test_agent_v2_wiring.py
"""TDD tests for Phase 7 Batch 1 — NeoAgent wires v2 components.

Task 7.1 per plan 2026-04-22-neoagent-v2.0-sdk.md

Contract refs: C2 (ABCs), C5 (hook integration)
Spec refs: § 15 SDK integration points
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from neoagent.config import NeoAgentConfig
from neoagent.v2.abc import (
    CompressionStrategy,
    MemoryProvider,
    MemoryReviewStrategy,
    WorkingMemoryStore,
)
from neoagent.v2.stores import InMemoryWorkingMemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> NeoAgentConfig:
    return NeoAgentConfig(api_key="test-key", model="test-model", **kwargs)


def _make_agent(config: NeoAgentConfig | None = None):
    """Create a NeoAgent with a mocked provider (no real API key needed)."""
    from neoagent.agent import NeoAgent

    if config is None:
        config = _make_config()

    with patch("neoagent.agent._create_provider") as mock_create:
        mock_prov = MagicMock()
        mock_prov.get_context_window.return_value = 200_000
        mock_prov.model = "test-model"
        mock_create.return_value = mock_prov
        agent = NeoAgent(config)
    return agent


# ---------------------------------------------------------------------------
# A. Config extension
# ---------------------------------------------------------------------------


class TestConfigExtension:
    def test_new_fields_have_defaults(self):
        """A1: NeoAgentConfig with only required fields does not raise."""
        # Suppress DeprecationWarning from PromptBuilder during NeoAgent construction;
        # here we only test Config itself.
        cfg = NeoAgentConfig(api_key="x", model="y")
        assert cfg.wm_store is None
        assert cfg.compression_strategy is None
        assert cfg.memory_review_strategy is None
        assert cfg.memory_provider is None
        assert cfg.enable_source_wrap is True
        assert cfg.enable_security_prompt_blocks is True

    def test_config_accepts_custom_wm_store(self):
        """A2: Passing a custom wm_store is preserved in config."""
        custom_store = InMemoryWorkingMemoryStore()
        cfg = _make_config(wm_store=custom_store)
        assert cfg.wm_store is custom_store


# ---------------------------------------------------------------------------
# B. NeoAgent assembly
# ---------------------------------------------------------------------------


class TestNeoAgentAssembly:
    def test_wm_store_default_is_in_memory(self):
        """B3: When no wm_store is configured, agent.wm_store is InMemoryWorkingMemoryStore."""
        agent = _make_agent()
        assert isinstance(agent.wm_store, InMemoryWorkingMemoryStore)

    def test_wm_store_custom_preserved(self):
        """B4: When config.wm_store is set, agent.wm_store is the same instance."""
        custom_store = InMemoryWorkingMemoryStore()
        cfg = _make_config(wm_store=custom_store)
        agent = _make_agent(cfg)
        assert agent.wm_store is custom_store

    def test_query_loop_receives_wm_store(self):
        """B5: agent._loop._wm_store is the same object as agent.wm_store."""
        agent = _make_agent()
        assert agent._loop._wm_store is agent.wm_store

    def test_compressor_strategy_none_by_default(self):
        """B6: Default config has no compression_strategy, compressor._strategy is None."""
        agent = _make_agent()
        assert agent._compressor._strategy is None

    def test_compressor_receives_event_bus(self):
        """B7: agent._compressor._bus is the same object as agent.event_bus."""
        agent = _make_agent()
        assert agent._compressor._bus is agent.event_bus

    def test_compressor_strategy_forwarded(self):
        """B_extra: When compression_strategy is set, compressor receives it."""
        # Create a concrete minimal strategy
        class _StubStrategy(CompressionStrategy):
            async def compress(self, context):
                from neoagent.v2.schema import CompressionDelta
                return CompressionDelta(batch_summary="stub", batch_members=[], working_memory_delta=[])

        strategy = _StubStrategy()
        cfg = _make_config(compression_strategy=strategy)
        agent = _make_agent(cfg)
        assert agent._compressor._strategy is strategy


# ---------------------------------------------------------------------------
# C. Hook registration
# ---------------------------------------------------------------------------


class TestHookRegistration:
    def test_source_wrap_executor_flag_enabled_by_default(self):
        """C8: With enable_source_wrap=True (default), executor._enable_source_wrap is True.

        Source wrapping is applied inline in ToolExecutor (not via hook) to avoid
        PostToolCallEvent shape mismatch (tool_name:str vs expected tool:BaseTool).
        """
        agent = _make_agent()
        assert agent._executor._enable_source_wrap is True

    def test_source_wrap_executor_flag_disabled_when_config_off(self):
        """C9: With enable_source_wrap=False, executor._enable_source_wrap is False."""
        cfg = _make_config(enable_source_wrap=False)
        agent = _make_agent(cfg)
        assert agent._executor._enable_source_wrap is False


# ---------------------------------------------------------------------------
# D. Security prompt blocks
# ---------------------------------------------------------------------------


class TestSecurityPromptBlocks:
    def test_four_security_sections_injected_by_default(self):
        """D10: With enable_security_prompt_blocks=True (default), 4 security sections exist."""
        agent = _make_agent()
        section_names = {s.name for s in agent._prompt_builder._sections}
        expected = {
            "v2_security_hard_constraints",
            "v2_security_heuristic_guidelines",
            "v2_security_security_boundary",
            "v2_security_tag_contract",
        }
        assert expected.issubset(section_names)

    def test_security_sections_not_injected_when_disabled(self):
        """D11: With enable_security_prompt_blocks=False, security sections are absent."""
        cfg = _make_config(enable_security_prompt_blocks=False)
        agent = _make_agent(cfg)
        section_names = {s.name for s in agent._prompt_builder._sections}
        for expected_name in (
            "v2_security_hard_constraints",
            "v2_security_heuristic_guidelines",
            "v2_security_security_boundary",
            "v2_security_tag_contract",
        ):
            assert expected_name not in section_names


# ---------------------------------------------------------------------------
# E. Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_legacy_construction_does_not_raise(self):
        """E12: NeoAgent with minimal config (no v2 fields) constructs without error."""
        cfg = NeoAgentConfig(api_key="x", model="y")
        agent = _make_agent(cfg)
        # Basic sanity: agent has key attributes
        assert agent._loop is not None
        assert agent._prompt_builder is not None
        assert agent._hook_manager is not None

    def test_memory_provider_stored_for_later_phases(self):
        """E_extra: memory_provider is stored on agent for future phases."""

        class _StubProvider(MemoryProvider):
            async def search(self, user_id, query, k=5):
                return []
            async def upsert(self, entries):
                pass
            async def delete(self, user_id, memory_ids):
                pass
            async def reinforce(self, user_id, memory_id):
                pass

        provider = _StubProvider()
        cfg = _make_config(memory_provider=provider)
        agent = _make_agent(cfg)
        assert agent._memory_provider is provider

    def test_memory_review_strategy_stored_for_later_phases(self):
        """E_extra2: memory_review_strategy is stored on agent for future phases."""

        class _StubReview(MemoryReviewStrategy):
            async def review(self, session_id, user_id, messages, wm):
                return []

        strategy = _StubReview()
        cfg = _make_config(memory_review_strategy=strategy)
        agent = _make_agent(cfg)
        assert agent._memory_review_strategy is strategy
