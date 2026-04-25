# tests/v2/test_system_prompt_injection.py
"""B3: <compressed_history> injection into system prompt via LayeredPromptBuilder.

Contract:
  - SessionState with 0 batches → system prompt does NOT contain <compressed_history>
  - SessionState with 1 batch (1 member, turn=0) → system prompt contains expected XML
  - SessionState with 2 batches → both appear in order
  - WM + batches → WM section appears BEFORE compressed_history
  - Layer ordering: identity < persistent_memory < capabilities < security < WM < compressed_history < memory_context
  - Integration: after compression fires, next chat() call contains <compressed_history> in system sent to provider
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from neoagent.core.loop import QueryLoop
from neoagent.core.prompt import LayeredPromptBuilder, PromptBuilder, PromptSection
from neoagent.core.types import Message, TextBlock
from neoagent.session import Session, SessionState
from neoagent.v2.schema import Batch, BatchMember, Layer, WorkingMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime(2026, 4, 24, 12, 0, 0)


def _batch(
    batch_id: str = "cm_1",
    turns_from: int = 0,
    turns_to: int = 2,
    members: list[BatchMember] | None = None,
) -> Batch:
    return Batch(
        session_id="s1",
        batch_id=batch_id,
        turns_from=turns_from,
        turns_to=turns_to,
        time_from=_now(),
        time_to=_now(),
        summary="PROGRESS: test done",
        members=members or [],
        trigger="token_threshold",
        created_at=_now(),
    )


def _wm(version: int = 1, at_turn: int = 0) -> WorkingMemory:
    return WorkingMemory(
        session_id="s1",
        version=version,
        at_turn=at_turn,
        constraints_and_preferences=["c01: prefer short"],
        progress="in progress",
        key_decisions=[],
        relevant_files=[],
        next_steps=[],
        critical_context="session ctx",
        updated_by="framework_init",
        updated_at=_now(),
    )


def _make_builder_with_identity(identity: str = "You are TestBot.") -> LayeredPromptBuilder:
    builder = LayeredPromptBuilder()
    builder.register_layer_section(
        Layer.IDENTITY,
        PromptSection(name="identity", content=identity, priority=0),
    )
    return builder


def _text_response(text: str = "ok"):
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [TextBlock(text=text)]
    resp.tool_use_blocks = []
    resp.input_tokens = 10
    resp.output_tokens = 5
    return resp


def _make_provider(response=None):
    p = MagicMock()
    p.model = "mock"
    p.get_context_window.return_value = 200_000
    p.create = AsyncMock(return_value=response or _text_response())
    return p


def _make_loop(layered: LayeredPromptBuilder) -> QueryLoop:
    """Create a minimal QueryLoop with given LayeredPromptBuilder."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pb = PromptBuilder()
    pb.add_section(PromptSection(name="identity", content="You are TestBot.", priority=0))
    registry = MagicMock()
    registry.get_schemas.return_value = []
    return QueryLoop(
        provider=_make_provider(),
        tool_registry=registry,
        prompt_builder=pb,
        layered_prompt_builder=layered,
    )


# ===========================================================================
# A. No batches — no <compressed_history> tag
# ===========================================================================


class TestNoBatches:
    @pytest.mark.asyncio
    async def test_no_compressed_history_when_no_batches(self):
        """System prompt must NOT contain <compressed_history> when batches=[].

        B3 contract: empty section → suppressed (no empty-tag pollution).
        """
        captured: list[str] = []
        provider = _make_provider()

        async def _spy(system, messages, tools, **kw):
            captured.append(system)
            return _text_response()

        provider.create = AsyncMock(side_effect=_spy)
        layered = _make_builder_with_identity()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pb = PromptBuilder()
        pb.add_section(PromptSection(name="identity", content="You are TestBot.", priority=0))
        registry = MagicMock()
        registry.get_schemas.return_value = []
        loop = QueryLoop(
            provider=provider,
            tool_registry=registry,
            prompt_builder=pb,
            layered_prompt_builder=layered,
        )

        session = Session.create()
        session.messages.append(Message(role="user", content="hello"))
        await loop.run(session=session)

        assert captured, "provider.create was never called"
        assert "<compressed_history>" not in captured[0]
        assert "</compressed_history>" not in captured[0]


# ===========================================================================
# B. One batch, one member
# ===========================================================================


class TestOneBatch:
    @pytest.mark.asyncio
    async def test_compressed_history_present_with_one_batch(self):
        """System prompt contains <compressed_history> with correct batch/turn structure."""
        captured: list[str] = []
        provider = _make_provider()

        async def _spy(system, messages, tools, **kw):
            captured.append(system)
            return _text_response()

        provider.create = AsyncMock(side_effect=_spy)
        layered = _make_builder_with_identity()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pb = PromptBuilder()
        pb.add_section(PromptSection(name="identity", content="You are TestBot.", priority=0))
        registry = MagicMock()
        registry.get_schemas.return_value = []
        loop = QueryLoop(
            provider=provider,
            tool_registry=registry,
            prompt_builder=pb,
            layered_prompt_builder=layered,
        )

        session = Session.create()
        session.messages.append(Message(role="user", content="hello"))
        member = BatchMember(id="m1", role="user", preview="hello", turn=0)
        session.state.batches.append(_batch(batch_id="cm_1", members=[member]))
        await loop.run(session=session)

        assert captured, "provider.create was never called"
        system = captured[0]
        assert "<compressed_history>" in system
        assert 'id="cm_1"' in system
        assert '<turn n="0">' in system
        assert 'id="m1"' in system
        assert "</compressed_history>" in system


# ===========================================================================
# C. Two batches — both present in order
# ===========================================================================


class TestTwoBatches:
    @pytest.mark.asyncio
    async def test_two_batches_both_present_in_order(self):
        """Two batches → both appear; cm_1 before cm_2 (insertion order)."""
        captured: list[str] = []
        provider = _make_provider()

        async def _spy(system, messages, tools, **kw):
            captured.append(system)
            return _text_response()

        provider.create = AsyncMock(side_effect=_spy)
        layered = _make_builder_with_identity()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pb = PromptBuilder()
        pb.add_section(PromptSection(name="identity", content="You are TestBot.", priority=0))
        registry = MagicMock()
        registry.get_schemas.return_value = []
        loop = QueryLoop(
            provider=provider,
            tool_registry=registry,
            prompt_builder=pb,
            layered_prompt_builder=layered,
        )

        session = Session.create()
        session.messages.append(Message(role="user", content="hello"))
        session.state.batches.append(_batch(batch_id="cm_1", turns_from=0, turns_to=2))
        session.state.batches.append(_batch(batch_id="cm_2", turns_from=3, turns_to=5))
        await loop.run(session=session)

        assert captured
        system = captured[0]
        assert 'id="cm_1"' in system
        assert 'id="cm_2"' in system
        assert system.index('id="cm_1"') < system.index('id="cm_2"')


# ===========================================================================
# D. WM + batches — WM before compressed_history
# ===========================================================================


class TestWMBeforeCompressedHistory:
    @pytest.mark.asyncio
    async def test_wm_appears_before_compressed_history(self):
        """With WM + batches, <working_memory> precedes <compressed_history>."""
        captured: list[str] = []
        provider = _make_provider()

        async def _spy(system, messages, tools, **kw):
            captured.append(system)
            return _text_response()

        provider.create = AsyncMock(side_effect=_spy)
        layered = _make_builder_with_identity()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pb = PromptBuilder()
        pb.add_section(PromptSection(name="identity", content="You are TestBot.", priority=0))
        registry = MagicMock()
        registry.get_schemas.return_value = []
        loop = QueryLoop(
            provider=provider,
            tool_registry=registry,
            prompt_builder=pb,
            layered_prompt_builder=layered,
        )

        session = Session.create()
        session.messages.append(Message(role="user", content="hello"))
        session.state._current_wm = _wm()
        session.state.batches.append(_batch(batch_id="cm_1"))
        await loop.run(session=session)

        assert captured
        system = captured[0]
        assert "<working_memory" in system
        assert "<compressed_history>" in system
        assert system.index("<working_memory") < system.index("<compressed_history>")


# ===========================================================================
# E. Layer ordering via build_ephemeral directly
# ===========================================================================


class TestLayerOrdering:
    def test_wm_before_compressed_history_in_build_ephemeral(self):
        """build_ephemeral order: <working_memory> before <compressed_history>."""
        builder = LayeredPromptBuilder()
        wm = _wm()
        batch = _batch(
            batch_id="cm_1",
            members=[BatchMember(id="m1", role="user", preview="hi", turn=0)],
        )
        result = builder.build_ephemeral(wm=wm, batches=[batch])

        assert "<working_memory" in result
        assert "<compressed_history>" in result
        assert result.index("<working_memory") < result.index("<compressed_history>")

    def test_compressed_history_before_memory_context(self):
        """build_ephemeral order: <compressed_history> before <memory-context>."""
        from neoagent.v2.schema import MemoryEntry

        builder = LayeredPromptBuilder()
        batch = _batch(batch_id="cm_1")
        entries = [
            MemoryEntry(
                user_id="u1",
                memory_id="me1",
                type="fact",
                category=None,
                content="Neo is from Shanghai",
                confidence=0.9,
            )
        ]
        result = builder.build_ephemeral(wm=None, batches=[batch], memory_entries=entries)

        assert "<compressed_history>" in result
        assert "<memory-context>" in result
        assert result.index("<compressed_history>") < result.index("<memory-context>")

    def test_all_three_order_wm_compressed_memory(self):
        """Full order: working_memory < compressed_history < memory-context."""
        from neoagent.v2.schema import MemoryEntry

        builder = LayeredPromptBuilder()
        wm = _wm()
        batch = _batch(batch_id="cm_1")
        entries = [
            MemoryEntry(
                user_id="u1",
                memory_id="me1",
                type="fact",
                category=None,
                content="test entry",
                confidence=0.9,
            )
        ]
        result = builder.build_ephemeral(wm=wm, batches=[batch], memory_entries=entries)

        pos_wm = result.index("<working_memory")
        pos_ch = result.index("<compressed_history>")
        pos_mc = result.index("<memory-context>")
        assert pos_wm < pos_ch < pos_mc


# ===========================================================================
# F. Integration: compression fires → next turn LLM sees <compressed_history>
# ===========================================================================


class TestIntegrationCompressionThenInject:
    @pytest.mark.asyncio
    async def test_compressed_history_in_system_after_compression(self):
        """After a batch is added to session.state.batches, the next loop turn
        sends <compressed_history> to the provider in the system prompt.

        This simulates B2a compression writing to session_state.batches, then
        B3 injecting the block every turn.
        """
        captured_systems: list[str] = []
        provider = _make_provider()

        async def _spy(system, messages, tools, **kw):
            captured_systems.append(system)
            return _text_response("done")

        provider.create = AsyncMock(side_effect=_spy)
        layered = _make_builder_with_identity()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pb = PromptBuilder()
        pb.add_section(PromptSection(name="identity", content="You are TestBot.", priority=0))
        registry = MagicMock()
        registry.get_schemas.return_value = []
        loop = QueryLoop(
            provider=provider,
            tool_registry=registry,
            prompt_builder=pb,
            layered_prompt_builder=layered,
        )

        # Turn 1: no batches yet — no <compressed_history>
        session = Session.create()
        session.messages.append(Message(role="user", content="first message"))
        await loop.run(session=session)

        assert captured_systems
        assert "<compressed_history>" not in captured_systems[0]

        # Simulate compression: add a batch to session.state (as B2a would)
        member = BatchMember(id="m1", role="user", preview="first message", turn=1)
        session.state.batches.append(
            _batch(batch_id="cm_1", turns_from=1, turns_to=1, members=[member])
        )

        # Turn 2: with batch present — <compressed_history> must be in system
        captured_systems.clear()
        session.messages.append(Message(role="user", content="second message"))
        await loop.run(session=session)

        assert captured_systems
        assert "<compressed_history>" in captured_systems[0]
        assert 'id="cm_1"' in captured_systems[0]
        assert '<turn n="1">' in captured_systems[0]
