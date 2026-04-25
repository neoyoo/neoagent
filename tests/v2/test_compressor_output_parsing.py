# tests/v2/test_compressor_output_parsing.py
"""B2b: Compressor output parsing with turn field on BatchMember.

Contract:
  - LLM output with turn in batch_members → BatchMember.turn populated
  - LLM output missing turn → fallback to Message.turn lookup
  - LLM output referencing nonexistent msg_id → turn=None, warning logged (not error)
"""
from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neoagent.core.types import Message
from neoagent.v2.schema import BatchMember, CompressionContext
from neoagent.v2.strategies.oneshot_compression import OneShotCompressionStrategy


def _make_strategy() -> OneShotCompressionStrategy:
    return OneShotCompressionStrategy(api_key="test-key", max_retries=1)


def _make_ctx(
    messages: list | None = None,
) -> CompressionContext:
    if messages is None:
        messages = [
            {"id": "m1", "role": "user", "content": "Hello", "turn": 0},
            {"id": "m2", "role": "assistant", "content": "World", "turn": 0},
            {"id": "m3", "role": "user", "content": "More", "turn": 1},
        ]
    return CompressionContext(
        session_id="s1",
        messages=messages,
        previous_batches=[],
        previous_wm={},
        trigger="token_threshold",
    )


def _mock_llm(payload: Any) -> MagicMock:
    """Build a mock that _call_llm will resolve to the given JSON string."""
    text = json.dumps(payload)
    mock_response = MagicMock()
    block = MagicMock(text=text)
    block.type = "text"
    mock_response.content = [block]
    return mock_response


def _valid_payload_with_turn() -> dict:
    return {
        "batch_members": [
            {"id": "m1", "turn": 0, "role": "user", "preview": "hello msg"},
            {"id": "m2", "turn": 0, "role": "assistant", "preview": "world response"},
            {"id": "m3", "turn": 1, "role": "user", "preview": "more msg"},
        ],
        "working_memory_delta": [],
    }


# ── Turn field from LLM output ────────────────────────────────────────────────

class TestTurnFieldFromLLMOutput:

    async def test_batch_members_have_turn_from_llm(self):
        """LLM output with turn field → each BatchMember.turn is set correctly."""
        strategy = _make_strategy()
        payload = _valid_payload_with_turn()

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm(payload)),
        ):
            result = await strategy.compress(_make_ctx())

        assert result.batch_members[0].turn == 0
        assert result.batch_members[1].turn == 0
        assert result.batch_members[2].turn == 1

    async def test_batch_member_turn_type_is_int(self):
        """BatchMember.turn is coerced to int from LLM JSON number."""
        strategy = _make_strategy()
        payload = _valid_payload_with_turn()

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm(payload)),
        ):
            result = await strategy.compress(_make_ctx())

        for member in result.batch_members:
            assert isinstance(member.turn, int)

    async def test_all_three_members_parsed_with_turns(self):
        """3 grouped batch_members → 3 BatchMember objects with correct turns."""
        strategy = _make_strategy()
        payload = _valid_payload_with_turn()

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm(payload)),
        ):
            result = await strategy.compress(_make_ctx())

        assert len(result.batch_members) == 3
        turns = [m.turn for m in result.batch_members]
        assert turns == [0, 0, 1]


# ── Fallback: turn missing from LLM → lookup from Message.turn ───────────────

class TestTurnFallbackFromMessage:

    async def test_missing_turn_in_llm_output_falls_back_to_message_turn(self):
        """When LLM omits 'turn' on a member → look up turn from ctx.messages."""
        strategy = _make_strategy()
        payload = _valid_payload_with_turn()
        # Remove turn from all batch_members (LLM forgot to include it)
        for member in payload["batch_members"]:
            del member["turn"]

        ctx = _make_ctx(messages=[
            {"id": "m1", "role": "user", "content": "Hello", "turn": 5},
            {"id": "m2", "role": "assistant", "content": "World", "turn": 5},
            {"id": "m3", "role": "user", "content": "More", "turn": 6},
        ])

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm(payload)),
        ):
            result = await strategy.compress(ctx)

        assert result.batch_members[0].turn == 5
        assert result.batch_members[1].turn == 5
        assert result.batch_members[2].turn == 6

    async def test_partial_turn_field_uses_mix_of_llm_and_fallback(self):
        """First member has turn from LLM, second is missing — fallback for second."""
        strategy = _make_strategy()
        payload = {
            "batch_members": [
                {"id": "m1", "turn": 3, "role": "user", "preview": "msg1"},
                # m2: no turn field
                {"id": "m2", "role": "assistant", "preview": "msg2"},
            ],
            "working_memory_delta": [],
        }
        ctx = _make_ctx(messages=[
            {"id": "m1", "role": "user", "content": "a", "turn": 3},
            {"id": "m2", "role": "assistant", "content": "b", "turn": 3},
        ])

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm(payload)),
        ):
            result = await strategy.compress(ctx)

        assert result.batch_members[0].turn == 3
        assert result.batch_members[1].turn == 3  # from Message.turn fallback

    async def test_fallback_works_with_message_objects(self):
        """Message pydantic objects (not dicts) are used in fallback turn lookup."""
        strategy = _make_strategy()
        payload = {
            "batch_members": [
                {"id": "m7", "role": "user", "preview": "msg"},  # no turn
            ],
            "working_memory_delta": [],
        }
        ctx = _make_ctx(messages=[
            Message(id="m7", role="user", content="hello", turn=9),
        ])

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm(payload)),
        ):
            result = await strategy.compress(ctx)

        assert result.batch_members[0].turn == 9


# ── Nonexistent msg_id → turn=None, warning logged ───────────────────────────

class TestOrphanMsgIdTurnNone:

    async def test_nonexistent_msg_id_gets_turn_none_not_error(self, caplog):
        """LLM references msg_id not in ctx.messages (empty context) → turn=None, warning, no raise."""
        strategy = _make_strategy()
        # Use empty ctx.messages so existing_ids is empty → rule 1 skipped
        payload = {
            "batch_members": [
                {"id": "m99", "role": "user", "preview": "ghost"},  # no turn, not in ctx
            ],
            "working_memory_delta": [],
        }
        ctx = _make_ctx(messages=[])  # empty → rule 1 skipped, but turn lookup also fails

        with caplog.at_level(logging.WARNING, logger="neoagent.v2.strategies.oneshot_compression"):
            with patch.object(
                strategy._client.messages, "create",
                new=AsyncMock(return_value=_mock_llm(payload)),
            ):
                result = await strategy.compress(ctx)

        # Should NOT raise; member turn=None
        assert result.batch_members[0].turn is None
        # Warning should be logged
        assert any("m99" in record.message for record in caplog.records)

    async def test_nonexistent_msg_id_warning_contains_msg_id(self, caplog):
        """Warning message includes the orphan msg_id."""
        strategy = _make_strategy()
        payload = {
            "batch_members": [
                {"id": "m42", "role": "user", "preview": "ghost"},
            ],
            "working_memory_delta": [],
        }
        ctx = _make_ctx(messages=[])

        with caplog.at_level(logging.WARNING, logger="neoagent.v2.strategies.oneshot_compression"):
            with patch.object(
                strategy._client.messages, "create",
                new=AsyncMock(return_value=_mock_llm(payload)),
            ):
                result = await strategy.compress(ctx)

        warning_messages = [r.message for r in caplog.records]
        assert any("m42" in m for m in warning_messages)

    async def test_valid_members_before_orphan_still_parsed(self, caplog):
        """Members before the orphan (turn=None) member are parsed correctly."""
        strategy = _make_strategy()
        payload = {
            "batch_members": [
                # m99: not in ctx (empty) → turn=None
                {"id": "m99", "role": "user", "preview": "ghost"},
            ],
            "working_memory_delta": [],
        }
        ctx = _make_ctx(messages=[])

        with caplog.at_level(logging.WARNING):
            with patch.object(
                strategy._client.messages, "create",
                new=AsyncMock(return_value=_mock_llm(payload)),
            ):
                result = await strategy.compress(ctx)

        assert len(result.batch_members) == 1
        assert result.batch_members[0].id == "m99"
        assert result.batch_members[0].turn is None


# ── BatchMember dataclass turn field ─────────────────────────────────────────

class TestBatchMemberTurnField:

    def test_batch_member_default_turn_is_none(self):
        """BatchMember without turn defaults to None (back-compat)."""
        member = BatchMember(id="m1", role="user", preview="test")
        assert member.turn is None

    def test_batch_member_with_turn(self):
        """BatchMember with explicit turn stores it correctly."""
        member = BatchMember(id="m1", role="user", preview="test", turn=3)
        assert member.turn == 3

    def test_batch_member_turn_zero(self):
        """turn=0 is distinct from turn=None."""
        member = BatchMember(id="m1", role="user", preview="test", turn=0)
        assert member.turn == 0
        assert member.turn is not None


# ── Bug #2: batch_summary silently ignored ────────────────────────────────────

class TestBatchSummarySilentlyIgnored:
    """Bug #2 fix: legacy LLM output with batch_summary → silently ignored, no raise."""

    async def test_legacy_payload_with_batch_summary_does_not_raise(self):
        """LLM still outputs batch_summary → parsed successfully, no error."""
        strategy = _make_strategy()
        payload = {
            "batch_summary": {
                "CONSTRAINTS_AND_PREFERENCES": [],
                "PROGRESS": "done",
                "KEY_DECISIONS": [],
                "RELEVANT_FILES": [],
                "NEXT_STEPS": [],
                "CRITICAL_CONTEXT": "",
            },
            "batch_members": [
                {"id": "m1", "turn": 0, "role": "user", "preview": "hello"},
            ],
            "working_memory_delta": [],
        }

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm(payload)),
        ):
            result = await strategy.compress(_make_ctx())

        # Should not raise; batch_members parsed correctly
        assert len(result.batch_members) == 1
        assert result.batch_members[0].id == "m1"

    async def test_legacy_payload_batch_summary_not_on_result(self):
        """CompressionDelta.batch_summary is None for new-style results."""
        strategy = _make_strategy()
        payload = {
            "batch_members": [
                {"id": "m1", "turn": 0, "role": "user", "preview": "hello"},
            ],
            "working_memory_delta": [],
        }

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm(payload)),
        ):
            result = await strategy.compress(_make_ctx())

        assert result.batch_summary is None
