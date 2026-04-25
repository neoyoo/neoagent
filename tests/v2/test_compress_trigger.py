# tests/v2/test_compress_trigger.py
"""B2a — should_compress double trigger: turn_count + token_threshold.

Tests cover:
  A. turn_count trigger (turns_since_last_compression >= 10)
  B. token_threshold trigger (tokens > budget * 0.7)
  C. no compression
  D. turn_count wins when both fire simultaneously
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neoagent.core.compress import ContextCompressor
from neoagent.core.types import Message


def _make_compressor() -> ContextCompressor:
    provider = MagicMock()
    provider.get_context_window = MagicMock(return_value=200_000)
    return ContextCompressor(provider=provider)


def _short_msgs() -> list[Message]:
    return [Message(role="user", content="hi")]


def _long_msgs() -> list[Message]:
    # enough tokens to exceed budget * 0.7 when budget is tiny
    return [Message(role="user", content="word " * 1000)]


class TestShouldCompressTurnCount:
    def test_ten_turns_triggers_turn_count(self):
        """turns_since_last_compression=10 → (True, 'turn_count')."""
        c = _make_compressor()
        result = c.should_compress(
            _short_msgs(),
            schemas=[],
            context_budget=200_000,
            turns_since_last_compression=10,
        )
        assert result == (True, "turn_count")

    def test_eleven_turns_also_triggers(self):
        """turns_since_last_compression=11 → (True, 'turn_count')."""
        c = _make_compressor()
        should, reason = c.should_compress(
            _short_msgs(),
            schemas=[],
            context_budget=200_000,
            turns_since_last_compression=11,
        )
        assert should is True
        assert reason == "turn_count"

    def test_nine_turns_no_trigger(self):
        """turns_since_last_compression=9, tokens under budget → (False, '')."""
        c = _make_compressor()
        result = c.should_compress(
            _short_msgs(),
            schemas=[],
            context_budget=200_000,
            turns_since_last_compression=9,
        )
        assert result == (False, "")


class TestShouldCompressTokenThreshold:
    def test_tokens_over_budget_triggers(self):
        """turns_since_last_compression=5, tokens over budget*0.7 → (True, 'token_threshold')."""
        c = _make_compressor()
        # budget=50 ensures long messages exceed 50 * 0.7 = 35 tokens
        result = c.should_compress(
            _long_msgs(),
            schemas=[],
            context_budget=50,
            turns_since_last_compression=5,
        )
        assert result == (True, "token_threshold")

    def test_tokens_under_budget_no_trigger(self):
        """turns_since_last_compression=5, tokens under budget → (False, '')."""
        c = _make_compressor()
        result = c.should_compress(
            _short_msgs(),
            schemas=[],
            context_budget=200_000,
            turns_since_last_compression=5,
        )
        assert result == (False, "")


class TestShouldCompressBothFire:
    def test_turn_count_wins_over_token_threshold(self):
        """turns_since_last_compression=10, tokens also over → (True, 'turn_count') (turn_count checked first)."""
        c = _make_compressor()
        result = c.should_compress(
            _long_msgs(),
            schemas=[],
            context_budget=50,
            turns_since_last_compression=10,
        )
        assert result == (True, "turn_count")


class TestShouldCompressZeroTurns:
    def test_zero_turns_no_trigger_under_budget(self):
        """Fresh session (0 turns, short messages) → (False, '')."""
        c = _make_compressor()
        result = c.should_compress(
            _short_msgs(),
            schemas=[],
            context_budget=200_000,
            turns_since_last_compression=0,
        )
        assert result == (False, "")
