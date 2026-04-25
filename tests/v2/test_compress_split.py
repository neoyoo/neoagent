# tests/v2/test_compress_split.py
"""B2a — _split_messages_for_compression.

Tests cover:
  A. Fewer than 5 user turns → nothing to compress
  B. Exactly 5 user turns → nothing to compress
  C. 6 user turns → turn 0 compressed, turns 1..5 kept
  D. 10 user turns with tool round-trips → turns 0..4 compressed, 5..9 kept
  E. Messages with turn=None interleaved → stay in to_keep at original position
"""
from __future__ import annotations

import pytest

from neoagent.core.compress import _split_messages_for_compression
from neoagent.core.types import Message, TextBlock, ToolResultBlock, ToolUseBlock


def _user(content: str, turn: int | None) -> Message:
    return Message(role="user", content=content, turn=turn)


def _assistant(content: str, turn: int | None) -> Message:
    return Message(role="assistant", content=content, turn=turn)


def _tool_use_msg(turn: int) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseBlock(id=f"t{turn}", name="bash", input={})],
        turn=turn,
    )


def _tool_result_msg(turn: int) -> Message:
    return Message(
        role="user",
        content=[ToolResultBlock(tool_use_id=f"t{turn}", content="ok")],
        turn=turn,
    )


class TestSplitFewerThan5Turns:
    def test_zero_messages(self):
        to_compress, to_keep = _split_messages_for_compression([], keep_recent_user_turns=5)
        assert to_compress == []
        assert to_keep == []

    def test_three_user_turns_no_compression(self):
        """3 distinct user turns → to_compress=[], to_keep=all."""
        msgs = [
            _user("u0", turn=0), _assistant("a0", turn=0),
            _user("u1", turn=1), _assistant("a1", turn=1),
            _user("u2", turn=2), _assistant("a2", turn=2),
        ]
        to_compress, to_keep = _split_messages_for_compression(msgs, keep_recent_user_turns=5)
        assert to_compress == []
        assert to_keep == msgs

    def test_four_user_turns_no_compression(self):
        """4 distinct user turns → to_compress=[], to_keep=all."""
        msgs = [_user(f"u{i}", turn=i) for i in range(4)]
        for msg in [_assistant(f"a{i}", turn=i) for i in range(4)]:
            msgs.append(msg)
        # re-sort to interleaved
        msgs = []
        for i in range(4):
            msgs.append(_user(f"u{i}", turn=i))
            msgs.append(_assistant(f"a{i}", turn=i))
        to_compress, to_keep = _split_messages_for_compression(msgs, keep_recent_user_turns=5)
        assert to_compress == []
        assert to_keep == msgs


class TestSplitExactly5Turns:
    def test_exactly_5_user_turns_no_compression(self):
        """Exactly 5 distinct user turns → to_compress=[], to_keep=all."""
        msgs = []
        for i in range(5):
            msgs.append(_user(f"u{i}", turn=i))
            msgs.append(_assistant(f"a{i}", turn=i))
        to_compress, to_keep = _split_messages_for_compression(msgs, keep_recent_user_turns=5)
        assert to_compress == []
        assert to_keep == msgs


class TestSplit6Turns:
    def test_6_user_turns_oldest_compressed(self):
        """6 distinct user turns → turn 0 messages go to to_compress, turns 1..5 kept."""
        msgs = []
        for i in range(6):
            msgs.append(_user(f"u{i}", turn=i))
            msgs.append(_assistant(f"a{i}", turn=i))
        # total 12 messages: turns 0..5

        to_compress, to_keep = _split_messages_for_compression(msgs, keep_recent_user_turns=5)

        # to_compress = messages with turn=0 (user+assistant)
        compressed_turns = {m.turn for m in to_compress}
        kept_turns = {m.turn for m in to_keep if m.turn is not None}

        assert compressed_turns == {0}
        assert kept_turns == {1, 2, 3, 4, 5}

        # Order preserved
        assert to_compress == [m for m in msgs if m.turn == 0]
        assert to_keep == [m for m in msgs if m.turn != 0]

    def test_6_turns_no_message_lost(self):
        """to_compress + to_keep must equal original messages."""
        msgs = []
        for i in range(6):
            msgs.append(_user(f"u{i}", turn=i))
            msgs.append(_assistant(f"a{i}", turn=i))
        to_compress, to_keep = _split_messages_for_compression(msgs, keep_recent_user_turns=5)
        assert to_compress + to_keep == msgs


class TestSplit10TurnsWithToolRoundTrips:
    def test_10_turns_with_tool_roundtrips(self):
        """10 user turns with tool round-trips → turns 0..4 compressed, 5..9 kept."""
        msgs = []
        for i in range(10):
            msgs.append(_user(f"u{i}", turn=i))
            msgs.append(_tool_use_msg(i))
            msgs.append(_tool_result_msg(i))
            msgs.append(_assistant(f"a{i}", turn=i))

        to_compress, to_keep = _split_messages_for_compression(msgs, keep_recent_user_turns=5)

        compressed_turns = {m.turn for m in to_compress if m.turn is not None}
        kept_turns = {m.turn for m in to_keep if m.turn is not None}

        assert compressed_turns == {0, 1, 2, 3, 4}
        assert kept_turns == {5, 6, 7, 8, 9}

    def test_10_turns_no_message_lost(self):
        """to_compress + to_keep == original messages (no loss, no duplication)."""
        msgs = []
        for i in range(10):
            msgs.append(_user(f"u{i}", turn=i))
            msgs.append(_tool_use_msg(i))
            msgs.append(_tool_result_msg(i))
            msgs.append(_assistant(f"a{i}", turn=i))

        to_compress, to_keep = _split_messages_for_compression(msgs, keep_recent_user_turns=5)
        assert to_compress + to_keep == msgs


class TestSplitTurnNoneInterleaved:
    def test_turn_none_messages_stay_in_to_keep(self):
        """Messages with turn=None are always in to_keep at their original position."""
        legacy_msg = Message(role="user", content="legacy no-turn")  # turn=None
        msgs = [
            _user("u0", turn=0), _assistant("a0", turn=0),
            legacy_msg,  # interleaved, no turn
            _user("u1", turn=1), _assistant("a1", turn=1),
            _user("u2", turn=2), _assistant("a2", turn=2),
            _user("u3", turn=3), _assistant("a3", turn=3),
            _user("u4", turn=4), _assistant("a4", turn=4),
            _user("u5", turn=5), _assistant("a5", turn=5),
        ]
        to_compress, to_keep = _split_messages_for_compression(msgs, keep_recent_user_turns=5)

        # legacy_msg must be in to_keep (turn=None always kept)
        assert legacy_msg in to_keep
        assert legacy_msg not in to_compress

    def test_turn_none_only_messages_no_compression(self):
        """All messages have turn=None → to_compress=[], to_keep=all."""
        msgs = [
            Message(role="user", content="old1"),
            Message(role="assistant", content="old2"),
        ]
        to_compress, to_keep = _split_messages_for_compression(msgs, keep_recent_user_turns=5)
        assert to_compress == []
        assert to_keep == msgs

    def test_no_message_lost_with_none_turns(self):
        """No message is lost: set(to_compress) + set(to_keep) == set(msgs) when None-turn messages present.
        Note: order-preserving concatenation may not hold when None-turn msgs fall before the split boundary.
        """
        legacy = Message(role="user", content="legacy")
        msgs = [
            legacy,
            _user("u0", turn=0), _assistant("a0", turn=0),
            _user("u1", turn=1), _assistant("a1", turn=1),
            _user("u2", turn=2), _assistant("a2", turn=2),
            _user("u3", turn=3), _assistant("a3", turn=3),
            _user("u4", turn=4), _assistant("a4", turn=4),
            _user("u5", turn=5), _assistant("a5", turn=5),
        ]
        to_compress, to_keep = _split_messages_for_compression(msgs, keep_recent_user_turns=5)
        # No message lost and no duplication
        assert len(to_compress) + len(to_keep) == len(msgs)
        assert set(id(m) for m in to_compress) | set(id(m) for m in to_keep) == set(id(m) for m in msgs)
        # legacy (turn=None) must be in to_keep, not in to_compress
        assert legacy in to_keep
        assert legacy not in to_compress
