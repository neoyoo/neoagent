# tests/v2/test_compressor_prompt_formatting.py
"""B2b: Compressor prompt turn-grouped formatting + msg_id_range_hint.

Contract:
  - Messages formatted as grouped Turn N: blocks
  - msg_id_range_hint has correct m1..mN and counts
  - turn=None messages go under Turn (unknown):
"""
from __future__ import annotations

import pytest

from neoagent.core.types import Message, TextBlock
from neoagent.v2.schema import CompressionContext
from neoagent.v2.strategies.oneshot_compression import OneShotCompressionStrategy


def _make_strategy() -> OneShotCompressionStrategy:
    return OneShotCompressionStrategy(api_key="test-key")


def _make_ctx(messages: list) -> CompressionContext:
    return CompressionContext(
        session_id="s1",
        messages=messages,
        previous_batches=[],
        previous_wm={},
        trigger="token_threshold",
    )


# ── Turn-label grouping ───────────────────────────────────────────────────────

class TestTurnGroupedFormatting:

    def test_three_user_turns_produce_turn_labels(self):
        """3 user turns → Turn 0:, Turn 1:, Turn 2: labels in formatted output."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "hi", "turn": 0},
            {"id": "m2", "role": "assistant", "content": "hello", "turn": 0},
            {"id": "m3", "role": "user", "content": "how are you", "turn": 1},
            {"id": "m4", "role": "assistant", "content": "fine", "turn": 1},
            {"id": "m5", "role": "user", "content": "ok bye", "turn": 2},
            {"id": "m6", "role": "assistant", "content": "bye", "turn": 2},
        ]
        ctx = _make_ctx(messages)
        prompt = strategy._build_prompt(ctx)

        assert "Turn 0:" in prompt
        assert "Turn 1:" in prompt
        assert "Turn 2:" in prompt

    def test_turn_0_messages_before_turn_1(self):
        """Turn 0 messages appear before Turn 1 messages in output."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "first", "turn": 0},
            {"id": "m2", "role": "user", "content": "second", "turn": 1},
        ]
        ctx = _make_ctx(messages)
        prompt = strategy._build_prompt(ctx)

        pos_turn0 = prompt.index("Turn 0:")
        pos_turn1 = prompt.index("Turn 1:")
        assert pos_turn0 < pos_turn1

    def test_turn_1_messages_before_turn_2(self):
        """Turn 1 messages appear before Turn 2 messages."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "a", "turn": 1},
            {"id": "m2", "role": "user", "content": "b", "turn": 2},
        ]
        ctx = _make_ctx(messages)
        prompt = strategy._build_prompt(ctx)

        pos_t1 = prompt.index("Turn 1:")
        pos_t2 = prompt.index("Turn 2:")
        assert pos_t1 < pos_t2

    def test_messages_within_turn_preserve_order(self):
        """m1 before m2 within same turn."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "first in turn", "turn": 0},
            {"id": "m2", "role": "assistant", "content": "second in turn", "turn": 0},
        ]
        ctx = _make_ctx(messages)
        prompt = strategy._build_prompt(ctx)

        pos_m1 = prompt.index("[m1]")
        pos_m2 = prompt.index("[m2]")
        assert pos_m1 < pos_m2

    def test_turn_with_tool_result_included(self):
        """Tool result messages are included in the turn block."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "run tool", "turn": 0},
            {"id": "m2", "role": "assistant", "content": "calling tool", "turn": 0},
            {"id": "m3", "role": "user", "content": "result here", "turn": 0},
        ]
        ctx = _make_ctx(messages)
        prompt = strategy._build_prompt(ctx)

        # All 3 should appear under Turn 0
        pos_turn0 = prompt.index("Turn 0:")
        try:
            pos_turn1 = prompt.index("Turn 1:")
        except ValueError:
            pos_turn1 = len(prompt)

        turn0_block = prompt[pos_turn0:pos_turn1]
        assert "[m1]" in turn0_block
        assert "[m2]" in turn0_block
        assert "[m3]" in turn0_block

    def test_message_objects_with_turn_attribute(self):
        """Message pydantic objects (not dicts) are also grouped by turn."""
        strategy = _make_strategy()
        messages = [
            Message(id="m1", role="user", content="hello", turn=0),
            Message(id="m2", role="assistant", content="world", turn=0),
            Message(id="m3", role="user", content="again", turn=1),
        ]
        ctx = _make_ctx(messages)
        prompt = strategy._build_prompt(ctx)

        assert "Turn 0:" in prompt
        assert "Turn 1:" in prompt
        assert "[m1]" in prompt
        assert "[m3]" in prompt


# ── turn=None goes to Turn (unknown): ─────────────────────────────────────────

class TestTurnNoneHandling:

    def test_none_turn_messages_under_unknown_label(self):
        """Messages with turn=None → grouped under Turn (unknown):."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "legacy msg", "turn": None},
            {"id": "m2", "role": "user", "content": "normal msg", "turn": 0},
        ]
        ctx = _make_ctx(messages)
        prompt = strategy._build_prompt(ctx)

        assert "Turn (unknown):" in prompt
        assert "Turn 0:" in prompt

    def test_none_turn_group_appears_before_int_turns(self):
        """Unknown group is rendered before int-turn groups."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "legacy", "turn": None},
            {"id": "m2", "role": "user", "content": "normal", "turn": 0},
        ]
        ctx = _make_ctx(messages)
        prompt = strategy._build_prompt(ctx)

        pos_unknown = prompt.index("Turn (unknown):")
        pos_turn0 = prompt.index("Turn 0:")
        assert pos_unknown < pos_turn0

    def test_none_turn_message_appears_under_unknown_block(self):
        """m1 with turn=None appears in the Turn (unknown) block."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "legacy msg", "turn": None},
            {"id": "m2", "role": "user", "content": "normal msg", "turn": 0},
        ]
        ctx = _make_ctx(messages)
        prompt = strategy._build_prompt(ctx)

        pos_unknown = prompt.index("Turn (unknown):")
        pos_turn0 = prompt.index("Turn 0:")
        unknown_block = prompt[pos_unknown:pos_turn0]
        assert "[m1]" in unknown_block

    def test_only_none_turn_messages(self):
        """All messages with turn=None → only Turn (unknown): group, no int groups."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "msg a", "turn": None},
            {"id": "m2", "role": "assistant", "content": "msg b", "turn": None},
        ]
        ctx = _make_ctx(messages)
        prompt = strategy._build_prompt(ctx)

        assert "Turn (unknown):" in prompt
        assert "Turn 0:" not in prompt


# ── msg_id_range_hint ─────────────────────────────────────────────────────────

class TestMsgIdRangeHint:

    def test_range_hint_present_in_prompt(self):
        """msg_id_range_hint appears in the prompt."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "a", "turn": 0},
            {"id": "m5", "role": "user", "content": "b", "turn": 1},
        ]
        ctx = _make_ctx(messages)
        prompt = strategy._build_prompt(ctx)

        assert "Available msg_id range:" in prompt

    def test_range_hint_correct_first_last(self):
        """Range hint shows m1..m5 for ids m1, m2, m3, m4, m5."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "a", "turn": 0},
            {"id": "m3", "role": "assistant", "content": "b", "turn": 0},
            {"id": "m5", "role": "user", "content": "c", "turn": 1},
        ]
        ctx = _make_ctx(messages)
        messages_text, hint = strategy._format_messages_by_turn(messages)

        assert "m1..m5" in hint

    def test_range_hint_message_count(self):
        """Range hint includes correct N (total message count)."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "a", "turn": 0},
            {"id": "m2", "role": "assistant", "content": "b", "turn": 0},
            {"id": "m3", "role": "user", "content": "c", "turn": 1},
        ]
        ctx = _make_ctx(messages)
        _, hint = strategy._format_messages_by_turn(messages)

        assert "3 messages" in hint

    def test_range_hint_user_turn_count(self):
        """Range hint includes correct K (distinct user-turn count)."""
        strategy = _make_strategy()
        messages = [
            {"id": "m1", "role": "user", "content": "a", "turn": 0},
            {"id": "m2", "role": "assistant", "content": "b", "turn": 0},
            {"id": "m3", "role": "user", "content": "c", "turn": 1},
            {"id": "m4", "role": "assistant", "content": "d", "turn": 1},
            {"id": "m5", "role": "user", "content": "e", "turn": 2},
        ]
        _, hint = strategy._format_messages_by_turn(messages)

        assert "3 user turns" in hint

    def test_range_hint_fabrication_warning(self):
        """Range hint contains fabrication warning."""
        strategy = _make_strategy()
        messages = [{"id": "m1", "role": "user", "content": "a", "turn": 0}]
        _, hint = strategy._format_messages_by_turn(messages)

        assert "Fabricating ids" in hint

    def test_empty_messages_no_hint(self):
        """Empty messages list → empty hint."""
        strategy = _make_strategy()
        _, hint = strategy._format_messages_by_turn([])
        assert hint == ""

    def test_non_mN_format_ids_use_first_last(self):
        """Non-mN ids fall back to first/last in list order."""
        strategy = _make_strategy()
        messages = [
            {"id": "msg_a", "role": "user", "content": "a", "turn": 0},
            {"id": "msg_z", "role": "assistant", "content": "z", "turn": 0},
        ]
        _, hint = strategy._format_messages_by_turn(messages)

        assert "msg_a..msg_z" in hint
