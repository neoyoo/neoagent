# tests/v2/strategies/test_oneshot_memory_review.py
"""TDD tests for OneShotMemoryReviewStrategy.

Spec refs:
  § 11.2  (lines 1884-1896) — MemoryReviewStrategy ABC
  § 11.3  (lines 1899-1964) — OneShotMemoryReviewStrategy implementation

Coverage:
  A. Happy path (valid list, empty list, multi-entry)         — tests 1-3
  B. Invalid individual entries are skipped                   — tests 4-6
  C. Degradation paths (retry + exhaust → empty list)         — tests 7-9
  D. Prompt construction                                      — tests 10-13
  E. MemoryEntry auto-fill                                    — tests 14-16
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neoagent.v2.schema import MemoryEntry, WorkingMemory
from neoagent.v2.strategies.oneshot_memory_review import OneShotMemoryReviewStrategy


# ── fixtures & helpers ────────────────────────────────────────────────────────

def _make_wm() -> WorkingMemory:
    return WorkingMemory(
        session_id="s-test",
        version=1,
        at_turn=5,
        constraints_and_preferences=["c01: budget conscious", "c02: prefers trains"],
        progress="discussed Kyoto and Tokyo",
        key_decisions=["d01: 10-day itinerary"],
        relevant_files=[],
        next_steps=["n01: book flights"],
        critical_context="spring 2025, cherry blossom season",
        updated_by="llm_tool",
        updated_at=datetime(2026, 4, 22, 10, 0, 0),
    )


def _make_messages() -> list[dict]:
    return [
        {"id": "m1", "role": "user", "content": "I prefer window seats on planes"},
        {"id": "m2", "role": "assistant", "content": "Noted, will keep that in mind"},
        {"id": "m3", "role": "user", "content": "I always stay in budget hotels"},
    ]


def _make_strategy() -> OneShotMemoryReviewStrategy:
    return OneShotMemoryReviewStrategy(api_key="test-key", max_retries=3)


# ── base_url support (Anthropic-protocol-compatible providers) ───────────────

class TestBaseUrlSupport:
    """Strategy must accept base_url for providers like DashScope-Anthropic."""

    def test_default_base_url_is_none(self) -> None:
        strategy = OneShotMemoryReviewStrategy(api_key="test-key")
        assert strategy.base_url is None

    def test_custom_base_url_stored(self) -> None:
        url = "https://dashscope.aliyuncs.com/api/v2/apps/anthropic"
        strategy = OneShotMemoryReviewStrategy(api_key="test-key", base_url=url)
        assert strategy.base_url == url
        assert strategy._client is not None


def _mock_llm_text(payload: Any) -> MagicMock:
    """Build an AsyncAnthropic-style mock that returns the given payload as JSON text."""
    text = json.dumps(payload, ensure_ascii=False)
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=text)]
    return mock_response


def _valid_entry(**overrides) -> dict:
    base = {
        "type": "preference",
        "category": "travel",
        "content": "Prefers window seats on planes",
        "confidence": 0.85,
        "evidence": {"msg_ids": ["m1"]},
    }
    base.update(overrides)
    return base


# ── A. Happy path ─────────────────────────────────────────────────────────────

class TestHappyPath:
    """Tests 1-3: Valid LLM responses produce correct MemoryEntry lists."""

    async def test_01_single_valid_entry_returns_one_memory_entry(self):
        """mock LLM returns [valid_entry] → review returns list[MemoryEntry] with 1 item."""
        strategy = _make_strategy()
        payload = [_valid_entry()]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_text(payload)),
        ):
            result = await strategy.review(
                session_id="s1",
                user_id="u1",
                messages=_make_messages(),
                wm=_make_wm(),
            )

        assert len(result) == 1
        assert isinstance(result[0], MemoryEntry)
        assert result[0].type == "preference"
        assert result[0].content == "Prefers window seats on planes"
        assert result[0].confidence == 0.85

    async def test_02_empty_list_returns_empty(self):
        """mock LLM returns [] → review returns []."""
        strategy = _make_strategy()

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_text([])),
        ):
            result = await strategy.review(
                session_id="s1",
                user_id="u1",
                messages=_make_messages(),
                wm=_make_wm(),
            )

        assert result == []

    async def test_03_multiple_valid_entries_returns_all(self):
        """mock LLM returns 3 valid entries → review returns 3 MemoryEntry items."""
        strategy = _make_strategy()
        payload = [
            _valid_entry(type="preference", content="Prefers window seats"),
            _valid_entry(type="fact", content="Has budget of $3000", category="finance"),
            _valid_entry(type="goal", content="Wants to see cherry blossoms", category="travel"),
        ]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_text(payload)),
        ):
            result = await strategy.review(
                session_id="s1",
                user_id="u1",
                messages=_make_messages(),
                wm=_make_wm(),
            )

        assert len(result) == 3
        types = [e.type for e in result]
        assert "preference" in types
        assert "fact" in types
        assert "goal" in types


# ── B. Invalid individual entries skipped ─────────────────────────────────────

class TestInvalidEntriesSkipped:
    """Tests 4-6: Invalid entries are skipped individually, valid ones preserved."""

    async def test_04_unknown_type_skipped_valid_kept(self, caplog):
        """type='unknown' skipped, type='fact' kept → returns 1 entry + warning log."""
        strategy = _make_strategy()
        payload = [
            {"type": "unknown_type", "content": "some content", "confidence": 0.5},
            _valid_entry(type="fact", content="Valid fact entry"),
        ]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_text(payload)),
        ):
            with caplog.at_level(logging.WARNING):
                result = await strategy.review(
                    session_id="s1",
                    user_id="u1",
                    messages=_make_messages(),
                    wm=_make_wm(),
                )

        assert len(result) == 1
        assert result[0].type == "fact"
        # Warning should have been logged for the invalid entry
        assert any("unknown_type" in record.message or "type" in record.message.lower()
                   for record in caplog.records)

    async def test_05_missing_content_skipped(self):
        """content=None → entry skipped, review returns []."""
        strategy = _make_strategy()
        payload = [
            {"type": "fact", "content": None, "confidence": 0.7},
        ]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_text(payload)),
        ):
            result = await strategy.review(
                session_id="s1",
                user_id="u1",
                messages=_make_messages(),
                wm=_make_wm(),
            )

        assert result == []

    async def test_06_confidence_out_of_range_skipped(self, caplog):
        """confidence=1.5 (>1.0) → entry skipped, review returns []."""
        strategy = _make_strategy()
        payload = [
            {"type": "fact", "content": "Some fact", "confidence": 1.5},
        ]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_text(payload)),
        ):
            with caplog.at_level(logging.WARNING):
                result = await strategy.review(
                    session_id="s1",
                    user_id="u1",
                    messages=_make_messages(),
                    wm=_make_wm(),
                )

        assert result == []
        assert any("confidence" in record.message.lower() for record in caplog.records)


# ── C. Degradation paths ──────────────────────────────────────────────────────

class TestDegradation:
    """Tests 7-9: Retry on bad JSON/non-list, exhaust returns [] without raising."""

    async def test_07_json_parse_failure_retries_succeeds_on_third(self):
        """First 2 calls return invalid JSON, 3rd returns valid list → success."""
        strategy = _make_strategy()
        valid_payload = [_valid_entry()]

        bad_response = MagicMock()
        bad_response.content = [MagicMock(text="not valid json {{{{")]

        good_response = _mock_llm_text(valid_payload)

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return bad_response
            return good_response

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(side_effect=side_effect),
        ):
            result = await strategy.review(
                session_id="s1",
                user_id="u1",
                messages=_make_messages(),
                wm=_make_wm(),
            )

        assert call_count == 3
        assert len(result) == 1
        assert isinstance(result[0], MemoryEntry)

    async def test_08_all_retries_exhausted_returns_empty_list_no_raise(self, caplog):
        """All 3 attempts return invalid JSON → review returns [] and does NOT raise."""
        strategy = _make_strategy()

        bad_response = MagicMock()
        bad_response.content = [MagicMock(text="completely invalid json ###")]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=bad_response),
        ):
            with caplog.at_level(logging.ERROR):
                # Must NOT raise — soft failure for memory review
                result = await strategy.review(
                    session_id="s1",
                    user_id="u1",
                    messages=_make_messages(),
                    wm=_make_wm(),
                )

        assert result == []
        # Error should have been logged
        assert any("exhausted" in record.message.lower() or "retries" in record.message.lower()
                   for record in caplog.records)

    async def test_09_top_level_non_list_triggers_retry_then_empty(self):
        """LLM returns a dict (not list) → treated as failure, retry, exhaust → []."""
        strategy = _make_strategy()

        dict_response = MagicMock()
        dict_response.content = [MagicMock(text=json.dumps({"error": "bad output"}))]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=dict_response),
        ):
            # Must NOT raise
            result = await strategy.review(
                session_id="s1",
                user_id="u1",
                messages=_make_messages(),
                wm=_make_wm(),
            )

        assert result == []


# ── D. Prompt construction ────────────────────────────────────────────────────

class TestPromptConstruction:
    """Tests 10-13: _build_prompt content verification."""

    def test_10_prompt_contains_user_id(self):
        """_build_prompt must include the user_id."""
        strategy = _make_strategy()
        prompt = strategy._build_prompt("u-specific-123", _make_messages(), _make_wm())

        assert "u-specific-123" in prompt

    def test_11_prompt_contains_wm_json(self):
        """_build_prompt must embed serialized WorkingMemory JSON."""
        strategy = _make_strategy()
        wm = _make_wm()
        prompt = strategy._build_prompt("u1", _make_messages(), wm)

        # WM progress should appear in serialized form
        assert "discussed Kyoto and Tokyo" in prompt
        # WM should be in JSON format (at minimum a { appears)
        assert "{" in prompt

    def test_12_prompt_contains_schema_keywords(self):
        """_build_prompt must include JSON schema example with key memory type words."""
        strategy = _make_strategy()
        prompt = strategy._build_prompt("u1", _make_messages(), _make_wm())

        # All valid types should be mentioned
        assert "preference" in prompt
        assert "fact" in prompt
        assert "confidence" in prompt

    def test_13_prompt_permits_empty_array(self):
        """_build_prompt must explicitly allow returning empty array []."""
        strategy = _make_strategy()
        prompt = strategy._build_prompt("u1", _make_messages(), _make_wm())

        # Some indication that returning [] is acceptable
        assert "[]" in prompt


# ── E. MemoryEntry auto-fill ──────────────────────────────────────────────────

class TestMemoryEntryAutoFill:
    """Tests 14-16: Auto-filled fields on constructed MemoryEntry objects."""

    async def test_14_entries_have_correct_user_id(self):
        """All returned entries must have user_id matching the passed user_id."""
        strategy = _make_strategy()
        payload = [
            _valid_entry(type="preference", content="Likes window seats"),
            _valid_entry(type="fact", content="Travels in spring"),
        ]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_text(payload)),
        ):
            result = await strategy.review(
                session_id="s1",
                user_id="user-xyz-42",
                messages=_make_messages(),
                wm=_make_wm(),
            )

        assert len(result) == 2
        for entry in result:
            assert entry.user_id == "user-xyz-42"

    async def test_15_entries_have_unique_nonempty_memory_ids(self):
        """Each returned entry must have a unique, non-empty memory_id (12-char hex)."""
        strategy = _make_strategy()
        payload = [
            _valid_entry(type="preference", content="First preference"),
            _valid_entry(type="fact", content="Second fact"),
            _valid_entry(type="goal", content="Third goal"),
        ]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_text(payload)),
        ):
            result = await strategy.review(
                session_id="s1",
                user_id="u1",
                messages=_make_messages(),
                wm=_make_wm(),
            )

        assert len(result) == 3
        memory_ids = [e.memory_id for e in result]

        # All non-empty
        for mid in memory_ids:
            assert mid and len(mid) > 0

        # All unique
        assert len(set(memory_ids)) == 3

        # Each should be 12 hex chars
        for mid in memory_ids:
            assert len(mid) == 12
            assert all(c in "0123456789abcdef" for c in mid)

    async def test_16_entries_have_created_at_datetime(self):
        """Each returned entry must have a non-None created_at datetime."""
        strategy = _make_strategy()
        payload = [_valid_entry()]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_text(payload)),
        ):
            result = await strategy.review(
                session_id="s1",
                user_id="u1",
                messages=_make_messages(),
                wm=_make_wm(),
            )

        assert len(result) == 1
        assert isinstance(result[0].created_at, datetime)
        assert result[0].created_at is not None
