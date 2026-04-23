# tests/v2/strategies/test_oneshot_compression.py
"""TDD tests for OneShotCompressionStrategy.

Spec refs:
  § 10.2  (lines 1589-1635) — CompressionStrategy ABC + OneShot signature
  § 10.3a (lines 1710-1766) — Compressor Output Contract (8 hard rules)
  § 10.4  (lines 1770-1800) — 7-section structured template format

Coverage:
  A. Rule validation (mock LLM return) — tests 1-7
  B. Degrade behaviour — tests 8-9
  C. JSON parse failure paths — tests 10-11
  D. Prompt construction — tests 12-14
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neoagent.v2.errors import CompressionError
from neoagent.v2.schema import BatchMember, CompressionContext, CompressionDelta
from neoagent.v2.strategies.oneshot_compression import OneShotCompressionStrategy


# ── fixtures & helpers ────────────────────────────────────────────────────────

def _make_context(
    messages: list[dict] | None = None,
    previous_wm: dict | None = None,
) -> CompressionContext:
    """Create a minimal CompressionContext for tests."""
    if messages is None:
        messages = [
            {"id": "m1", "role": "user", "content": "Hello"},
            {"id": "m2", "role": "assistant", "content": "World"},
            {"id": "m3", "role": "tool", "content": "done"},
        ]
    if previous_wm is None:
        previous_wm = {
            "goal": "build the login system",
            "constraints_and_preferences": ["c01: use JWT"],
            "progress": "started",
            "key_decisions": ["d01: use FastAPI"],
            "relevant_files": ["f01: auth.py"],
            "next_steps": ["n01: write tests"],
            "critical_context": "must be HTTPS in prod",
        }
    return CompressionContext(
        session_id="s1",
        messages=messages,
        previous_batches=[],
        previous_wm=previous_wm,
        trigger="token_threshold",
    )


def _make_strategy() -> OneShotCompressionStrategy:
    return OneShotCompressionStrategy(api_key="test-key", max_retries=3)


# ── base_url support (Anthropic-protocol-compatible providers) ───────────────

class TestBaseUrlSupport:
    """Strategy must accept base_url for providers like DashScope-Anthropic."""

    def test_default_base_url_is_none(self) -> None:
        strategy = OneShotCompressionStrategy(api_key="test-key")
        assert strategy.base_url is None

    def test_custom_base_url_stored(self) -> None:
        url = "https://dashscope.aliyuncs.com/api/v2/apps/anthropic"
        strategy = OneShotCompressionStrategy(api_key="test-key", base_url=url)
        assert strategy.base_url == url
        # Client constructed without error (AsyncAnthropic accepts base_url kwarg)
        assert strategy._client is not None


def _mock_llm_response(json_payload: Any) -> MagicMock:
    """Build a mock that _call_llm will resolve to the given JSON string."""
    text = json.dumps(json_payload, ensure_ascii=False)
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=text)]
    return mock_response


def _valid_delta_json() -> dict:
    """A fully compliant CompressionDelta JSON payload."""
    return {
        "batch_summary": {
            "GOAL": "build the login system",
            "CONSTRAINTS_AND_PREFERENCES": ["c01: use JWT"],
            "PROGRESS": "login done",
            "KEY_DECISIONS": ["d01: use FastAPI", "d02: 24h token"],
            "RELEVANT_FILES": ["f01: auth.py"],
            "NEXT_STEPS": ["n01: write tests"],
            "CRITICAL_CONTEXT": "HTTPS in prod",
        },
        "batch_members": [
            {"id": "m1", "role": "user", "preview": "Hello greeting message"},
            {"id": "m2", "role": "assistant", "preview": "World response back"},
            {"id": "m3", "role": "tool", "preview": "Tool execution result done"},
        ],
        "working_memory_delta": [
            {"field": "key_decisions", "op": "append", "value": "d02: 24h token expiry"},
            {"field": "progress", "op": "set", "value": "login endpoint complete"},
        ],
    }


# ── A. Rule validation ────────────────────────────────────────────────────────

class TestRuleValidation:
    """Tests 1-7: Each contract rule violation causes wm_delta to be dropped."""

    async def test_01_valid_delta_round_trip(self):
        """A fully valid payload returns complete CompressionDelta with all fields."""
        strategy = _make_strategy()
        payload = _valid_delta_json()

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_response(payload)),
        ):
            result = await strategy.compress(_make_context())

        assert isinstance(result, CompressionDelta)
        assert result.batch_summary == payload["batch_summary"]
        assert len(result.batch_members) == 3
        assert result.batch_members[0] == BatchMember(
            id="m1", role="user", preview="Hello greeting message"
        )
        assert len(result.working_memory_delta) == 2
        assert result.working_memory_delta[0]["field"] == "key_decisions"
        assert result.working_memory_delta[0]["op"] == "append"

    async def test_02_rule1_violation_unknown_id_drops_wm_delta(self):
        """Rule 1: batch_member id 'm99' not in context → wm_delta dropped."""
        strategy = _make_strategy()
        payload = _valid_delta_json()
        payload["batch_members"] = [{"id": "m99", "role": "user", "preview": "fake"}]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_response(payload)),
        ):
            result = await strategy.compress(_make_context())

        assert result.working_memory_delta == []
        assert result.batch_summary == payload["batch_summary"]

    async def test_03_rule2_violation_invalid_role_drops_wm_delta(self):
        """Rule 2: role='admin' not in valid set → wm_delta dropped."""
        strategy = _make_strategy()
        payload = _valid_delta_json()
        payload["batch_members"] = [{"id": "m1", "role": "admin", "preview": "x"}]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_response(payload)),
        ):
            result = await strategy.compress(_make_context())

        assert result.working_memory_delta == []
        assert result.batch_summary == payload["batch_summary"]

    async def test_04_rule3_violation_scalar_field_with_append_drops_wm_delta(self):
        """Rule 3: progress (scalar) with op='append' is forbidden → wm_delta dropped.
        Also touches rule 5 if goal is used; here we use progress for pure rule 3 test.
        """
        strategy = _make_strategy()
        payload = _valid_delta_json()
        # progress is a scalar field, only 'set' is allowed
        payload["working_memory_delta"] = [
            {"field": "progress", "op": "append", "value": "extra info"},
        ]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_response(payload)),
        ):
            result = await strategy.compress(_make_context())

        assert result.working_memory_delta == []
        assert result.batch_summary == payload["batch_summary"]
        assert len(result.batch_members) == 3  # batch_members preserved

    async def test_05_rule4a_violation_list_value_missing_prefix_drops_wm_delta(self):
        """Rule 4a: key_decisions value without prefix pattern → wm_delta dropped."""
        strategy = _make_strategy()
        payload = _valid_delta_json()
        payload["working_memory_delta"] = [
            {"field": "key_decisions", "op": "append", "value": "no prefix here"},
        ]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_response(payload)),
        ):
            result = await strategy.compress(_make_context())

        assert result.working_memory_delta == []
        assert result.batch_summary == payload["batch_summary"]

    async def test_06_rule4b_violation_remove_without_item_id_drops_wm_delta(self):
        """Rule 4b: remove op on list field without item_id → wm_delta dropped."""
        strategy = _make_strategy()
        payload = _valid_delta_json()
        payload["working_memory_delta"] = [
            {"field": "key_decisions", "op": "remove", "value": None},
            # No item_id provided
        ]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_response(payload)),
        ):
            result = await strategy.compress(_make_context())

        assert result.working_memory_delta == []
        assert result.batch_summary == payload["batch_summary"]

    async def test_07_rule5_violation_goal_in_delta_drops_wm_delta(self):
        """Rule 5: field='goal' in working_memory_delta is forbidden → wm_delta dropped."""
        strategy = _make_strategy()
        payload = _valid_delta_json()
        payload["working_memory_delta"] = [
            {"field": "goal", "op": "set", "value": "new goal — should be rejected"},
        ]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_response(payload)),
        ):
            result = await strategy.compress(_make_context())

        assert result.working_memory_delta == []
        assert result.batch_summary == payload["batch_summary"]


# ── B. Degrade behaviour ──────────────────────────────────────────────────────

class TestDegradeBehaviour:
    """Tests 8-9: Verify degrade strategy (a) semantics."""

    async def test_08_hard_violation_does_not_raise(self):
        """Any rule 1-5 violation must not raise — returns CompressionDelta."""
        strategy = _make_strategy()
        payload = _valid_delta_json()
        # Rule 1 violation
        payload["batch_members"] = [{"id": "m99", "role": "user", "preview": "x"}]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_response(payload)),
        ):
            result = await strategy.compress(_make_context())

        assert isinstance(result, CompressionDelta)  # no raise

    async def test_09_batch_summary_preserved_on_wm_delta_drop(self):
        """batch_summary must be preserved even when wm_delta is dropped."""
        strategy = _make_strategy()
        payload = _valid_delta_json()
        # Rule 5 violation: goal in delta
        payload["working_memory_delta"] = [
            {"field": "goal", "op": "set", "value": "hijacked goal"},
        ]
        expected_summary = payload["batch_summary"]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_response(payload)),
        ):
            result = await strategy.compress(_make_context())

        assert result.batch_summary == expected_summary
        assert result.working_memory_delta == []


# ── C. JSON parse failure paths ───────────────────────────────────────────────

class TestJsonParseFailure:
    """Tests 10-11: Retry on JSON parse failure and exhaust behaviour."""

    async def test_10_json_parse_fail_retries_and_succeeds_on_third(self):
        """First 2 calls return invalid JSON, 3rd returns valid — final success."""
        strategy = _make_strategy()
        payload = _valid_delta_json()

        bad_response = MagicMock()
        bad_response.content = [MagicMock(text="not valid json at all {{{{")]

        good_response = _mock_llm_response(payload)

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
            result = await strategy.compress(_make_context())

        assert call_count == 3
        assert isinstance(result, CompressionDelta)
        assert len(result.working_memory_delta) == 2

    async def test_11_retry_exhausted_raises_compression_error(self):
        """All 3 attempts return invalid JSON → CompressionError raised."""
        strategy = _make_strategy()

        bad_response = MagicMock()
        bad_response.content = [MagicMock(text="definitely not json")]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=bad_response),
        ):
            with pytest.raises(CompressionError) as exc_info:
                await strategy.compress(_make_context())

        assert "3" in str(exc_info.value) or "Exhausted" in str(exc_info.value)


# ── D. Prompt construction ────────────────────────────────────────────────────

class TestPromptConstruction:
    """Tests 12-14: Verify _build_prompt content."""

    def test_12_prompt_contains_all_8_rule_keywords(self):
        """_build_prompt output must reference all 8 hard contract rules."""
        strategy = _make_strategy()
        ctx = _make_context()
        prompt = strategy._build_prompt(ctx)

        # Rule 1: batch_members[].id must exist
        assert "batch_members[].id" in prompt or "batch_member" in prompt.lower()

        # Rule 2: role ∈ {user, assistant, tool}
        assert "role ∈" in prompt or 'role' in prompt

        # Rule 3: list fields vs scalar fields
        assert "append" in prompt or "list 段" in prompt

        # Rule 4: prefix + remove item_id
        assert "remove" in prompt and "item_id" in prompt
        assert "前缀" in prompt or "^(c|d|f|n)" in prompt or "list 段 value" in prompt

        # Rule 5: goal immutable
        assert "goal immutable" in prompt or "goal 是不可变" in prompt or "不可变" in prompt

        # Rule 8: preview quality
        assert "preview" in prompt

    def test_13_prompt_contains_session_goal(self):
        """_build_prompt must include the session_goal explicitly."""
        strategy = _make_strategy()
        ctx = _make_context(previous_wm={"goal": "my specific unique session goal"})
        prompt = strategy._build_prompt(ctx)

        assert "my specific unique session goal" in prompt

    def test_14_prompt_contains_json_schema_top_level_keys(self):
        """_build_prompt must include all 3 top-level C4 JSON schema keys."""
        strategy = _make_strategy()
        ctx = _make_context()
        prompt = strategy._build_prompt(ctx)

        assert '"batch_summary"' in prompt
        assert '"batch_members"' in prompt
        assert '"working_memory_delta"' in prompt


# ── Additional edge cases ─────────────────────────────────────────────────────

class TestEdgeCases:
    """Additional contract edge cases."""

    async def test_rule1_skipped_when_context_has_no_messages(self):
        """Rule 1 validation is skipped when ctx.messages is empty (no id set to check against)."""
        strategy = _make_strategy()
        payload = _valid_delta_json()
        # msg ids in payload can be anything when messages is empty
        payload["batch_members"] = [{"id": "any_id", "role": "user", "preview": "x"}]

        ctx = _make_context(messages=[])

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_response(payload)),
        ):
            result = await strategy.compress(ctx)

        # When no messages in context, existing_ids is empty → rule 1 passes
        assert isinstance(result, CompressionDelta)
        assert len(result.batch_members) == 1

    async def test_rule4_valid_remove_with_item_id_passes(self):
        """A remove op with item_id is valid."""
        strategy = _make_strategy()
        payload = _valid_delta_json()
        payload["working_memory_delta"] = [
            {"field": "key_decisions", "op": "remove", "item_id": "d01"},
        ]

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_response(payload)),
        ):
            result = await strategy.compress(_make_context())

        assert len(result.working_memory_delta) == 1
        assert result.working_memory_delta[0]["op"] == "remove"

    async def test_all_three_roles_are_valid(self):
        """All three roles (user/assistant/tool) pass rule 2."""
        strategy = _make_strategy()
        payload = _valid_delta_json()
        payload["batch_members"] = [
            {"id": "m1", "role": "user", "preview": "user msg"},
            {"id": "m2", "role": "assistant", "preview": "assistant msg"},
            {"id": "m3", "role": "tool", "preview": "tool result"},
        ]
        payload["working_memory_delta"] = []

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(return_value=_mock_llm_response(payload)),
        ):
            result = await strategy.compress(_make_context())

        assert len(result.batch_members) == 3
        roles = [m.role for m in result.batch_members]
        assert roles == ["user", "assistant", "tool"]

    async def test_missing_top_level_key_triggers_retry(self):
        """A response missing 'batch_members' top-level key triggers retry."""
        strategy = _make_strategy()

        incomplete = {"batch_summary": {}, "working_memory_delta": []}
        # Missing batch_members
        bad_response = MagicMock()
        bad_response.content = [
            MagicMock(text=json.dumps(incomplete))
        ]

        valid_payload = _valid_delta_json()
        good_response = _mock_llm_response(valid_payload)

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad_response
            return good_response

        with patch.object(
            strategy._client.messages, "create",
            new=AsyncMock(side_effect=side_effect),
        ):
            result = await strategy.compress(_make_context())

        assert call_count == 2
        assert isinstance(result, CompressionDelta)
