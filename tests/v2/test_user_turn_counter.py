# tests/v2/test_user_turn_counter.py
"""B2a — user_turn_counter on SessionState + agent.chat() stamping.

Tests cover:
  A. SessionState.user_turn_counter field default + serialization
  B. agent.chat() increments counter each call
  C. Messages in chat #2 all have Message.turn=2
  D. turns_since_last_compression incremented per-chat; reset on compression
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from neoagent.core.types import Message, TextBlock
from neoagent.session import (
    Session,
    SessionState,
    _session_from_dict,
    _session_to_dict,
)


# ── A. SessionState.user_turn_counter field ───────────────────────────────────

class TestUserTurnCounterField:
    def test_default_is_zero(self):
        """SessionState.user_turn_counter must default to 0."""
        state = SessionState()
        assert state.user_turn_counter == 0

    def test_can_set_value(self):
        state = SessionState()
        state.user_turn_counter = 3
        assert state.user_turn_counter == 3

    def test_roundtrip_to_dict_from_dict(self):
        """user_turn_counter survives _session_to_dict / _session_from_dict."""
        session = Session.create(session_id="utc-rt")
        session.state.user_turn_counter = 7
        d = _session_to_dict(session)
        recovered = _session_from_dict(d)
        assert recovered.state.user_turn_counter == 7

    def test_backward_compat_old_dict_missing_field(self):
        """Old session dict without user_turn_counter deserializes to 0."""
        old_dict = {
            "id": "old-utc",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "state": {
                "previous_summary": None,
                "compression_failures": 0,
                "memory_tool_calls": 0,
                "memory_token_baseline": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "promoted_tools": [],
                "freed_tool_results": {},
                "tool_use_to_tool_name": {},
            },
            "messages": [],
        }
        session = _session_from_dict(old_dict)
        assert session.state.user_turn_counter == 0

    def test_roundtrip_zero_preserved(self):
        """user_turn_counter=0 stays 0 after roundtrip."""
        session = Session.create(session_id="utc-zero")
        d = _session_to_dict(session)
        recovered = _session_from_dict(d)
        assert recovered.state.user_turn_counter == 0


# ── B-D. agent.chat() increments counter + stamps Messages ───────────────────

def _make_mock_loop(reply_text: str = "ok"):
    """Build a minimal QueryLoop mock that appends an assistant message to session."""
    from neoagent.core.types import ConversationResult, Turn

    async def _run(session=None, max_turns=None, user_turn=None):
        # append an assistant message so chat() has something to return
        session.messages.append(
            Message(
                id=session.state.id_gen.next_msg_id(),
                turn=user_turn,  # stamp with the passed user_turn
                role="assistant",
                content=reply_text,
            )
        )
        turn = Turn(
            response=session.messages[-1],
            tool_calls=[],
            tool_results=[],
            stop_reason="end_turn",
        )
        return ConversationResult(turns=[turn], reason="completed")

    loop = MagicMock()
    loop.run = AsyncMock(side_effect=_run)
    return loop


def _make_agent_with_mock_loop(reply_text: str = "ok"):
    """Create NeoAgent with a mocked QueryLoop."""
    from neoagent.config import NeoAgentConfig
    from neoagent.agent import NeoAgent

    config = NeoAgentConfig(api_key="fake-key", model="claude-sonnet-4-6")
    agent = NeoAgent(config)
    agent._loop = _make_mock_loop(reply_text)
    return agent


class TestAgentChatUserTurnCounter:
    @pytest.mark.asyncio
    async def test_counter_increments_each_chat(self):
        """After each agent.chat() call, user_turn_counter increments by 1."""
        agent = _make_agent_with_mock_loop()
        session = agent.new_session()

        await agent.chat("msg1", session=session)
        assert session.state.user_turn_counter == 1

        await agent.chat("msg2", session=session)
        assert session.state.user_turn_counter == 2

        await agent.chat("msg3", session=session)
        assert session.state.user_turn_counter == 3

    @pytest.mark.asyncio
    async def test_user_message_gets_correct_turn(self):
        """User message appended in chat() gets turn = user_turn_counter BEFORE increment."""
        agent = _make_agent_with_mock_loop()
        session = agent.new_session()

        # First chat: counter=0 before increment → user msg turn=1 (after increment)
        await agent.chat("first", session=session)
        user_msgs = [m for m in session.messages if m.role == "user" and isinstance(m.content, str)]
        assert len(user_msgs) >= 1
        # The user message should have turn=1 (the value after increment)
        assert user_msgs[0].turn == 1

    @pytest.mark.asyncio
    async def test_counter_starts_at_zero_for_new_session(self):
        """Fresh session starts with user_turn_counter=0."""
        agent = _make_agent_with_mock_loop()
        session = agent.new_session()
        assert session.state.user_turn_counter == 0

    @pytest.mark.asyncio
    async def test_turns_since_last_compression_increments_per_chat(self):
        """turns_since_last_compression increments by 1 per successful chat() call."""
        agent = _make_agent_with_mock_loop()
        session = agent.new_session()

        await agent.chat("a", session=session)
        assert session.state.turns_since_last_compression == 1

        await agent.chat("b", session=session)
        assert session.state.turns_since_last_compression == 2
