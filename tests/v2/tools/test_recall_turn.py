# tests/v2/tools/test_recall_turn.py
"""Unit tests for RecallTurnTool — new session.messages-based interface.

Spec refs: § 14.2, § 15.4
Contract refs: C6

RecallTurnTool now looks up Messages by id from session.messages (not from
BatchMember.preview). Content is returned verbatim (str or list of block dicts).
Missing ids go into 'missing' field, not is_error.
"""
from __future__ import annotations

import json
import pytest

from neoagent.core.types import Message, TextBlock, ToolUseBlock, ToolResultBlock


class FakeSession:
    """Minimal session stub exposing .messages list."""
    def __init__(self, messages: list[Message] | None = None):
        self.messages = messages if messages is not None else []


def _make_tool(messages: list[Message] | None = None):
    from neoagent.tools.builtin.recall_turn import RecallTurnTool
    session = FakeSession(messages=messages)
    return RecallTurnTool(session_ref=lambda: session)


# ── 1. single id hit — str content ───────────────────────────────────────────

class TestSingleIdHit:
    @pytest.mark.asyncio
    async def test_single_id_str_content(self):
        msgs = [
            Message(id="m1", role="user", content="hello world"),
            Message(id="m2", role="assistant", content="hi there"),
        ]
        tool = _make_tool(msgs)

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        result = await tool.execute(RecallTurnInput(turn_ids=["m1"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert len(data["recalled"]) == 1
        r = data["recalled"][0]
        assert r["id"] == "m1"
        assert r["role"] == "user"
        assert r["content"] == "hello world"


# ── 2. multiple ids — some hit, some miss ─────────────────────────────────────

class TestPartialMiss:
    @pytest.mark.asyncio
    async def test_hit_and_miss_go_to_respective_fields(self):
        msgs = [Message(id="m1", role="user", content="msg1")]
        tool = _make_tool(msgs)

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        result = await tool.execute(RecallTurnInput(turn_ids=["m1", "m99"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert len(data["recalled"]) == 1
        assert data["recalled"][0]["id"] == "m1"
        assert "m99" in data["missing"]

    @pytest.mark.asyncio
    async def test_multiple_ids_all_hit(self):
        msgs = [
            Message(id="m3", role="user", content="msg3"),
            Message(id="m4", role="assistant", content="msg4"),
        ]
        tool = _make_tool(msgs)

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        result = await tool.execute(RecallTurnInput(turn_ids=["m3", "m4"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert len(data["recalled"]) == 2
        ids = {r["id"] for r in data["recalled"]}
        assert ids == {"m3", "m4"}


# ── 3. all ids missing ────────────────────────────────────────────────────────

class TestAllMissing:
    @pytest.mark.asyncio
    async def test_all_missing_goes_to_missing_field_not_error(self):
        msgs = [Message(id="m1", role="user", content="msg1")]
        tool = _make_tool(msgs)

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        result = await tool.execute(RecallTurnInput(turn_ids=["x1", "x2"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert data["recalled"] == []
        assert "x1" in data["missing"]
        assert "x2" in data["missing"]


# ── 4. empty turn_ids ─────────────────────────────────────────────────────────

class TestEmptyTurnIds:
    @pytest.mark.asyncio
    async def test_empty_turn_ids_returns_empty_recalled(self):
        msgs = [Message(id="m1", role="user", content="msg1")]
        tool = _make_tool(msgs)

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        result = await tool.execute(RecallTurnInput(turn_ids=[]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert data["recalled"] == []

    @pytest.mark.asyncio
    async def test_empty_turn_ids_no_messages_still_ok(self):
        tool = _make_tool([])

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        result = await tool.execute(RecallTurnInput(turn_ids=[]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert data["recalled"] == []


# ── 5. list content — block dicts returned ────────────────────────────────────

class TestListContent:
    @pytest.mark.asyncio
    async def test_text_block_content_serialized_as_dict(self):
        msgs = [
            Message(id="m1", role="assistant", content=[TextBlock(text="hi")]),
        ]
        tool = _make_tool(msgs)

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        result = await tool.execute(RecallTurnInput(turn_ids=["m1"]))

        assert result.is_error is False
        data = json.loads(result.output)
        content = data["recalled"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "hi"

    @pytest.mark.asyncio
    async def test_tool_use_block_serialized(self):
        msgs = [
            Message(id="m2", role="assistant", content=[
                ToolUseBlock(id="tu1", name="search", input={"query": "Bangkok"}),
            ]),
        ]
        tool = _make_tool(msgs)

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        result = await tool.execute(RecallTurnInput(turn_ids=["m2"]))

        assert result.is_error is False
        data = json.loads(result.output)
        content = data["recalled"][0]["content"]
        assert content[0]["type"] == "tool_use"
        assert content[0]["name"] == "search"
        assert content[0]["input"]["query"] == "Bangkok"

    @pytest.mark.asyncio
    async def test_tool_result_block_serialized(self):
        msgs = [
            Message(id="m3", role="user", content=[
                ToolResultBlock(tool_use_id="tu1", content="results here", is_error=False),
            ]),
        ]
        tool = _make_tool(msgs)

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        result = await tool.execute(RecallTurnInput(turn_ids=["m3"]))

        assert result.is_error is False
        data = json.loads(result.output)
        content = data["recalled"][0]["content"]
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "tu1"
        assert content[0]["content"] == "results here"


# ── 6. session_ref returns object without .messages → is_error=True ──────────

class TestMissingMessagesAttr:
    @pytest.mark.asyncio
    async def test_no_messages_attr_returns_error(self):
        from neoagent.tools.builtin.recall_turn import RecallTurnTool

        class BrokenRef:
            pass  # no .messages, no .session

        tool = RecallTurnTool(session_ref=lambda: BrokenRef())
        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        result = await tool.execute(RecallTurnInput(turn_ids=["m1"]))

        assert result.is_error is True
        assert "missing" in result.output.lower() or "messages" in result.output.lower()

    @pytest.mark.asyncio
    async def test_session_with_nested_session_messages_works(self):
        """session_ref returns obj.session.messages (SessionState-like fallback)."""
        from neoagent.tools.builtin.recall_turn import RecallTurnTool

        inner = FakeSession(messages=[Message(id="m1", role="user", content="nested")])

        class StateWrapper:
            session = inner

        tool = RecallTurnTool(session_ref=lambda: StateWrapper())
        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        result = await tool.execute(RecallTurnInput(turn_ids=["m1"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert data["recalled"][0]["content"] == "nested"


# ── 7. messages without id are skipped in index ───────────────────────────────

class TestMessageWithoutId:
    @pytest.mark.asyncio
    async def test_message_without_id_not_indexed(self):
        """Message with id=None is not indexed and cannot be recalled."""
        msgs = [
            Message(id=None, role="user", content="no-id message"),
            Message(id="m1", role="assistant", content="has id"),
        ]
        tool = _make_tool(msgs)

        from neoagent.tools.builtin.recall_turn import RecallTurnInput
        result = await tool.execute(RecallTurnInput(turn_ids=["m1"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert len(data["recalled"]) == 1
        assert data["recalled"][0]["id"] == "m1"
