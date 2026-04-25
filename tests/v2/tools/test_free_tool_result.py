# tests/v2/tools/test_free_tool_result.py
"""Unit tests for FreeToolResultTool."""
from __future__ import annotations

import json
import pytest

from neoagent.core.types import Message, ToolResultBlock
from neoagent.session import Session, SessionState, FreedToolResult
from neoagent.tools.builtin.free_tool_result import FreeToolResultTool, FreeToolResultInput
from neoagent.tools.builtin.tool_search import set_current_session


def _make_session(messages: list[Message] | None = None) -> Session:
    session = Session.create()
    if messages:
        session.messages = messages
    return session


def _make_tool() -> FreeToolResultTool:
    return FreeToolResultTool()


# ── 1. No session set → is_error=True ─────────────────────────────────────────

class TestNoSession:
    @pytest.mark.asyncio
    async def test_no_session_returns_is_error(self):
        set_current_session(None)
        tool = _make_tool()
        result = await tool.execute(FreeToolResultInput(tool_use_ids=["id1"]))
        assert result.is_error is True
        assert "no active session context" in result.output


# ── 2. Free an existing tool_use_id ───────────────────────────────────────────

class TestFreeExisting:
    @pytest.mark.asyncio
    async def test_free_existing_id_goes_to_freed_list(self):
        content = "some tool output"
        msg = Message(role="user", content=[
            ToolResultBlock(tool_use_id="id1", content=content),
        ])
        session = _make_session([msg])
        set_current_session(session)

        tool = _make_tool()
        result = await tool.execute(FreeToolResultInput(tool_use_ids=["id1"]))

        assert result.is_error is False
        data = json.loads(result.output)
        assert "id1" in data["freed"]
        assert data["not_found"] == []
        assert data["already_freed"] == []
        assert "id1" in session.state.freed_tool_results

    @pytest.mark.asyncio
    async def test_freed_tool_result_fields_populated(self):
        content = "short content"
        msg = Message(role="user", content=[
            ToolResultBlock(tool_use_id="id2", content=content),
        ])
        session = _make_session([msg])
        session.state.tool_use_to_tool_name["id2"] = "my_tool"
        set_current_session(session)

        tool = _make_tool()
        await tool.execute(FreeToolResultInput(tool_use_ids=["id2"]))

        freed = session.state.freed_tool_results["id2"]
        assert isinstance(freed, FreedToolResult)
        assert freed.id == "id2"
        assert freed.tool_name == "my_tool"
        assert freed.original_content == content
        assert freed.size == len(content.encode())

    @pytest.mark.asyncio
    async def test_preview_truncated_to_80_chars_with_ellipsis(self):
        long_content = "x" * 100
        msg = Message(role="user", content=[
            ToolResultBlock(tool_use_id="id3", content=long_content),
        ])
        session = _make_session([msg])
        set_current_session(session)

        tool = _make_tool()
        await tool.execute(FreeToolResultInput(tool_use_ids=["id3"]))

        freed = session.state.freed_tool_results["id3"]
        assert freed.preview == "x" * 80 + "…"

    @pytest.mark.asyncio
    async def test_preview_no_ellipsis_when_content_le_80_chars(self):
        short_content = "y" * 80
        msg = Message(role="user", content=[
            ToolResultBlock(tool_use_id="id4", content=short_content),
        ])
        session = _make_session([msg])
        set_current_session(session)

        tool = _make_tool()
        await tool.execute(FreeToolResultInput(tool_use_ids=["id4"]))

        freed = session.state.freed_tool_results["id4"]
        assert freed.preview == short_content
        assert "…" not in freed.preview


# ── 3. Free a non-existent id → not_found ─────────────────────────────────────

class TestFreeNonExistent:
    @pytest.mark.asyncio
    async def test_nonexistent_id_goes_to_not_found(self):
        session = _make_session([])
        set_current_session(session)

        tool = _make_tool()
        result = await tool.execute(FreeToolResultInput(tool_use_ids=["ghost"]))

        data = json.loads(result.output)
        assert "ghost" in data["not_found"]
        assert data["freed"] == []
        assert data["already_freed"] == []


# ── 4. Free an already-freed id → already_freed, no double-write ──────────────

class TestAlreadyFreed:
    @pytest.mark.asyncio
    async def test_already_freed_id_goes_to_already_freed(self):
        session = _make_session([])
        original = FreedToolResult(
            id="id5", tool_name="t", size=3, preview="abc", original_content="abc"
        )
        session.state.freed_tool_results["id5"] = original
        set_current_session(session)

        tool = _make_tool()
        result = await tool.execute(FreeToolResultInput(tool_use_ids=["id5"]))

        data = json.loads(result.output)
        assert "id5" in data["already_freed"]
        assert data["freed"] == []
        # No overwrite
        assert session.state.freed_tool_results["id5"] is original


# ── 5. Mixed: freed + not_found + already_freed ────────────────────────────────

class TestMixedInput:
    @pytest.mark.asyncio
    async def test_mixed_input_all_three_lists_populated(self):
        msg = Message(role="user", content=[
            ToolResultBlock(tool_use_id="new_id", content="hello"),
        ])
        session = _make_session([msg])
        session.state.freed_tool_results["old_id"] = FreedToolResult(
            id="old_id", tool_name="x", size=1, preview="x", original_content="x"
        )
        set_current_session(session)

        tool = _make_tool()
        result = await tool.execute(FreeToolResultInput(
            tool_use_ids=["new_id", "ghost_id", "old_id"]
        ))

        data = json.loads(result.output)
        assert "new_id" in data["freed"]
        assert "ghost_id" in data["not_found"]
        assert "old_id" in data["already_freed"]


# ── 6. Tool name resolved via tool_use_to_tool_name; missing → "unknown" ───────

class TestToolNameResolution:
    @pytest.mark.asyncio
    async def test_known_tool_name_used(self):
        msg = Message(role="user", content=[
            ToolResultBlock(tool_use_id="tu1", content="data"),
        ])
        session = _make_session([msg])
        session.state.tool_use_to_tool_name["tu1"] = "web_search"
        set_current_session(session)

        tool = _make_tool()
        await tool.execute(FreeToolResultInput(tool_use_ids=["tu1"]))

        assert session.state.freed_tool_results["tu1"].tool_name == "web_search"

    @pytest.mark.asyncio
    async def test_unknown_tool_name_defaults_to_unknown(self):
        msg = Message(role="user", content=[
            ToolResultBlock(tool_use_id="tu2", content="data"),
        ])
        session = _make_session([msg])
        # No entry in tool_use_to_tool_name
        set_current_session(session)

        tool = _make_tool()
        await tool.execute(FreeToolResultInput(tool_use_ids=["tu2"]))

        assert session.state.freed_tool_results["tu2"].tool_name == "unknown"
