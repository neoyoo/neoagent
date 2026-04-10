from __future__ import annotations

import pytest
from neoagent.core.types import (
    ConversationResult, Message, TextBlock, ToolCall, ToolResult,
    ToolResultBlock, ToolUseBlock, Turn,
)

class TestTextBlock:
    def test_construction(self) -> None:
        block = TextBlock(text="hello")
        assert block.type == "text"
        assert block.text == "hello"

class TestToolUseBlock:
    def test_construction(self) -> None:
        block = ToolUseBlock(id="tu_1", name="read", input={"path": "/tmp/f"})
        assert block.type == "tool_use"
        assert block.id == "tu_1"
        assert block.name == "read"
        assert block.input == {"path": "/tmp/f"}

class TestToolResultBlock:
    def test_construction_success(self) -> None:
        block = ToolResultBlock(tool_use_id="tu_1", content="file contents")
        assert block.type == "tool_result"
        assert block.tool_use_id == "tu_1"
        assert block.content == "file contents"
        assert block.is_error is False

    def test_construction_error(self) -> None:
        block = ToolResultBlock(tool_use_id="tu_1", content="not found", is_error=True)
        assert block.is_error is True

class TestMessage:
    def test_string_content(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_block_content(self) -> None:
        blocks = [TextBlock(text="hi"), ToolUseBlock(id="tu_1", name="bash", input={})]
        msg = Message(role="assistant", content=blocks)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_role_is_validated(self) -> None:
        with pytest.raises(Exception):
            Message(role="system", content="oops")

class TestToolCall:
    def test_construction(self) -> None:
        call = ToolCall(id="tc_1", name="grep", input={"pattern": "TODO"})
        assert call.id == "tc_1"

class TestToolResult:
    def test_construction_default(self) -> None:
        result = ToolResult(call_id="tc_1", output="found 3 matches")
        assert result.is_error is False

    def test_construction_error(self) -> None:
        result = ToolResult(call_id="tc_1", output="permission denied", is_error=True)
        assert result.is_error is True

class TestTurn:
    def test_construction_end_turn(self) -> None:
        msg = Message(role="assistant", content="done")
        turn = Turn(response=msg, tool_calls=[], tool_results=[], stop_reason="end_turn")
        assert turn.stop_reason == "end_turn"

    def test_construction_tool_use(self) -> None:
        msg = Message(role="assistant", content=[])
        call = ToolCall(id="tc_1", name="read", input={"path": "/f"})
        result = ToolResult(call_id="tc_1", output="content")
        turn = Turn(response=msg, tool_calls=[call], tool_results=[result], stop_reason="tool_use")
        assert len(turn.tool_calls) == 1

    def test_stop_reason_validated(self) -> None:
        msg = Message(role="assistant", content="x")
        with pytest.raises(Exception):
            Turn(response=msg, tool_calls=[], tool_results=[], stop_reason="unknown")

class TestConversationResult:
    def test_completed(self) -> None:
        result = ConversationResult(turns=[], reason="completed")
        assert result.reason == "completed"

    def test_reason_validated(self) -> None:
        with pytest.raises(Exception):
            ConversationResult(turns=[], reason="timeout")
