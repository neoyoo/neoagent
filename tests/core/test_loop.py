from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock
import pytest
from neoagent.core.loop import QueryLoop
from neoagent.core.types import ConversationResult, Message, TextBlock, ToolCall, ToolResult, ToolUseBlock, ToolResultBlock, Turn

def _text_response(text="done"):
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [TextBlock(text=text)]
    resp.text_content = text
    resp.tool_use_blocks = []
    return resp

def _tool_use_response(tool_id, tool_name, tool_input):
    tu = ToolUseBlock(id=tool_id, name=tool_name, input=tool_input)
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [tu]
    resp.text_content = ""
    resp.tool_use_blocks = [tu]
    return resp

def _max_tokens_response():
    resp = MagicMock()
    resp.stop_reason = "max_tokens"
    resp.content = []
    resp.text_content = ""
    resp.tool_use_blocks = []
    return resp

def _make_provider(*responses):
    p = MagicMock()
    p.get_context_window.return_value = 200_000
    p.create = AsyncMock(side_effect=list(responses))
    return p

def _make_registry(tool_results=None):
    r = MagicMock()
    r.get_schemas.return_value = []
    r.execute = AsyncMock(return_value=tool_results or [])
    return r

def _make_prompt_builder(text="You are neoagent."):
    b = MagicMock()
    b.build.return_value = text
    return b

class TestConstruction:
    def test_default_budget_from_provider(self):
        loop = QueryLoop(provider=_make_provider(), tool_registry=_make_registry(), prompt_builder=_make_prompt_builder())
        assert loop.context_budget == 200_000

    def test_explicit_budget(self):
        loop = QueryLoop(provider=_make_provider(), tool_registry=_make_registry(), prompt_builder=_make_prompt_builder(), context_budget=50_000)
        assert loop.context_budget == 50_000

    def test_default_max_turns(self):
        loop = QueryLoop(provider=_make_provider(), tool_registry=_make_registry(), prompt_builder=_make_prompt_builder())
        assert loop.max_turns == 30

class TestNormalCompletion:
    async def test_single_turn(self):
        p = _make_provider(_text_response("All done."))
        loop = QueryLoop(provider=p, tool_registry=_make_registry(), prompt_builder=_make_prompt_builder())
        result = await loop.run([Message(role="user", content="Hello")])
        assert result.reason == "completed"
        assert len(result.turns) == 1
        assert result.turns[0].stop_reason == "end_turn"

    async def test_system_prompt_passed(self):
        p = _make_provider(_text_response())
        loop = QueryLoop(provider=p, tool_registry=_make_registry(), prompt_builder=_make_prompt_builder("SYS"))
        await loop.run([Message(role="user", content="go")])
        assert p.create.call_args.kwargs["system"] == "SYS"

    async def test_tools_passed(self):
        p = _make_provider(_text_response())
        reg = _make_registry()
        reg.get_schemas.return_value = [{"name": "read"}]
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder())
        await loop.run([Message(role="user", content="go")])
        assert p.create.call_args.kwargs["tools"] == [{"name": "read"}]

class TestMaxTurns:
    async def test_returns_max_turns(self):
        resps = [_tool_use_response(f"t{i}", "bash", {}) for i in range(10)]
        p = _make_provider(*resps)
        reg = _make_registry([ToolResult(call_id="x", output="ok")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder(), max_turns=3)
        result = await loop.run([Message(role="user", content="go")])
        assert result.reason == "max_turns"
        assert len(result.turns) == 3

    async def test_provider_called_max_times(self):
        resps = [_tool_use_response(f"t{i}", "bash", {}) for i in range(5)]
        p = _make_provider(*resps)
        reg = _make_registry([ToolResult(call_id="x", output="ok")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder(), max_turns=3)
        await loop.run([Message(role="user", content="go")])
        assert p.create.call_count == 3

class TestToolFlow:
    async def test_tool_executed_and_fed_back(self):
        p = _make_provider(_tool_use_response("t1", "read", {"path": "/f"}), _text_response("done"))
        reg = _make_registry([ToolResult(call_id="t1", output="contents")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder())
        result = await loop.run([Message(role="user", content="read")])
        assert result.reason == "completed"
        assert len(result.turns) == 2
        assert result.turns[0].stop_reason == "tool_use"
        assert result.turns[0].tool_calls[0].id == "t1"

    async def test_registry_called(self):
        p = _make_provider(_tool_use_response("t1", "grep", {"p": "TODO"}), _text_response("done"))
        reg = _make_registry([ToolResult(call_id="t1", output="found")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder())
        await loop.run([Message(role="user", content="search")])
        reg.execute.assert_called_once()
        assert reg.execute.call_args.args[0][0].name == "grep"

    async def test_tool_result_in_messages(self):
        p = _make_provider(_tool_use_response("t1", "bash", {}), _text_response("done"))
        reg = _make_registry([ToolResult(call_id="t1", output="output")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder())
        await loop.run([Message(role="user", content="go")])
        second_msgs = p.create.call_args_list[1].kwargs["messages"]
        last = second_msgs[-1]
        assert last.role == "user"
        assert isinstance(last.content, list)
        assert any(isinstance(b, ToolResultBlock) for b in last.content)

class TestOnTurn:
    async def test_called_per_turn(self):
        p = _make_provider(_tool_use_response("t1", "r", {}), _text_response("done"))
        reg = _make_registry([ToolResult(call_id="t1", output="ok")])
        calls = []
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder(), on_turn=calls.append)
        await loop.run([Message(role="user", content="go")])
        assert len(calls) == 2
        assert calls[0].stop_reason == "tool_use"
        assert calls[1].stop_reason == "end_turn"

    async def test_none_callback_ok(self):
        p = _make_provider(_text_response())
        loop = QueryLoop(provider=p, tool_registry=_make_registry(), prompt_builder=_make_prompt_builder())
        result = await loop.run([Message(role="user", content="hi")])
        assert result.reason == "completed"

class TestMaxTokensRetry:
    async def test_retries(self):
        p = _make_provider(_max_tokens_response(), _text_response("ok"))
        loop = QueryLoop(provider=p, tool_registry=_make_registry(), prompt_builder=_make_prompt_builder())
        result = await loop.run([Message(role="user", content="go")])
        assert p.create.call_count == 2
        assert result.reason == "completed"
