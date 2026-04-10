from __future__ import annotations
import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from neoagent.core.types import Message, TextBlock, ToolResultBlock, ToolUseBlock
from neoagent.core.compress import ContextCompressor

def _user(text: str) -> Message:
    return Message(role="user", content=text)

def _assistant(text: str) -> Message:
    return Message(role="assistant", content=text)

def _assistant_with_tool(tool_id: str, tool_name: str) -> Message:
    return Message(role="assistant", content=[ToolUseBlock(id=tool_id, name=tool_name, input={})])

def _tool_result(tool_id: str, output: str = "ok") -> Message:
    return Message(role="user", content=[ToolResultBlock(tool_use_id=tool_id, content=output)])

def _make_provider(summary: str = "SUMMARY") -> MagicMock:
    response = MagicMock()
    response.text_content = summary
    provider = MagicMock()
    provider.create = AsyncMock(return_value=response)
    return provider

class TestEstimateTokens:
    def test_empty(self):
        c = ContextCompressor(provider=_make_provider())
        assert c.estimate_tokens([]) == 0

    def test_nonzero(self):
        c = ContextCompressor(provider=_make_provider())
        assert c.estimate_tokens([_user("Hello")]) > 0

    def test_longer_more_tokens(self):
        c = ContextCompressor(provider=_make_provider())
        assert c.estimate_tokens([_user("Hi " * 200)]) > c.estimate_tokens([_user("Hi")])

    def test_list_content(self):
        c = ContextCompressor(provider=_make_provider())
        msgs = [Message(role="assistant", content=[TextBlock(text="check"), ToolUseBlock(id="t1", name="read", input={"p": "/f"})])]
        assert c.estimate_tokens(msgs) > 0

class TestEstimateToolsTokens:
    def test_empty(self):
        c = ContextCompressor(provider=_make_provider())
        assert c.estimate_tools_tokens([]) == 0

    def test_nonzero(self):
        c = ContextCompressor(provider=_make_provider())
        assert c.estimate_tools_tokens([{"name": "read", "description": "Read", "input_schema": {}}]) > 0

class TestShouldCompress:
    def test_below_threshold(self):
        c = ContextCompressor(provider=_make_provider())
        assert c.should_compress([_user("Hi")], [], context_budget=100_000) is False

    def test_above_threshold(self):
        c = ContextCompressor(provider=_make_provider())
        assert c.should_compress([_user("word " * 500)], [], context_budget=50) is True

    def test_tools_counted(self):
        c = ContextCompressor(provider=_make_provider())
        big = [{"name": f"t{i}", "description": "x" * 200, "input_schema": {}} for i in range(10)]
        assert c.should_compress([_user("short")], big, context_budget=50) is True

class TestSanitizeToolPairs:
    def test_clean_unchanged(self):
        c = ContextCompressor(provider=_make_provider())
        msgs = [_user("go"), _assistant_with_tool("t1", "read"), _tool_result("t1"), _assistant("done")]
        assert len(c._sanitize_tool_pairs(msgs)) == 4

    def test_orphan_use_removed(self):
        c = ContextCompressor(provider=_make_provider())
        msgs = [_user("go"), _assistant_with_tool("orphan", "read"), _assistant("done")]
        result = c._sanitize_tool_pairs(msgs)
        for msg in result:
            if isinstance(msg.content, list):
                for b in msg.content:
                    assert not isinstance(b, ToolUseBlock)

    def test_orphan_result_removed(self):
        c = ContextCompressor(provider=_make_provider())
        msgs = [_user("go"), _tool_result("missing"), _assistant("done")]
        result = c._sanitize_tool_pairs(msgs)
        for msg in result:
            if isinstance(msg.content, list):
                for b in msg.content:
                    assert not isinstance(b, ToolResultBlock)

    def test_matched_preserved(self):
        c = ContextCompressor(provider=_make_provider())
        msgs = [_user("go"), _assistant_with_tool("t1", "bash"), _tool_result("t1")]
        result = c._sanitize_tool_pairs(msgs)
        has_use = any(isinstance(m.content, list) and any(isinstance(b, ToolUseBlock) for b in m.content) for m in result)
        has_res = any(isinstance(m.content, list) and any(isinstance(b, ToolResultBlock) for b in m.content) for m in result)
        assert has_use and has_res

    def test_empty(self):
        c = ContextCompressor(provider=_make_provider())
        assert c._sanitize_tool_pairs([]) == []

class TestCompress:
    async def test_returns_list(self):
        c = ContextCompressor(provider=_make_provider("Summary"))
        msgs = [_user("start")] + [_user(f"m{i}") for i in range(10)]
        result = await c.compress(msgs, context_budget=200_000)
        assert isinstance(result, list) and len(result) > 0

    async def test_first_preserved(self):
        c = ContextCompressor(provider=_make_provider("SUM"))
        first = _user("ANCHOR")
        msgs = [first] + [_user(f"m{i}") for i in range(8)]
        result = await c.compress(msgs, context_budget=200_000)
        assert result[0].content == "ANCHOR"

    async def test_summary_injected(self):
        c = ContextCompressor(provider=_make_provider("CONTEXT_SUMMARY"))
        msgs = [_user("init")] + [_user(f"t{i}") for i in range(6)]
        result = await c.compress(msgs, context_budget=200_000)
        all_text = " ".join(m.content if isinstance(m.content, str) else "" for m in result)
        assert "CONTEXT_SUMMARY" in all_text

    async def test_success_resets_failures(self):
        c = ContextCompressor(provider=_make_provider("ok"))
        c._consecutive_failures = 2
        msgs = [_user("a")] + [_user(f"b{i}") for i in range(4)]
        await c.compress(msgs, context_budget=200_000)
        assert c._consecutive_failures == 0

class TestCircuitBreaker:
    async def test_failure_increments(self):
        p = MagicMock()
        p.create = AsyncMock(side_effect=RuntimeError("down"))
        c = ContextCompressor(provider=p, max_failures=3)
        msgs = [_user("a")] + [_user(f"b{i}") for i in range(4)]
        await c.compress(msgs, context_budget=200_000)
        assert c._consecutive_failures == 1

    async def test_breaker_open_skips_provider(self):
        p = MagicMock()
        p.create = AsyncMock(side_effect=RuntimeError("down"))
        c = ContextCompressor(provider=p, max_failures=3)
        c._consecutive_failures = 3
        msgs = [_user("first")] + [_user(f"x{i}") for i in range(10)]
        result = await c.compress(msgs, context_budget=200_000)
        p.create.assert_not_called()
        assert isinstance(result, list) and len(result) > 0

    async def test_fallback_preserves_first(self):
        p = MagicMock()
        p.create = AsyncMock(side_effect=RuntimeError("dead"))
        c = ContextCompressor(provider=p, max_failures=1)
        c._consecutive_failures = 1
        first = _user("anchor")
        msgs = [first] + [_user(f"x{i}") for i in range(20)]
        result = await c.compress(msgs, context_budget=200_000)
        assert result[0].content == "anchor"
