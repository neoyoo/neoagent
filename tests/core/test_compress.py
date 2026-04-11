from __future__ import annotations
import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from neoagent.core.types import Message, TextBlock, ToolResultBlock, ToolUseBlock
from neoagent.core.compress import ContextCompressor
from neoagent.session import SessionState

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


def _make_messages(n: int) -> list[Message]:
    return [
        Message(role="user" if i % 2 == 0 else "assistant", content=f"message {i}")
        for i in range(n)
    ]


# --- v2 iterative summary tests (session_state-based) ---

@pytest.mark.asyncio
async def test_iterative_summary_passes_previous_to_llm() -> None:
    """previous_summary from session_state must appear in the LLM system prompt on compression."""
    provider = _make_provider("GOAL: original goal\nPROGRESS: done step 1")
    compressor = ContextCompressor(provider=provider)
    state = SessionState(previous_summary="GOAL: original goal\nPROGRESS: nothing yet")
    msgs = _make_messages(10)
    await compressor.compress(msgs, context_budget=1000, session_state=state)
    call_args = provider.create.call_args
    system_text = call_args.kwargs.get("system", "")
    assert "GOAL: original goal" in system_text


@pytest.mark.asyncio
async def test_iterative_summary_updates_previous_summary() -> None:
    """After compression, session_state.previous_summary is updated to the new summary."""
    new_summary = "GOAL: build agent\nPROGRESS: completed v1"
    provider = _make_provider(new_summary)
    compressor = ContextCompressor(provider=provider)
    state = SessionState()
    assert state.previous_summary is None
    msgs = _make_messages(10)
    await compressor.compress(msgs, context_budget=1000, session_state=state)
    assert state.previous_summary == new_summary


def test_sanitize_tool_pairs_maintains_role_alternation() -> None:
    """When filtering drops a user message between two assistant messages,
    a placeholder must be inserted to preserve alternating roles."""
    c = ContextCompressor(provider=_make_provider())
    msgs = [
        _user("go"),
        _assistant("first"),
        _assistant("second"),  # consecutive same role — needs placeholder
    ]
    result = c._sanitize_tool_pairs(msgs)
    roles = [m.role for m in result]
    # No two consecutive messages should share the same role
    for i in range(1, len(roles)):
        assert roles[i] != roles[i - 1], (
            f"Role alternation violated at index {i}: {roles}"
        )
    # Both assistant messages should still be present
    assistant_texts = [
        m.content for m in result if m.role == "assistant" and isinstance(m.content, str)
    ]
    assert "first" in assistant_texts
    assert "second" in assistant_texts


@pytest.mark.asyncio
async def test_structured_template_keywords_in_system_prompt() -> None:
    """The compression system prompt must include structured template keywords."""
    provider = _make_provider("GOAL: x\nPROGRESS: y")
    compressor = ContextCompressor(provider=provider)
    msgs = _make_messages(10)
    await compressor.compress(msgs, context_budget=1000)
    call_args = provider.create.call_args
    system_text = call_args.kwargs.get("system", "")
    for keyword in ("GOAL", "PROGRESS", "DECISIONS"):
        assert keyword in system_text, f"Missing keyword {keyword!r} in system prompt"


# --- NEW: Session integration tests ---

def test_compress_no_instance_previous_summary() -> None:
    """ContextCompressor must NOT have _previous_summary as an instance attribute."""
    c = ContextCompressor(provider=_make_provider())
    assert not hasattr(c, "_previous_summary"), (
        "_previous_summary should have been removed from instance — it's in SessionState now"
    )


@pytest.mark.asyncio
async def test_compress_reads_and_writes_session_state() -> None:
    """compress() reads previous_summary from session_state and writes back the new summary."""
    new_summary = "GOAL: test\nPROGRESS: step 1"
    provider = _make_provider(new_summary)
    compressor = ContextCompressor(provider=provider)

    state = SessionState(previous_summary="GOAL: test\nPROGRESS: nothing yet")
    msgs = _make_messages(10)
    await compressor.compress(msgs, context_budget=1000, session_state=state)

    # Old summary should appear in the LLM call
    call_args = provider.create.call_args
    system_text = call_args.kwargs.get("system", "")
    assert "GOAL: test" in system_text

    # New summary written back to state
    assert state.previous_summary == new_summary


@pytest.mark.asyncio
async def test_compress_without_session_state_still_works() -> None:
    """compress() works fine with no session_state (no previous summary context)."""
    provider = _make_provider("GOAL: fresh")
    compressor = ContextCompressor(provider=provider)
    msgs = _make_messages(10)
    result = await compressor.compress(msgs, context_budget=1000)
    assert isinstance(result, list)
    assert len(result) > 0
