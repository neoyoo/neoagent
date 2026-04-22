# tests/v2/e2e/test_source_wrap.py
"""Phase 8 Batch 1 — Task 8.5: <source> auto-wrap E2E

Tests the end-to-end chain: NeoAgent (with enable_source_wrap=True) registers
source_wrap_hook as a post_tool_call hook; ToolExecutor calls it after each
tool execution.

Spec refs: § 18.2 (source tag wrapping semantics)
Contract refs: C5 (hook integration)

NOTE on known SDK bug (xfail tag):
  source_wrap_hook expects event.tool (BaseTool) and event.result (ToolResult),
  but ToolExecutor's PostToolCallEvent has event.tool_name (str) and
  event.result (str). The hook silently no-ops via the real ToolExecutor
  because getattr(event, 'tool', None) returns None.

  Tests exercising the full NeoAgent → ToolExecutor → hook chain are marked
  xfail(strict=False) with reason="BUG: source_wrap_hook event shape mismatch".
  Bug must be fixed by a dedicated bug-fix subagent (no SDK changes in this batch).
"""
from __future__ import annotations

import asyncio
import warnings
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from neoagent.agent import NeoAgent
from neoagent.config import NeoAgentConfig
from neoagent.core.types import Message, TextBlock, ToolResult, ToolUseBlock
from neoagent.events import EventBus
from neoagent.providers.base import Provider, Response
from neoagent.session import Session
from neoagent.tools.base import BaseTool
from neoagent.v2.security.source_wrap import source_wrap_hook, wrap_source_content


# ── Helpers ───────────────────────────────────────────────────────────────────


class _EmptyInput(BaseModel):
    pass


class MockExternalTool(BaseTool):
    """Simulates a web_fetch-style tool returning external content."""
    name = "mock_external_fetch"
    description = "Fetches external web content (mock)"
    input_model = _EmptyInput
    permission = "auto"
    is_concurrent_safe = True
    returns_external_content = True

    async def execute(self, input: _EmptyInput) -> ToolResult:
        return ToolResult(call_id="", output="这是网页内容", is_error=False)


class MockInternalTool(BaseTool):
    """Simulates an internal tool that does NOT return external content."""
    name = "mock_internal_tool"
    description = "An internal tool (mock)"
    input_model = _EmptyInput
    permission = "auto"
    is_concurrent_safe = True
    returns_external_content = False

    async def execute(self, input: _EmptyInput) -> ToolResult:
        return ToolResult(call_id="", output="这是网页内容", is_error=False)


class MockErrorTool(BaseTool):
    """Simulates an external tool that returns an error."""
    name = "mock_error_fetch"
    description = "Fetches external content but returns error (mock)"
    input_model = _EmptyInput
    permission = "auto"
    is_concurrent_safe = True
    returns_external_content = True

    async def execute(self, input: _EmptyInput) -> ToolResult:
        return ToolResult(call_id="", output="fetch error: timeout", is_error=True)


def _make_tool_use_response(tool_name: str, call_id: str = "call_001") -> Response:
    """Return a provider response that calls the given tool."""
    return Response(
        content=[
            ToolUseBlock(id=call_id, name=tool_name, input={}),
        ],
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=5,
    )


def _make_end_turn_response(text: str = "Done.") -> Response:
    """Return a provider response that ends the turn."""
    return Response(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        input_tokens=5,
        output_tokens=3,
    )


class MockProvider(Provider):
    """Minimal mock provider that returns pre-configured responses in sequence."""

    def __init__(self, responses: list[Response]) -> None:
        self._responses = iter(responses)
        self.model = "mock-model"

    async def create(self, system: str, messages: list, tools: list, **kwargs) -> Response:
        return next(self._responses)

    def get_context_window(self) -> int:
        return 200_000


def _make_agent(
    provider: Provider,
    extra_tools: list[BaseTool] | None = None,
    enable_source_wrap: bool = True,
) -> NeoAgent:
    """Construct NeoAgent with a pre-built mock provider (no real API key)."""
    cfg = NeoAgentConfig(
        api_key="test-key",
        model="mock-model",
        auto_approve_tools=True,
        enable_source_wrap=enable_source_wrap,
        enable_security_prompt_blocks=False,  # keep prompts clean for testing
    )
    with patch("neoagent.agent._create_provider", return_value=provider):
        agent = NeoAgent(cfg)
    if extra_tools:
        for tool in extra_tools:
            agent.register_tool(tool)
    return agent


# ── Unit-level hook tests (bypass ToolExecutor, call hook directly) ────────────
# These verify the hook function's logic regardless of the executor event shape.


class TestSourceWrapHookUnit:
    """Unit tests: call source_wrap_hook directly with the expected event shape."""

    class _ToolStub:
        def __init__(self, name: str, returns_external_content: bool):
            self.name = name
            self.returns_external_content = returns_external_content

    class _ResultStub:
        def __init__(self, output: str, is_error: bool = False):
            self.output = output
            self.is_error = is_error

    class _Event:
        def __init__(self, tool=None, result=None):
            if tool is not None:
                self.tool = tool
            if result is not None:
                self.result = result

    @pytest.mark.asyncio
    async def test_8_5_1_source_wrap_default_enabled(self):
        """8.5.1: returns_external_content=True → output wrapped in <source> tag."""
        tool = self._ToolStub("mock_external_fetch", returns_external_content=True)
        result = self._ResultStub("这是网页内容")
        event = self._Event(tool=tool, result=result)

        await source_wrap_hook(event)

        assert '<source type="mock_external_fetch">' in result.output
        assert "这是网页内容" in result.output
        assert "</source>" in result.output

    @pytest.mark.asyncio
    async def test_8_5_2_returns_external_false_no_wrap(self):
        """8.5.2: returns_external_content=False → output unchanged."""
        tool = self._ToolStub("mock_internal_tool", returns_external_content=False)
        result = self._ResultStub("这是网页内容")
        event = self._Event(tool=tool, result=result)

        await source_wrap_hook(event)

        assert result.output == "这是网页内容"
        assert "<source" not in result.output

    @pytest.mark.asyncio
    async def test_8_5_4_error_result_not_wrapped(self):
        """8.5.4: is_error=True → output not wrapped even for external tool."""
        tool = self._ToolStub("mock_external_fetch", returns_external_content=True)
        result = self._ResultStub("fetch error: timeout", is_error=True)
        event = self._Event(tool=tool, result=result)

        await source_wrap_hook(event)

        assert result.output == "fetch error: timeout"
        assert "<source" not in result.output

    @pytest.mark.asyncio
    async def test_8_5_5_multi_call_independent(self):
        """8.5.5: Multiple calls each independently wrap their own output."""
        tool = self._ToolStub("mock_external_fetch", returns_external_content=True)

        result_a = self._ResultStub("content A")
        event_a = self._Event(tool=tool, result=result_a)
        await source_wrap_hook(event_a)

        result_b = self._ResultStub("content B")
        event_b = self._Event(tool=tool, result=result_b)
        await source_wrap_hook(event_b)

        expected_a = wrap_source_content("content A", tool_name="mock_external_fetch")
        expected_b = wrap_source_content("content B", tool_name="mock_external_fetch")
        assert result_a.output == expected_a
        assert result_b.output == expected_b
        # verify they are independent
        assert "content A" not in result_b.output
        assert "content B" not in result_a.output

    @pytest.mark.asyncio
    async def test_8_5_no_tool_on_event(self):
        """Hook no-ops when event lacks 'tool' attribute."""
        result = self._ResultStub("data")
        event = self._Event(result=result)  # no 'tool' attr

        ret = await source_wrap_hook(event)

        assert ret is None
        assert result.output == "data"


# ── E2E tests: NeoAgent → ToolExecutor → hook chain ──────────────────────────
# XFAIL because PostToolCallEvent shape does not match what source_wrap_hook expects.
# Bug: executor passes event.tool_name (str) but hook reads event.tool (BaseTool).


_BUG_REASON = (
    "BUG: source_wrap_hook event shape mismatch — "
    "ToolExecutor emits PostToolCallEvent with tool_name:str + result:str, "
    "but source_wrap_hook expects event.tool:BaseTool + event.result.output:str. "
    "Hook silently no-ops; wrapping never occurs via real ToolExecutor chain."
)


class TestSourceWrapE2EChain:
    """E2E: NeoAgent + real ToolExecutor + source_wrap_hook registration."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(strict=False, reason=_BUG_REASON)
    async def test_8_5_1_e2e_source_wrap_external_tool(self):
        """8.5.1 E2E: external tool output is wrapped when enable_source_wrap=True."""
        responses = [
            _make_tool_use_response("mock_external_fetch", "call_001"),
            _make_end_turn_response("Done."),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, extra_tools=[MockExternalTool()], enable_source_wrap=True)

        session = agent.new_session()
        session.messages.append(Message(role="user", content="fetch me something"))
        result = await agent.run(session.messages, session=session)

        # The last tool_use turn should have the wrapped output in messages
        all_tool_results = []
        for turn in result.turns:
            for tr in turn.tool_results:
                all_tool_results.append(tr)

        assert len(all_tool_results) > 0
        wrapped_output = all_tool_results[0].output
        assert '<source type="mock_external_fetch">' in wrapped_output

    @pytest.mark.asyncio
    @pytest.mark.xfail(strict=False, reason=_BUG_REASON)
    async def test_8_5_2_e2e_internal_tool_not_wrapped(self):
        """8.5.2 E2E: internal tool output is NOT wrapped (returns_external_content=False)."""
        responses = [
            _make_tool_use_response("mock_internal_tool", "call_001"),
            _make_end_turn_response("Done."),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, extra_tools=[MockInternalTool()], enable_source_wrap=True)

        session = agent.new_session()
        session.messages.append(Message(role="user", content="do internal thing"))
        result = await agent.run(session.messages, session=session)

        all_tool_results = []
        for turn in result.turns:
            for tr in turn.tool_results:
                all_tool_results.append(tr)

        assert len(all_tool_results) > 0
        output = all_tool_results[0].output
        assert "<source" not in output
        assert output == "这是网页内容"

    @pytest.mark.asyncio
    @pytest.mark.xfail(strict=False, reason=_BUG_REASON)
    async def test_8_5_3_e2e_disable_source_wrap(self):
        """8.5.3 E2E: enable_source_wrap=False → external tool output NOT wrapped."""
        responses = [
            _make_tool_use_response("mock_external_fetch", "call_001"),
            _make_end_turn_response("Done."),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(
            provider,
            extra_tools=[MockExternalTool()],
            enable_source_wrap=False,
        )

        session = agent.new_session()
        session.messages.append(Message(role="user", content="fetch me something"))
        result = await agent.run(session.messages, session=session)

        all_tool_results = []
        for turn in result.turns:
            for tr in turn.tool_results:
                all_tool_results.append(tr)

        assert len(all_tool_results) > 0
        output = all_tool_results[0].output
        assert "<source" not in output
        assert output == "这是网页内容"

    @pytest.mark.asyncio
    @pytest.mark.xfail(strict=False, reason=_BUG_REASON)
    async def test_8_5_4_e2e_error_result_not_wrapped(self):
        """8.5.4 E2E: is_error=True → output not wrapped even for external tool."""
        responses = [
            _make_tool_use_response("mock_error_fetch", "call_001"),
            _make_end_turn_response("Done."),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(provider, extra_tools=[MockErrorTool()], enable_source_wrap=True)

        session = agent.new_session()
        session.messages.append(Message(role="user", content="fetch me something"))
        result = await agent.run(session.messages, session=session)

        all_tool_results = []
        for turn in result.turns:
            for tr in turn.tool_results:
                all_tool_results.append(tr)

        assert len(all_tool_results) > 0
        output = all_tool_results[0].output
        assert "<source" not in output
        assert output == "fetch error: timeout"


class TestSourceWrapE2EDisabledVerification:
    """E2E tests that can pass: verify enable_source_wrap=False does NOT register the hook."""

    def test_8_5_3_disable_source_wrap_no_hook_registered(self):
        """8.5.3: enable_source_wrap=False → source_wrap_hook NOT in post_tool_call hooks."""
        from neoagent.v2.security.source_wrap import source_wrap_hook

        provider = MockProvider([])
        agent = _make_agent(provider, enable_source_wrap=False)

        handlers = [
            entry.handler
            for entry in agent._hook_manager._hooks.get("post_tool_call", [])
        ]
        assert source_wrap_hook not in handlers

    def test_8_5_1_enable_source_wrap_hook_registered(self):
        """8.5.1: enable_source_wrap=True (default) → source_wrap_hook IS in post_tool_call hooks."""
        from neoagent.v2.security.source_wrap import source_wrap_hook

        provider = MockProvider([])
        agent = _make_agent(provider, enable_source_wrap=True)

        handlers = [
            entry.handler
            for entry in agent._hook_manager._hooks.get("post_tool_call", [])
        ]
        assert source_wrap_hook in handlers
