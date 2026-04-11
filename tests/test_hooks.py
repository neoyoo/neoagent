from __future__ import annotations
import asyncio
import pytest
from unittest.mock import patch, MagicMock
from neoagent.hooks import (
    HookResult,
    HookManager,
    PreToolCallEvent,
    PostToolCallEvent,
    PreProviderCallEvent,
    PostProviderCallEvent,
)
from neoagent.core.types import Message, TextBlock
from neoagent.providers.base import Response
from neoagent.config import NeoAgentConfig
from neoagent.agent import NeoAgent


def _make_mock_provider():
    mock = MagicMock()
    mock.get_context_window.return_value = 200_000
    return mock


# ── HookResult construction ───────────────────────────────────────────────────

def test_hookresult_allow():
    r = HookResult.allow()
    assert r.action == "allow"
    assert r.reason is None
    assert r.modified_data is None


def test_hookresult_deny_no_reason():
    r = HookResult.deny()
    assert r.action == "deny"
    assert r.reason == ""


def test_hookresult_deny_with_reason():
    r = HookResult.deny("dangerous command")
    assert r.action == "deny"
    assert r.reason == "dangerous command"


def test_hookresult_modify():
    r = HookResult.modify({"key": "val"})
    assert r.action == "modify"
    assert r.modified_data == {"key": "val"}
    assert r.reason is None


def test_hookresult_modify_with_reason():
    r = HookResult.modify({"x": 1}, reason="changed input")
    assert r.reason == "changed input"


def test_hookresult_is_frozen():
    r = HookResult.allow()
    with pytest.raises((AttributeError, TypeError)):
        r.action = "deny"  # type: ignore


# ── Hook event types ─────────────────────────────────────────────────────────

def test_pre_tool_call_event_fields():
    e = PreToolCallEvent(tool_name="bash", tool_input={"command": "ls"}, call_id="c1")
    assert e.tool_name == "bash"
    assert e.tool_input == {"command": "ls"}
    assert e.call_id == "c1"


def test_pre_tool_call_event_is_frozen():
    e = PreToolCallEvent(tool_name="bash", tool_input={}, call_id="c1")
    with pytest.raises((AttributeError, TypeError)):
        e.tool_name = "other"  # type: ignore


def test_post_tool_call_event_fields():
    e = PostToolCallEvent(
        tool_name="bash", tool_input={"command": "ls"},
        call_id="c1", result="output text", is_error=False,
    )
    assert e.result == "output text"
    assert e.is_error is False


def test_post_tool_call_event_error_flag():
    e = PostToolCallEvent(
        tool_name="bash", tool_input={}, call_id="c2",
        result="error msg", is_error=True,
    )
    assert e.is_error is True


def test_pre_provider_call_event_fields():
    msgs = [Message(role="user", content="hi")]
    e = PreProviderCallEvent(system="sys", messages=msgs, tools=[{"name": "bash"}])
    assert e.system == "sys"
    assert e.messages == msgs
    assert e.tools == [{"name": "bash"}]


def test_post_provider_call_event_fields():
    resp = Response(
        content=[TextBlock(text="hello")],
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
    )
    e = PostProviderCallEvent(response=resp, input_tokens=10, output_tokens=5)
    assert e.input_tokens == 10
    assert e.output_tokens == 5
    assert e.response is resp


# ── HookManager tests ────────────────────────────────────────────────────────

def _make_pre_event(**kwargs):
    defaults = dict(tool_name="bash", tool_input={"cmd": "ls"}, call_id="c1")
    defaults.update(kwargs)
    return PreToolCallEvent(**defaults)


def _make_post_event(**kwargs):
    defaults = dict(tool_name="bash", tool_input={"cmd": "ls"}, call_id="c1", result="ok", is_error=False)
    defaults.update(kwargs)
    return PostToolCallEvent(**defaults)


@pytest.mark.asyncio
async def test_hookmanager_no_handlers_returns_allow():
    mgr = HookManager()
    event = _make_pre_event()
    result = await mgr.run_pre("pre_tool_call", event)
    assert result.action == "allow"


@pytest.mark.asyncio
async def test_hookmanager_no_handlers_post_returns_allow():
    mgr = HookManager()
    event = _make_post_event()
    result = await mgr.run_post("post_tool_call", event)
    assert result.action == "allow"


@pytest.mark.asyncio
async def test_hookmanager_register_and_call():
    mgr = HookManager()
    calls = []

    async def handler(event):
        calls.append(event)
        return HookResult.allow()

    mgr.register("pre_tool_call", handler)
    event = _make_pre_event()
    await mgr.run_pre("pre_tool_call", event)
    assert len(calls) == 1
    assert calls[0] is event


@pytest.mark.asyncio
async def test_hookmanager_unregister_stops_calls():
    mgr = HookManager()
    calls = []

    async def handler(event):
        calls.append(event)
        return HookResult.allow()

    mgr.register("pre_tool_call", handler)
    mgr.unregister("pre_tool_call", handler)
    await mgr.run_pre("pre_tool_call", _make_pre_event())
    assert calls == []


@pytest.mark.asyncio
async def test_hookmanager_unregister_nonexistent_no_error():
    mgr = HookManager()

    async def handler(event):
        return HookResult.allow()

    # Should not raise even if handler was never registered
    mgr.unregister("pre_tool_call", handler)
    mgr.unregister("post_tool_call", handler)


@pytest.mark.asyncio
async def test_hookmanager_priority_order():
    mgr = HookManager()
    order = []

    async def h1(event):
        order.append(1)
        return HookResult.allow()

    async def h2(event):
        order.append(2)
        return HookResult.allow()

    async def h3(event):
        order.append(3)
        return HookResult.allow()

    # Register in reverse priority order
    mgr.register("pre_tool_call", h3, priority=30)
    mgr.register("pre_tool_call", h1, priority=10)
    mgr.register("pre_tool_call", h2, priority=20)

    await mgr.run_pre("pre_tool_call", _make_pre_event())
    assert order == [1, 2, 3]


@pytest.mark.asyncio
async def test_hookmanager_same_priority_registration_order():
    mgr = HookManager()
    order = []

    async def ha(event):
        order.append("a")
        return HookResult.allow()

    async def hb(event):
        order.append("b")
        return HookResult.allow()

    async def hc(event):
        order.append("c")
        return HookResult.allow()

    mgr.register("pre_tool_call", ha, priority=0)
    mgr.register("pre_tool_call", hb, priority=0)
    mgr.register("pre_tool_call", hc, priority=0)

    await mgr.run_pre("pre_tool_call", _make_pre_event())
    assert order == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_pre_hook_deny_short_circuits():
    mgr = HookManager()
    calls = []

    async def h_deny(event):
        calls.append("deny")
        return HookResult.deny("blocked")

    async def h_after(event):
        calls.append("after")
        return HookResult.allow()

    mgr.register("pre_tool_call", h_deny, priority=10)
    mgr.register("pre_tool_call", h_after, priority=20)

    await mgr.run_pre("pre_tool_call", _make_pre_event())
    assert calls == ["deny"]
    assert "after" not in calls


@pytest.mark.asyncio
async def test_pre_hook_deny_returns_deny_result():
    mgr = HookManager()

    async def h_deny(event):
        return HookResult.deny("not allowed")

    mgr.register("pre_tool_call", h_deny)
    result = await mgr.run_pre("pre_tool_call", _make_pre_event())
    assert result.action == "deny"
    assert result.reason == "not allowed"


@pytest.mark.asyncio
async def test_pre_hook_modify_passed_to_next_handler():
    mgr = HookManager()
    received = []

    async def h_modify(event):
        return HookResult.modify({"tool_input": {"cmd": "modified"}})

    async def h_check(event):
        received.append(event.tool_input)
        return HookResult.allow()

    mgr.register("pre_tool_call", h_modify, priority=10)
    mgr.register("pre_tool_call", h_check, priority=20)

    await mgr.run_pre("pre_tool_call", _make_pre_event())
    assert received == [{"cmd": "modified"}]


@pytest.mark.asyncio
async def test_pre_hook_modify_chain_accumulates():
    mgr = HookManager()

    async def h1(event):
        return HookResult.modify({"tool_input": {"cmd": "step1"}})

    async def h2(event):
        # Receives modified event from h1, modifies tool_name
        return HookResult.modify({"tool_name": "new_tool"})

    mgr.register("pre_tool_call", h1, priority=10)
    mgr.register("pre_tool_call", h2, priority=20)

    result = await mgr.run_pre("pre_tool_call", _make_pre_event())
    assert result.action == "modify"
    assert result.modified_data["tool_input"] == {"cmd": "step1"}
    assert result.modified_data["tool_name"] == "new_tool"


@pytest.mark.asyncio
async def test_post_hook_deny_is_ignored():
    mgr = HookManager()
    calls = []

    async def h_deny(event):
        calls.append("deny")
        return HookResult.deny("too late")

    async def h_after(event):
        calls.append("after")
        return HookResult.allow()

    mgr.register("post_tool_call", h_deny, priority=10)
    mgr.register("post_tool_call", h_after, priority=20)

    result = await mgr.run_post("post_tool_call", _make_post_event())
    # deny is ignored in post; both handlers should run
    assert calls == ["deny", "after"]
    assert result.action == "allow"


@pytest.mark.asyncio
async def test_post_hook_modify_accumulates():
    mgr = HookManager()

    async def h1(event):
        return HookResult.modify({"result": "modified_by_h1"})

    async def h2(event):
        return HookResult.modify({"is_error": True})

    mgr.register("post_tool_call", h1, priority=10)
    mgr.register("post_tool_call", h2, priority=20)

    result = await mgr.run_post("post_tool_call", _make_post_event())
    assert result.action == "modify"
    assert result.modified_data["result"] == "modified_by_h1"
    assert result.modified_data["is_error"] is True


@pytest.mark.asyncio
async def test_handler_exception_skipped_pre():
    mgr = HookManager()
    calls = []

    async def h_boom(event):
        raise RuntimeError("unexpected error")

    async def h_ok(event):
        calls.append("ok")
        return HookResult.allow()

    mgr.register("pre_tool_call", h_boom, priority=10)
    mgr.register("pre_tool_call", h_ok, priority=20)

    result = await mgr.run_pre("pre_tool_call", _make_pre_event())
    # Exception is caught, chain continues
    assert calls == ["ok"]
    assert result.action == "allow"


@pytest.mark.asyncio
async def test_handler_exception_skipped_post():
    mgr = HookManager()
    calls = []

    async def h_boom(event):
        raise ValueError("boom")

    async def h_ok(event):
        calls.append("ok")
        return HookResult.allow()

    mgr.register("post_tool_call", h_boom, priority=10)
    mgr.register("post_tool_call", h_ok, priority=20)

    result = await mgr.run_post("post_tool_call", _make_post_event())
    assert calls == ["ok"]
    assert result.action == "allow"


# ── NeoAgent hook API tests ───────────────────────────────────────────────────

def _make_agent() -> NeoAgent:
    config = NeoAgentConfig(api_key="test-key", model="claude-3-5-haiku-20241022")
    return NeoAgent(config)


@patch("neoagent.agent._create_provider")
def test_agent_has_hook_manager(mock_create):
    mock_create.return_value = _make_mock_provider()
    agent = _make_agent()
    assert isinstance(agent._hook_manager, HookManager)


@patch("neoagent.agent._create_provider")
def test_agent_hook_registers_handler(mock_create):
    mock_create.return_value = _make_mock_provider()
    agent = _make_agent()

    async def handler(event):
        return HookResult.allow()

    agent.hook("pre_tool_call", handler)
    entries = agent._hook_manager._hooks.get("pre_tool_call", [])
    assert len(entries) == 1


@patch("neoagent.agent._create_provider")
def test_agent_unhook_removes_handler(mock_create):
    mock_create.return_value = _make_mock_provider()
    agent = _make_agent()

    async def handler(event):
        return HookResult.allow()

    agent.hook("pre_tool_call", handler)
    agent.unhook("pre_tool_call", handler)
    entries = agent._hook_manager._hooks.get("pre_tool_call", [])
    assert len(entries) == 0


@patch("neoagent.agent._create_provider")
def test_agent_hook_priority_forwarded(mock_create):
    mock_create.return_value = _make_mock_provider()
    agent = _make_agent()

    async def handler(event):
        return HookResult.allow()

    agent.hook("pre_tool_call", handler, priority=99)
    entries = agent._hook_manager._hooks.get("pre_tool_call", [])
    assert entries[0].priority == 99


@patch("neoagent.agent._create_provider")
def test_agent_on_decorator_registers(mock_create):
    mock_create.return_value = _make_mock_provider()
    agent = _make_agent()

    @agent.on("pre_tool_call")
    async def handler(event):
        return HookResult.allow()

    entries = agent._hook_manager._hooks.get("pre_tool_call", [])
    assert len(entries) == 1


@patch("neoagent.agent._create_provider")
def test_agent_on_decorator_returns_original_function(mock_create):
    mock_create.return_value = _make_mock_provider()
    agent = _make_agent()

    async def handler(event):
        return HookResult.allow()

    decorated = agent.on("pre_tool_call")(handler)
    assert decorated is handler


@patch("neoagent.agent._create_provider")
def test_agent_on_decorator_priority(mock_create):
    mock_create.return_value = _make_mock_provider()
    agent = _make_agent()

    @agent.on("pre_tool_call", priority=42)
    async def handler(event):
        return HookResult.allow()

    entries = agent._hook_manager._hooks.get("pre_tool_call", [])
    assert entries[0].priority == 42


@patch("neoagent.agent._create_provider")
def test_agent_on_decorator_syntax_equivalent_to_hook(mock_create):
    mock_create.return_value = _make_mock_provider()
    agent = _make_agent()

    async def handler(event):
        return HookResult.allow()

    agent.hook("pre_tool_call", handler, priority=5)
    entries_hook = agent._hook_manager._hooks.get("pre_tool_call", [])
    assert len(entries_hook) == 1
    assert entries_hook[0].priority == 5
    assert entries_hook[0].handler is handler
