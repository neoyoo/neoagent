from __future__ import annotations
import pytest
from neoagent.hooks import (
    HookResult,
    PreToolCallEvent,
    PostToolCallEvent,
    PreProviderCallEvent,
    PostProviderCallEvent,
)
from neoagent.core.types import Message, TextBlock
from neoagent.providers.base import Response


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
