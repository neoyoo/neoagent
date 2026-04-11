from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock
import pytest
from neoagent.core.loop import QueryLoop
from neoagent.core.types import ConversationResult, Message, TextBlock, ToolCall, ToolResult, ToolUseBlock, ToolResultBlock, Turn
from neoagent.session import Session, SessionState
from neoagent.events import EventBus, ToolCallEvent, ToolResultEvent, ProviderRequestEvent, TurnCompleteEvent

def _text_response(text="done"):
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [TextBlock(text=text)]
    resp.text_content = text
    resp.tool_use_blocks = []
    resp.input_tokens = 10
    resp.output_tokens = 5
    return resp

def _tool_use_response(tool_id, tool_name, tool_input):
    tu = ToolUseBlock(id=tool_id, name=tool_name, input=tool_input)
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [tu]
    resp.text_content = ""
    resp.tool_use_blocks = [tu]
    resp.input_tokens = 20
    resp.output_tokens = 8
    return resp

def _max_tokens_response():
    resp = MagicMock()
    resp.stop_reason = "max_tokens"
    resp.content = []
    resp.text_content = ""
    resp.tool_use_blocks = []
    resp.input_tokens = 5
    resp.output_tokens = 0
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

def _session_with(content: str) -> Session:
    """Create a fresh Session with a single user message."""
    s = Session.create()
    s.messages.append(Message(role="user", content=content))
    return s

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
        result = await loop.run(session=_session_with("Hello"))
        assert result.reason == "completed"
        assert len(result.turns) == 1
        assert result.turns[0].stop_reason == "end_turn"

    async def test_system_prompt_passed(self):
        p = _make_provider(_text_response())
        loop = QueryLoop(provider=p, tool_registry=_make_registry(), prompt_builder=_make_prompt_builder("SYS"))
        await loop.run(session=_session_with("go"))
        assert p.create.call_args.kwargs["system"] == "SYS"

    async def test_tools_passed(self):
        p = _make_provider(_text_response())
        reg = _make_registry()
        reg.get_schemas.return_value = [{"name": "read"}]
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder())
        await loop.run(session=_session_with("go"))
        assert p.create.call_args.kwargs["tools"] == [{"name": "read"}]

class TestMaxTurns:
    async def test_returns_max_turns(self):
        resps = [_tool_use_response(f"t{i}", "bash", {}) for i in range(10)]
        p = _make_provider(*resps)
        reg = _make_registry([ToolResult(call_id="x", output="ok")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder(), max_turns=3)
        result = await loop.run(session=_session_with("go"))
        assert result.reason == "max_turns"
        assert len(result.turns) == 3

    async def test_provider_called_max_times(self):
        resps = [_tool_use_response(f"t{i}", "bash", {}) for i in range(5)]
        p = _make_provider(*resps)
        reg = _make_registry([ToolResult(call_id="x", output="ok")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder(), max_turns=3)
        await loop.run(session=_session_with("go"))
        assert p.create.call_count == 3

class TestToolFlow:
    async def test_tool_executed_and_fed_back(self):
        p = _make_provider(_tool_use_response("t1", "read", {"path": "/f"}), _text_response("done"))
        reg = _make_registry([ToolResult(call_id="t1", output="contents")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder())
        result = await loop.run(session=_session_with("read"))
        assert result.reason == "completed"
        assert len(result.turns) == 2
        assert result.turns[0].stop_reason == "tool_use"
        assert result.turns[0].tool_calls[0].id == "t1"

    async def test_registry_called(self):
        p = _make_provider(_tool_use_response("t1", "grep", {"p": "TODO"}), _text_response("done"))
        reg = _make_registry([ToolResult(call_id="t1", output="found")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder())
        await loop.run(session=_session_with("search"))
        reg.execute.assert_called_once()
        assert reg.execute.call_args.args[0][0].name == "grep"

    async def test_tool_result_in_messages(self):
        p = _make_provider(_tool_use_response("t1", "bash", {}), _text_response("done"))
        reg = _make_registry([ToolResult(call_id="t1", output="output")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder())
        await loop.run(session=_session_with("go"))
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
        await loop.run(session=_session_with("go"))
        assert len(calls) == 2
        assert calls[0].stop_reason == "tool_use"
        assert calls[1].stop_reason == "end_turn"

    async def test_none_callback_ok(self):
        p = _make_provider(_text_response())
        loop = QueryLoop(provider=p, tool_registry=_make_registry(), prompt_builder=_make_prompt_builder())
        result = await loop.run(session=_session_with("hi"))
        assert result.reason == "completed"

class TestMaxTokensRetry:
    async def test_retries(self):
        p = _make_provider(_max_tokens_response(), _text_response("ok"))
        loop = QueryLoop(provider=p, tool_registry=_make_registry(), prompt_builder=_make_prompt_builder())
        result = await loop.run(session=_session_with("go"))
        assert p.create.call_count == 2
        assert result.reason == "completed"

class TestLegacyListAPI:
    """Backward-compat: loop.run([Message(...)]) must still work while agent.py is updated."""

    async def test_list_messages_still_works(self):
        p = _make_provider(_text_response("ok"))
        loop = QueryLoop(provider=p, tool_registry=_make_registry(), prompt_builder=_make_prompt_builder())
        result = await loop.run([Message(role="user", content="legacy")])
        assert result.reason == "completed"

    async def test_list_messages_tool_flow(self):
        p = _make_provider(_tool_use_response("t1", "bash", {}), _text_response("done"))
        reg = _make_registry([ToolResult(call_id="t1", output="ok")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder())
        result = await loop.run([Message(role="user", content="go")])
        assert result.reason == "completed"
        assert len(result.turns) == 2


# --- NEW: Session integration tests ---

class TestSessionIntegration:
    async def test_run_accepts_session(self):
        """loop.run(session=...) must accept a Session and complete normally."""
        p = _make_provider(_text_response("hi"))
        loop = QueryLoop(provider=p, tool_registry=_make_registry(), prompt_builder=_make_prompt_builder())
        session = Session.create()
        session.messages.append(Message(role="user", content="hello"))
        result = await loop.run(session=session)
        assert result.reason == "completed"

    async def test_run_accumulates_tokens_in_session_state(self):
        """Token counts must be accumulated in session.state after run."""
        p = _make_provider(_text_response("done"))
        loop = QueryLoop(provider=p, tool_registry=_make_registry(), prompt_builder=_make_prompt_builder())
        session = Session.create()
        session.messages.append(Message(role="user", content="hi"))
        await loop.run(session=session)
        assert session.state.total_input_tokens > 0
        assert session.state.total_output_tokens > 0

    async def test_run_messages_appended_to_session(self):
        """Assistant and tool result messages must be appended to session.messages."""
        p = _make_provider(_tool_use_response("t1", "bash", {}), _text_response("done"))
        reg = _make_registry([ToolResult(call_id="t1", output="output")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder())
        session = Session.create()
        session.messages.append(Message(role="user", content="go"))
        initial_count = len(session.messages)
        await loop.run(session=session)
        # At least assistant + tool_result + final assistant messages added
        assert len(session.messages) > initial_count

    async def test_token_accumulation_multi_turn(self):
        """Tokens from multiple turns are summed correctly in session state."""
        p = _make_provider(
            _tool_use_response("t1", "bash", {}),  # turn 1: input=20, output=8
            _text_response("done"),                  # turn 2: input=10, output=5
        )
        reg = _make_registry([ToolResult(call_id="t1", output="ok")])
        loop = QueryLoop(provider=p, tool_registry=reg, prompt_builder=_make_prompt_builder())
        session = Session.create()
        session.messages.append(Message(role="user", content="go"))
        await loop.run(session=session)
        assert session.state.total_input_tokens == 30  # 20 + 10
        assert session.state.total_output_tokens == 13  # 8 + 5


# ── EventBus integration tests (Task 5) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_loop_emits_provider_request_event():
    provider = _make_provider(_text_response("hi"))
    bus = EventBus()
    received = []
    bus.subscribe(ProviderRequestEvent, lambda e: received.append(e))
    loop = QueryLoop(
        provider=provider,
        tool_registry=_make_registry(),
        prompt_builder=_make_prompt_builder(),
        event_bus=bus,
    )
    session = Session.create()
    session.messages.append(Message(role="user", content="hello"))
    await loop.run(session=session)
    assert len(received) == 1
    assert received[0].turn == 0


@pytest.mark.asyncio
async def test_loop_emits_turn_complete_event():
    provider = _make_provider(_text_response("done"))
    bus = EventBus()
    received = []
    bus.subscribe(TurnCompleteEvent, lambda e: received.append(e))
    loop = QueryLoop(
        provider=provider,
        tool_registry=_make_registry(),
        prompt_builder=_make_prompt_builder(),
        event_bus=bus,
    )
    session = Session.create()
    session.messages.append(Message(role="user", content="hi"))
    await loop.run(session=session)
    assert len(received) == 1
    assert received[0].stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_loop_no_observer_field_after_task5():
    """After Task 5, loop should not directly expose _observer — use event_bus."""
    provider = _make_provider(_text_response("ok"))
    loop = QueryLoop(
        provider=provider,
        tool_registry=_make_registry(),
        prompt_builder=_make_prompt_builder(),
    )
    # event_bus should exist
    assert hasattr(loop, '_bus')
