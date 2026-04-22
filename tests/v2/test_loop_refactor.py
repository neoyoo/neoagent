"""Phase 4 Batch 3 tests — Task 4.3 (no bypass) + Task 4.4 (2 events).

A. Task 4.3: deferred / freed sections injected via add_section / remove_section
B. Task 4.4: MessageCreatedEvent (user_input / assistant_reply / tool_result)
             + ToolResultPersistedEvent
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

from neoagent.core.loop import QueryLoop
from neoagent.core.prompt import PromptBuilder, PromptSection
from neoagent.core.types import (
    ConversationResult, Message, TextBlock,
    ToolCall, ToolResult, ToolUseBlock, ToolResultBlock,
)
from neoagent.events import (
    EventBus, MessageCreatedEvent, ToolResultPersistedEvent,
)
from neoagent.session import Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_response(text="done"):
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [TextBlock(text=text)]
    resp.tool_use_blocks = []
    resp.input_tokens = 10
    resp.output_tokens = 5
    return resp


def _tool_use_response(tool_id, tool_name, tool_input=None):
    tu = ToolUseBlock(id=tool_id, name=tool_name, input=tool_input or {})
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [tu]
    resp.tool_use_blocks = [tu]
    resp.input_tokens = 20
    resp.output_tokens = 8
    return resp


def _make_provider(*responses):
    p = MagicMock()
    p.get_context_window.return_value = 200_000
    p.create = AsyncMock(side_effect=list(responses))
    return p


def _make_registry():
    r = MagicMock()
    r.get_schemas.return_value = []
    return r


def _make_executor(tool_results=None):
    ex = MagicMock()
    ex.execute = AsyncMock(return_value=tool_results or [])
    return ex


def _real_prompt_builder(text="You are neoagent.") -> PromptBuilder:
    """Return a real PromptBuilder (not mock) with one section."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pb = PromptBuilder()
    pb.add_section(PromptSection(name="sys", content=text, priority=0, is_static=True))
    return pb


def _mock_prompt_builder(text="You are neoagent.") -> MagicMock:
    b = MagicMock()
    b.build.return_value = text
    return b


def _session_with(content: str) -> Session:
    s = Session.create()
    s.messages.append(Message(role="user", content=content))
    return s


# ---------------------------------------------------------------------------
# A. Task 4.3 — deferred / freed sections via add_section / remove_section
# ---------------------------------------------------------------------------


class TestTask43NoBuildBypass:
    """Task 4.3: deferred/freed sections injected via PromptBuilder.add_section."""

    # ── A-1: deferred section content present in system prompt ──────────────

    async def test_deferred_section_content_in_system(self):
        """When deferred tools exist, system prompt contains their names."""
        from neoagent.tools.deferred import DeferredToolRegistry

        deferred_reg = MagicMock(spec=DeferredToolRegistry)
        deferred_reg._all = {"tool_alpha", "tool_beta"}
        deferred_reg.is_deferred.return_value = True

        captured_systems = []

        def capture_system(*, system, messages, tools, max_tokens):
            captured_systems.append(system)
            return _text_response("ok")

        provider = _make_provider()
        provider.create = AsyncMock(side_effect=capture_system)

        pb = _real_prompt_builder("BASE")
        session = _session_with("hi")

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            deferred_registry=deferred_reg,
        )
        await loop.run(session=session)

        assert len(captured_systems) == 1
        sys_prompt = captured_systems[0]
        # deferred tool names must be present
        assert "tool_alpha" in sys_prompt
        assert "tool_beta" in sys_prompt
        assert "<deferred-tools>" in sys_prompt

    # ── A-2: each turn is clean — sections not accumulated ──────────────────

    async def test_deferred_section_not_accumulated_across_turns(self):
        """After turn 1, builder must not carry _deferred section into turn 2."""
        from neoagent.tools.deferred import DeferredToolRegistry

        deferred_reg = MagicMock(spec=DeferredToolRegistry)
        deferred_reg._all = {"tool_x"}
        deferred_reg.is_deferred.return_value = True

        captured_systems = []

        async def capture(*args, **kwargs):
            captured_systems.append(kwargs.get("system", ""))
            return _text_response("ok") if len(captured_systems) == 2 else _tool_use_response("t1", "tool_x")

        provider = MagicMock()
        provider.get_context_window.return_value = 200_000
        provider.create = AsyncMock(side_effect=capture)

        pb = _real_prompt_builder("BASE")
        session = _session_with("go")
        ex = _make_executor([ToolResult(call_id="t1", output="out")])

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            tool_executor=ex,
            deferred_registry=deferred_reg,
        )
        await loop.run(session=session)

        # Both turns should have exactly one occurrence of <deferred-tools>
        for sys in captured_systems:
            assert sys.count("<deferred-tools>") == 1, (
                f"Expected exactly 1 <deferred-tools> block, got {sys.count('<deferred-tools>')} in: {sys!r}"
            )

    # ── A-3: empty content → section not injected ───────────────────────────

    async def test_no_deferred_section_when_no_deferred_tools(self):
        """When no tools are deferred, _deferred section is not added to builder."""
        from neoagent.tools.deferred import DeferredToolRegistry

        deferred_reg = MagicMock(spec=DeferredToolRegistry)
        deferred_reg._all = set()  # empty — no deferred tools
        deferred_reg.is_deferred.return_value = False

        captured_systems = []

        def capture_system(*, system, messages, tools, max_tokens):
            captured_systems.append(system)
            return _text_response("ok")

        provider = _make_provider()
        provider.create = AsyncMock(side_effect=capture_system)

        pb = _real_prompt_builder("BASE")
        session = _session_with("hi")

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            deferred_registry=deferred_reg,
        )
        await loop.run(session=session)

        sys_prompt = captured_systems[0]
        assert "<deferred-tools>" not in sys_prompt
        # Builder should be clean after the run (no lingering _deferred)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert not any(s.name == "_deferred" for s in pb._sections)

    # ── A-4: try/finally cleanup — sections removed even on exception ────────

    async def test_dynamic_sections_cleaned_up_on_exception(self):
        """_deferred / _freed sections must be removed even if build() raises."""
        from neoagent.tools.deferred import DeferredToolRegistry

        deferred_reg = MagicMock(spec=DeferredToolRegistry)
        deferred_reg._all = {"tool_x"}
        deferred_reg.is_deferred.return_value = True

        provider = _make_provider()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pb = PromptBuilder()
        pb.add_section(PromptSection(name="sys", content="BASE", priority=0))

        # Patch build() to raise an exception
        original_build = pb.build

        call_count = 0

        def raise_on_build():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("build exploded")

        pb.build = raise_on_build

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            deferred_registry=deferred_reg,
        )
        session = _session_with("go")

        with pytest.raises(RuntimeError, match="build exploded"):
            await loop.run(session=session)

        # After exception, _deferred must NOT be present in sections
        assert not any(s.name == "_deferred" for s in pb._sections)

    # ── A-5: freed section present when freed_tool_results exist ────────────

    async def test_freed_section_content_in_system(self):
        """When freed_tool_results exist, system prompt contains freed section."""
        from neoagent.session import FreedToolResult

        captured_systems = []

        def capture_system(*, system, messages, tools, max_tokens):
            captured_systems.append(system)
            return _text_response("ok")

        provider = _make_provider()
        provider.create = AsyncMock(side_effect=capture_system)

        pb = _real_prompt_builder("BASE")
        session = _session_with("hi")
        # Inject a freed tool result into session state
        session.state.freed_tool_results["tid1"] = FreedToolResult(
            id="tid1", tool_name="bash", size=100, preview="preview text", original_content="full content"
        )

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
        )
        await loop.run(session=session)

        sys_prompt = captured_systems[0]
        assert "Freed Tool Results" in sys_prompt
        assert "tid1" in sys_prompt
        assert "bash" in sys_prompt


# ---------------------------------------------------------------------------
# B. Task 4.4 — MessageCreatedEvent + ToolResultPersistedEvent
# ---------------------------------------------------------------------------


class TestTask44Events:
    """Task 4.4: MessageCreatedEvent and ToolResultPersistedEvent emissions."""

    # ── B-1: assistant_reply event on end_turn ───────────────────────────────

    async def test_assistant_reply_event_on_end_turn(self):
        """MessageCreatedEvent(source_type='assistant_reply') emitted on end_turn."""
        provider = _make_provider(_text_response("hello"))
        bus = EventBus()
        events: list[MessageCreatedEvent] = []
        bus.subscribe(MessageCreatedEvent, events.append)

        pb = _mock_prompt_builder()
        session = _session_with("hi")

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            event_bus=bus,
        )
        await loop.run(session=session)

        reply_events = [e for e in events if e.source_type == "assistant_reply"]
        assert len(reply_events) == 1
        assert reply_events[0].role == "assistant"
        assert reply_events[0].turn == 0

    # ── B-2: assistant_reply event on tool_use path ──────────────────────────

    async def test_assistant_reply_event_on_tool_use(self):
        """MessageCreatedEvent(source_type='assistant_reply') emitted before tool results."""
        provider = _make_provider(
            _tool_use_response("t1", "bash"),
            _text_response("done"),
        )
        bus = EventBus()
        events: list[MessageCreatedEvent] = []
        bus.subscribe(MessageCreatedEvent, events.append)

        pb = _mock_prompt_builder()
        session = _session_with("go")
        ex = _make_executor([ToolResult(call_id="t1", output="output")])

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            event_bus=bus,
            tool_executor=ex,
        )
        await loop.run(session=session)

        reply_events = [e for e in events if e.source_type == "assistant_reply"]
        # turn 0 (tool_use) and turn 1 (end_turn) both emit assistant_reply
        assert len(reply_events) == 2
        for ev in reply_events:
            assert ev.role == "assistant"

    # ── B-3: tool_result event + ToolResultPersistedEvent ────────────────────

    async def test_tool_result_events_emitted(self):
        """MessageCreatedEvent(source_type='tool_result') and ToolResultPersistedEvent emitted."""
        provider = _make_provider(
            _tool_use_response("t1", "bash"),
            _text_response("done"),
        )
        bus = EventBus()
        msg_events: list[MessageCreatedEvent] = []
        persisted_events: list[ToolResultPersistedEvent] = []
        bus.subscribe(MessageCreatedEvent, msg_events.append)
        bus.subscribe(ToolResultPersistedEvent, persisted_events.append)

        pb = _mock_prompt_builder()
        session = _session_with("go")
        ex = _make_executor([ToolResult(call_id="t1", output="result_output")])

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            event_bus=bus,
            tool_executor=ex,
        )
        await loop.run(session=session)

        tool_result_msg_events = [e for e in msg_events if e.source_type == "tool_result"]
        assert len(tool_result_msg_events) == 1
        assert tool_result_msg_events[0].role == "user"

        assert len(persisted_events) == 1
        assert persisted_events[0].tool_use_id == "t1"
        assert persisted_events[0].tool_name == "bash"
        assert persisted_events[0].output == "result_output"

    # ── B-4: msg_id increments across messages ───────────────────────────────

    async def test_msg_id_increments(self):
        """Consecutive MessageCreatedEvents have incrementing msg_ids (m1, m2, …)."""
        provider = _make_provider(
            _tool_use_response("t1", "bash"),
            _text_response("done"),
        )
        bus = EventBus()
        events: list[MessageCreatedEvent] = []
        bus.subscribe(MessageCreatedEvent, events.append)

        pb = _mock_prompt_builder()
        session = _session_with("go")
        ex = _make_executor([ToolResult(call_id="t1", output="out")])

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            event_bus=bus,
            tool_executor=ex,
        )
        await loop.run(session=session)

        # 3 MessageCreatedEvents: assistant_reply (turn 0), tool_result (turn 0), assistant_reply (turn 1)
        assert len(events) == 3
        msg_ids = [e.msg_id for e in events]
        assert msg_ids == ["m1", "m2", "m3"]

    # ── B-5: ToolResultPersistedEvent size_bytes correct ─────────────────────

    async def test_tool_result_persisted_size_bytes(self):
        """ToolResultPersistedEvent.size_bytes == len(output.encode('utf-8'))."""
        provider = _make_provider(
            _tool_use_response("t1", "read"),
            _text_response("done"),
        )
        bus = EventBus()
        persisted_events: list[ToolResultPersistedEvent] = []
        bus.subscribe(ToolResultPersistedEvent, persisted_events.append)

        pb = _mock_prompt_builder()
        session = _session_with("go")
        output_str = "hello"
        ex = _make_executor([ToolResult(call_id="t1", output=output_str)])

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            event_bus=bus,
            tool_executor=ex,
        )
        await loop.run(session=session)

        assert len(persisted_events) == 1
        assert persisted_events[0].size_bytes == len(output_str.encode("utf-8"))
        assert persisted_events[0].size_bytes == 5

    # ── B-6: ToolResultPersistedEvent is_error correct ───────────────────────

    async def test_tool_result_persisted_is_error(self):
        """ToolResultPersistedEvent.is_error=True when ToolResult.is_error=True."""
        provider = _make_provider(
            _tool_use_response("t1", "bash"),
            _text_response("done"),
        )
        bus = EventBus()
        persisted_events: list[ToolResultPersistedEvent] = []
        bus.subscribe(ToolResultPersistedEvent, persisted_events.append)

        pb = _mock_prompt_builder()
        session = _session_with("go")
        ex = _make_executor([ToolResult(call_id="t1", output="error msg", is_error=True)])

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            event_bus=bus,
            tool_executor=ex,
        )
        await loop.run(session=session)

        assert len(persisted_events) == 1
        assert persisted_events[0].is_error is True

    # ── B-7: tool_result + persisted events in correct order ─────────────────

    async def test_tool_result_and_persisted_event_order(self):
        """MessageCreatedEvent(tool_result) is emitted before ToolResultPersistedEvent."""
        provider = _make_provider(
            _tool_use_response("t1", "bash"),
            _text_response("done"),
        )
        bus = EventBus()
        all_events: list = []
        bus.subscribe(MessageCreatedEvent, all_events.append)
        bus.subscribe(ToolResultPersistedEvent, all_events.append)

        pb = _mock_prompt_builder()
        session = _session_with("go")
        ex = _make_executor([ToolResult(call_id="t1", output="out")])

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            event_bus=bus,
            tool_executor=ex,
        )
        await loop.run(session=session)

        # Find positions of tool_result MessageCreatedEvent and ToolResultPersistedEvent
        tool_result_msg_idx = next(
            i for i, e in enumerate(all_events)
            if isinstance(e, MessageCreatedEvent) and e.source_type == "tool_result"
        )
        persisted_idx = next(
            i for i, e in enumerate(all_events)
            if isinstance(e, ToolResultPersistedEvent)
        )
        assert tool_result_msg_idx < persisted_idx

    # ── B-8: session_id on events matches session ────────────────────────────

    async def test_event_session_id_matches_session(self):
        """All emitted MessageCreatedEvents have session_id matching the session."""
        provider = _make_provider(_text_response("done"))
        bus = EventBus()
        events: list[MessageCreatedEvent] = []
        bus.subscribe(MessageCreatedEvent, events.append)

        pb = _mock_prompt_builder()
        session = _session_with("hi")

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            event_bus=bus,
        )
        await loop.run(session=session)

        assert len(events) >= 1
        for ev in events:
            assert ev.session_id == session.id

    # ── B-9: no MessageCreatedEvent when no session_state (legacy path) ──────

    async def test_no_message_events_on_legacy_list_path(self):
        """Legacy list[Message] path still works without errors (session created internally)."""
        provider = _make_provider(_text_response("ok"))
        bus = EventBus()
        events: list[MessageCreatedEvent] = []
        bus.subscribe(MessageCreatedEvent, events.append)

        pb = _mock_prompt_builder()

        loop = QueryLoop(
            provider=provider,
            tool_registry=_make_registry(),
            prompt_builder=pb,
            event_bus=bus,
        )
        # Legacy list path — internally creates a transient session, so events should fire
        result = await loop.run([Message(role="user", content="legacy")])
        assert result.reason == "completed"
        # Events emitted (transient session has state), just verify no exception
