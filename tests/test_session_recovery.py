"""
tests/test_session_recovery.py
Session Recovery 相关测试。
"""
from __future__ import annotations
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from neoagent.session import Session, JsonFileStorage
from neoagent.core.types import Message, TextBlock, ToolUseBlock, ToolResult
from neoagent.providers.base import Response


def _make_end_turn_response(text="done"):
    return Response(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
    )

def _make_tool_use_response(tool_name="echo", call_id="c1"):
    return Response(
        content=[ToolUseBlock(id=call_id, name=tool_name, input={"msg": "hi"})],
        stop_reason="tool_use",
        input_tokens=15,
        output_tokens=8,
    )


class TestQueryLoopAutoSave:

    @pytest.mark.asyncio
    async def test_auto_save_after_end_turn(self, tmp_path):
        """end_turn: save_if_storage() called at least once."""
        from neoagent.core.loop import QueryLoop
        from neoagent.tools.registry import ToolRegistry
        from neoagent.core.prompt import PromptBuilder, PromptSection

        storage = JsonFileStorage(tmp_path)
        session = Session.create(session_id="end-turn-save")
        session.bind_storage(storage)
        session.messages.append(Message(role="user", content="hello"))

        provider = MagicMock()
        provider.get_context_window.return_value = 200_000
        provider.create = AsyncMock(return_value=_make_end_turn_response("world"))

        registry = ToolRegistry()
        pb = PromptBuilder()
        pb.add_section(PromptSection(name="sys", content="test", priority=0, is_static=True))

        loop = QueryLoop(provider=provider, tool_registry=registry, prompt_builder=pb)

        call_count = [0]
        original = session.save_if_storage
        def counting_save():
            call_count[0] += 1
            original()
        session.save_if_storage = counting_save

        await loop.run(session=session)
        assert call_count[0] >= 1
        assert (tmp_path / "end-turn-save.json").exists()

    @pytest.mark.asyncio
    async def test_auto_save_after_tool_use_turn(self, tmp_path):
        """tool_use + end_turn: save called at least twice."""
        from neoagent.core.loop import QueryLoop
        from neoagent.tools.registry import ToolRegistry
        from neoagent.tools.executor import ToolExecutor
        from neoagent.tools.permission import PermissionChecker
        from neoagent.core.prompt import PromptBuilder, PromptSection
        from neoagent.events import EventBus
        from neoagent.tools.base import BaseTool

        storage = JsonFileStorage(tmp_path)
        session = Session.create(session_id="tool-use-save")
        session.bind_storage(storage)
        session.messages.append(Message(role="user", content="use tool"))

        provider = MagicMock()
        provider.get_context_window.return_value = 200_000
        provider.create = AsyncMock(side_effect=[
            _make_tool_use_response("mock_tool", "c1"),
            _make_end_turn_response("done"),
        ])

        registry = ToolRegistry()
        bus = EventBus()
        permission = PermissionChecker(auto_approve=True)
        executor = ToolExecutor(registry=registry, permission_checker=permission, event_bus=bus)

        class MockInput(BaseModel):
            msg: str = ""

        class MockTool(BaseTool):
            name = "mock_tool"
            description = "test"
            input_model: type[BaseModel] = MockInput
            permission = "auto"
            is_concurrent_safe = True

            async def execute(self, input: BaseModel) -> ToolResult:
                return ToolResult(call_id="", output="ok")

        registry.register(MockTool())

        pb = PromptBuilder()
        pb.add_section(PromptSection(name="sys", content="test", priority=0, is_static=True))

        loop = QueryLoop(provider=provider, tool_registry=registry, tool_executor=executor, prompt_builder=pb, event_bus=bus)

        save_calls = []
        original = session.save_if_storage
        def tracking_save():
            save_calls.append(1)
            original()
        session.save_if_storage = tracking_save

        await loop.run(session=session)
        assert len(save_calls) >= 2

    @pytest.mark.asyncio
    async def test_auto_save_skipped_without_storage(self):
        """No storage bound: loop completes without error."""
        from neoagent.core.loop import QueryLoop
        from neoagent.tools.registry import ToolRegistry
        from neoagent.core.prompt import PromptBuilder, PromptSection

        session = Session.create(session_id="no-storage")
        session.messages.append(Message(role="user", content="hello"))

        provider = MagicMock()
        provider.get_context_window.return_value = 200_000
        provider.create = AsyncMock(return_value=_make_end_turn_response("ok"))

        registry = ToolRegistry()
        pb = PromptBuilder()
        pb.add_section(PromptSection(name="sys", content="test", priority=0, is_static=True))

        loop = QueryLoop(provider=provider, tool_registry=registry, prompt_builder=pb)
        result = await loop.run(session=session)
        assert result.reason == "completed"


# ── Helpers shared by NeoAgent tests ─────────────────────────────────────────

def _make_agent(tmp_path=None, storage=None):
    """Create a NeoAgent with a mock provider."""
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig

    config = NeoAgentConfig(
        provider="anthropic",
        api_key="test-key",
        model="claude-3-5-haiku-20241022",
        session_dir=tmp_path,
    )
    agent = NeoAgent(config=config, storage=storage)
    # Replace provider with mock
    agent._provider = MagicMock()
    agent._provider.get_context_window.return_value = 200_000
    agent._provider.create = AsyncMock(return_value=_make_end_turn_response("ok"))
    # Rebuild loop with mock provider
    from neoagent.core.loop import QueryLoop
    agent._loop = QueryLoop(
        provider=agent._provider,
        tool_registry=agent._registry,
        tool_executor=agent._executor,
        prompt_builder=agent._prompt_builder,
        event_bus=agent._event_bus,
        hook_manager=agent._hook_manager,
        deferred_registry=agent._deferred_registry,
    )
    return agent


class TestNeoAgentAutoSave:

    @pytest.mark.asyncio
    async def test_run_auto_saves_at_end(self, tmp_path):
        """run() with a named session and storage saves a file after completion."""
        agent = _make_agent(tmp_path=tmp_path)
        session = Session.create(session_id="run-save-test")

        await agent.run(
            messages=[Message(role="user", content="hello")],
            session=session,
        )

        assert (tmp_path / "run-save-test.json").exists()

    @pytest.mark.asyncio
    async def test_chat_auto_saves_per_turn(self, tmp_path):
        """chat() with a named session and storage persists the session."""
        agent = _make_agent(tmp_path=tmp_path)
        session = Session.create(session_id="chat-save-test")

        await agent.chat("hello", session=session)

        assert (tmp_path / "chat-save-test.json").exists()


class TestResumeValidation:

    def _make_saved_session(self, storage, session_id, workspace_path=None, updated_at=None):
        """Helper: create and save a session with optional metadata/updated_at."""
        session = Session.create(session_id=session_id)
        if workspace_path is not None:
            session.metadata["workspace_path"] = workspace_path
        if updated_at is not None:
            session.updated_at = updated_at
        storage.save(session)
        return session

    def test_resume_validation_warns_on_missing_workspace(self, tmp_path):
        """validate=True emits SessionResumeWarningEvent with reason=workspace_missing."""
        from neoagent.events import SessionResumeWarningEvent

        storage = JsonFileStorage(tmp_path)
        self._make_saved_session(
            storage,
            session_id="missing-ws",
            workspace_path="/nonexistent/path/that/does/not/exist",
        )

        agent = _make_agent(storage=storage)
        warnings = []
        agent.event_bus.subscribe(SessionResumeWarningEvent, warnings.append)

        agent.resume("missing-ws", validate=True)

        assert len(warnings) == 1
        assert warnings[0].reason == "workspace_missing"
        assert warnings[0].session_id == "missing-ws"
        assert "/nonexistent/path/that/does/not/exist" in warnings[0].details

    def test_resume_validation_warns_on_stale_session(self, tmp_path):
        """validate=True emits SessionResumeWarningEvent with reason=stale_session."""
        from neoagent.events import SessionResumeWarningEvent

        storage = JsonFileStorage(tmp_path)
        stale_time = datetime.now() - timedelta(hours=25)
        self._make_saved_session(
            storage,
            session_id="stale-sess",
            updated_at=stale_time,
        )

        agent = _make_agent(storage=storage)
        warnings = []
        agent.event_bus.subscribe(SessionResumeWarningEvent, warnings.append)

        agent.resume("stale-sess", validate=True)

        assert len(warnings) == 1
        assert warnings[0].reason == "stale_session"
        assert warnings[0].session_id == "stale-sess"
        assert "h ago" in warnings[0].details

    def test_resume_validation_passes_when_ok(self, tmp_path):
        """validate=True emits no warning when session is fresh and workspace exists."""
        from neoagent.events import SessionResumeWarningEvent

        storage = JsonFileStorage(tmp_path)
        self._make_saved_session(
            storage,
            session_id="ok-sess",
            workspace_path=str(tmp_path),  # tmp_path exists
        )

        agent = _make_agent(storage=storage)
        warnings = []
        agent.event_bus.subscribe(SessionResumeWarningEvent, warnings.append)

        agent.resume("ok-sess", validate=True)

        assert warnings == []

    def test_resume_validate_false_skips_check(self, tmp_path):
        """validate=False skips all validation, even for stale/missing workspace."""
        from neoagent.events import SessionResumeWarningEvent

        storage = JsonFileStorage(tmp_path)
        stale_time = datetime.now() - timedelta(hours=48)
        self._make_saved_session(
            storage,
            session_id="skip-val",
            workspace_path="/nonexistent/path",
            updated_at=stale_time,
        )

        agent = _make_agent(storage=storage)
        warnings = []
        agent.event_bus.subscribe(SessionResumeWarningEvent, warnings.append)

        agent.resume("skip-val", validate=False)

        assert warnings == []

    def test_resume_no_storage_raises(self):
        """resume() raises RuntimeError when no storage is configured."""
        from neoagent.agent import NeoAgent
        from neoagent.config import NeoAgentConfig

        config = NeoAgentConfig(
            provider="anthropic",
            api_key="test-key",
            model="claude-3-5-haiku-20241022",
        )
        agent = NeoAgent(config=config)

        with pytest.raises(RuntimeError, match="No SessionStorage configured"):
            agent.resume("any-id")
