from __future__ import annotations
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from neoagent.config import NeoAgentConfig
from neoagent.agent import NeoAgent
from neoagent.core.types import Message, TextBlock, ConversationResult, Turn
from neoagent.tools.base import BaseTool
from neoagent.session import Session, SessionStorage
from pydantic import BaseModel
from neoagent.core.types import ToolResult


class TestConfig:
    def test_defaults(self):
        cfg = NeoAgentConfig(api_key="sk-test")
        assert cfg.model == "claude-sonnet-4-20250514"
        assert cfg.max_turns == 30
        assert cfg.context_budget == 0
        assert cfg.max_result_size == 50000

    def test_custom(self):
        cfg = NeoAgentConfig(api_key="sk-x", model="claude-opus-4-20250514", max_turns=10, context_budget=100_000, max_result_size=10000)
        assert cfg.model == "claude-opus-4-20250514"
        assert cfg.max_turns == 10


class DummyInput(BaseModel):
    x: str


class DummyTool(BaseTool):
    name: str = "dummy"
    description: str = "test"
    input_model: type[BaseModel] = DummyInput
    permission: str = "auto"

    async def execute(self, input):
        return ToolResult(call_id="x", output="ok")


def _make_mock_provider():
    mock_prov = MagicMock()
    mock_prov.get_context_window.return_value = 200_000
    return mock_prov


class TestNeoAgent:
    @patch("neoagent.agent._create_provider")
    def test_construction(self, mock_create):
        mock_create.return_value = _make_mock_provider()
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
        assert agent is not None

    @patch("neoagent.agent._create_provider")
    def test_register_tool(self, mock_create):
        mock_create.return_value = _make_mock_provider()
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
        agent.register_tool(DummyTool())
        schemas = agent._registry.get_schemas()
        assert any(s["name"] == "dummy" for s in schemas)

    @patch("neoagent.agent._create_provider")
    async def test_chat_returns_string(self, mock_create):
        mock_create.return_value = _make_mock_provider()
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
        # Mock the loop's run method
        turn = Turn(
            response=Message(role="assistant", content=[TextBlock(text="Hello!")]),
            tool_calls=[], tool_results=[], stop_reason="end_turn"
        )
        agent._loop.run = AsyncMock(return_value=ConversationResult(turns=[turn], reason="completed"))
        result = await agent.chat("Hi")
        assert result == "Hello!"

    @patch("neoagent.agent._create_provider")
    async def test_run_returns_conversation_result(self, mock_create):
        mock_create.return_value = _make_mock_provider()
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
        turn = Turn(
            response=Message(role="assistant", content="done"),
            tool_calls=[], tool_results=[], stop_reason="end_turn"
        )
        agent._loop.run = AsyncMock(return_value=ConversationResult(turns=[turn], reason="completed"))
        result = await agent.run([Message(role="user", content="go")])
        assert isinstance(result, ConversationResult)
        assert result.reason == "completed"

    # --- Session integration tests ---

    @patch("neoagent.agent._create_provider")
    def test_new_session_returns_session(self, mock_create):
        mock_create.return_value = _make_mock_provider()
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
        session = agent.new_session()
        assert isinstance(session, Session)
        assert session.id is not None
        assert session.messages == []

    @patch("neoagent.agent._create_provider")
    def test_new_session_explicit_id(self, mock_create):
        mock_create.return_value = _make_mock_provider()
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
        session = agent.new_session(session_id="my-custom-id")
        assert session.id == "my-custom-id"

    @patch("neoagent.agent._create_provider")
    def test_resume_without_storage_raises(self, mock_create):
        mock_create.return_value = _make_mock_provider()
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))  # no storage
        with pytest.raises(RuntimeError, match="No SessionStorage configured"):
            agent.resume("some-id")

    @patch("neoagent.agent._create_provider")
    def test_resume_with_storage(self, mock_create):
        mock_create.return_value = _make_mock_provider()
        # Build a fake storage that returns a known session
        fake_session = Session.create("known-id")
        mock_storage = MagicMock(spec=SessionStorage)
        mock_storage.load.return_value = fake_session

        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"), storage=mock_storage)
        loaded = agent.resume("known-id")
        mock_storage.load.assert_called_once_with("known-id")
        assert loaded.id == "known-id"

    @patch("neoagent.agent._create_provider")
    async def test_chat_backward_compat(self, mock_create):
        """chat(msg) with no session still works (temp session, no save)."""
        mock_create.return_value = _make_mock_provider()
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
        turn = Turn(
            response=Message(role="assistant", content=[TextBlock(text="Hi!")]),
            tool_calls=[], tool_results=[], stop_reason="end_turn",
        )
        agent._loop.run = AsyncMock(return_value=ConversationResult(turns=[turn], reason="completed"))
        result = await agent.chat("hello")
        assert result == "Hi!"

    @patch("neoagent.agent._create_provider")
    async def test_chat_with_session_saves_to_storage(self, mock_create):
        """When a session and storage are present, chat() auto-saves."""
        mock_create.return_value = _make_mock_provider()
        mock_storage = MagicMock(spec=SessionStorage)

        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"), storage=mock_storage)
        session = agent.new_session()
        turn = Turn(
            response=Message(role="assistant", content="ok"),
            tool_calls=[], tool_results=[], stop_reason="end_turn",
        )
        agent._loop.run = AsyncMock(return_value=ConversationResult(turns=[turn], reason="completed"))
        await agent.chat("hello", session=session)
        mock_storage.save.assert_called_once_with(session)

    @patch("neoagent.agent._create_provider")
    async def test_chat_without_storage_no_save(self, mock_create):
        """chat() with a session but no storage must not crash."""
        mock_create.return_value = _make_mock_provider()
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))  # no storage
        session = agent.new_session()
        turn = Turn(
            response=Message(role="assistant", content="ok"),
            tool_calls=[], tool_results=[], stop_reason="end_turn",
        )
        agent._loop.run = AsyncMock(return_value=ConversationResult(turns=[turn], reason="completed"))
        result = await agent.chat("hello", session=session)
        assert result == "ok"

    @patch("neoagent.agent._create_provider")
    def test_config_has_session_dir(self, mock_create):
        """NeoAgentConfig must expose session_dir field."""
        mock_create.return_value = _make_mock_provider()
        cfg = NeoAgentConfig(api_key="sk-test", session_dir=Path("/tmp/sessions"))
        assert cfg.session_dir == Path("/tmp/sessions")

    def test_config_session_dir_default_none(self):
        """session_dir defaults to None."""
        cfg = NeoAgentConfig(api_key="sk-test")
        assert cfg.session_dir is None
