from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from neoagent.config import NeoAgentConfig
from neoagent.agent import NeoAgent
from neoagent.core.types import Message, TextBlock, ConversationResult, Turn
from neoagent.tools.base import BaseTool
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


class TestNeoAgent:
    @patch("neoagent.agent.AnthropicProvider")
    def test_construction(self, mock_prov_cls):
        mock_prov_cls.return_value = MagicMock()
        mock_prov_cls.return_value.get_context_window.return_value = 200_000
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
        assert agent is not None

    @patch("neoagent.agent.AnthropicProvider")
    def test_register_tool(self, mock_prov_cls):
        mock_prov_cls.return_value = MagicMock()
        mock_prov_cls.return_value.get_context_window.return_value = 200_000
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
        agent.register_tool(DummyTool())
        schemas = agent._registry.get_schemas()
        assert any(s["name"] == "dummy" for s in schemas)

    @patch("neoagent.agent.AnthropicProvider")
    async def test_chat_returns_string(self, mock_prov_cls):
        mock_prov = MagicMock()
        mock_prov.get_context_window.return_value = 200_000
        mock_prov_cls.return_value = mock_prov
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
        # Mock the loop's run method
        turn = Turn(
            response=Message(role="assistant", content=[TextBlock(text="Hello!")]),
            tool_calls=[], tool_results=[], stop_reason="end_turn"
        )
        agent._loop.run = AsyncMock(return_value=ConversationResult(turns=[turn], reason="completed"))
        result = await agent.chat("Hi")
        assert result == "Hello!"

    @patch("neoagent.agent.AnthropicProvider")
    async def test_run_returns_conversation_result(self, mock_prov_cls):
        mock_prov = MagicMock()
        mock_prov.get_context_window.return_value = 200_000
        mock_prov_cls.return_value = mock_prov
        agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
        turn = Turn(
            response=Message(role="assistant", content="done"),
            tool_calls=[], tool_results=[], stop_reason="end_turn"
        )
        agent._loop.run = AsyncMock(return_value=ConversationResult(turns=[turn], reason="completed"))
        result = await agent.run([Message(role="user", content="go")])
        assert isinstance(result, ConversationResult)
        assert result.reason == "completed"
