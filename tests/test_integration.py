from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock
import pytest
from pydantic import BaseModel

from neoagent.core.types import (
    ConversationResult, Message, TextBlock, ToolCall, ToolResult,
    ToolResultBlock, ToolUseBlock, Turn,
)
from neoagent.core.loop import QueryLoop
from neoagent.core.prompt import PromptBuilder, PromptSection
from neoagent.core.compress import ContextCompressor
from neoagent.tools.base import BaseTool
from neoagent.tools.registry import ToolRegistry
from neoagent.providers.base import Provider, Response


# --- MockProvider ---

class MockProvider(Provider):
    def __init__(self, responses: list[Response]):
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[dict] = []

    async def create(self, system: str, messages: list, tools: list, **kwargs) -> Response:
        self.calls.append({"system": system, "messages": messages, "tools": tools, **kwargs})
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    def get_context_window(self) -> int:
        return 200_000


def _text_resp(text: str) -> Response:
    return Response(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        input_tokens=10, output_tokens=5,
    )

def _tool_use_resp(tool_id: str, tool_name: str, tool_input: dict) -> Response:
    return Response(
        content=[ToolUseBlock(id=tool_id, name=tool_name, input=tool_input)],
        stop_reason="tool_use",
        input_tokens=20, output_tokens=10,
    )


# --- EchoTool for testing ---

class EchoInput(BaseModel):
    text: str

class EchoTool(BaseTool):
    name: str = "echo"
    description: str = "Echo back the input text"
    input_model: type[BaseModel] = EchoInput
    permission: str = "auto"
    is_concurrent_safe: bool = True

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, EchoInput)
        return ToolResult(call_id="", output=f"echoed: {input.text}")


# --- Scene 1: Simple conversation ---

class TestSimpleConversation:
    async def test_single_turn_no_tools(self):
        provider = MockProvider([_text_resp("Hello there!")])
        registry = ToolRegistry()
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="sys", content="You are helpful.", priority=0))
        loop = QueryLoop(provider=provider, tool_registry=registry, prompt_builder=builder)

        result = await loop.run([Message(role="user", content="Hi")])

        assert isinstance(result, ConversationResult)
        assert result.reason == "completed"
        assert len(result.turns) == 1
        assert result.turns[0].stop_reason == "end_turn"
        assert len(provider.calls) == 1
        assert provider.calls[0]["system"] == "# sys\nYou are helpful."

    async def test_prompt_builder_sections_in_system(self):
        provider = MockProvider([_text_resp("ok")])
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="identity", content="I am neoagent.", priority=0, is_static=True))
        builder.add_section(PromptSection(name="rules", content="Be concise.", priority=1, is_static=True))
        loop = QueryLoop(provider=provider, tool_registry=ToolRegistry(), prompt_builder=builder)
        await loop.run([Message(role="user", content="go")])
        system = provider.calls[0]["system"]
        assert "# identity" in system
        assert "# rules" in system
        assert system.index("# identity") < system.index("# rules")


# --- Scene 2: Tool call flow ---

class TestToolCallFlow:
    async def test_tool_use_then_end(self, tmp_path):
        # Create a real file for ReadTool
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\n")

        from neoagent.tools.builtin.read import ReadTool

        provider = MockProvider([
            _tool_use_resp("tu_1", "read", {"file_path": str(test_file)}),
            _text_resp("I read the file."),
        ])
        registry = ToolRegistry()
        registry.register(ReadTool(allowed_directories=[tmp_path]))
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="sys", content="Help.", priority=0))
        loop = QueryLoop(provider=provider, tool_registry=registry, prompt_builder=builder)

        result = await loop.run([Message(role="user", content="read the file")])

        assert result.reason == "completed"
        assert len(result.turns) == 2
        # First turn: tool use
        assert result.turns[0].stop_reason == "tool_use"
        assert result.turns[0].tool_calls[0].name == "read"
        assert "line1" in result.turns[0].tool_results[0].output
        # Second turn: end
        assert result.turns[1].stop_reason == "end_turn"

    async def test_tool_result_fed_back_to_provider(self):
        provider = MockProvider([
            _tool_use_resp("tu_1", "echo", {"text": "hello"}),
            _text_resp("done"),
        ])
        registry = ToolRegistry()
        registry.register(EchoTool())
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="sys", content="x", priority=0))
        loop = QueryLoop(provider=provider, tool_registry=registry, prompt_builder=builder)
        await loop.run([Message(role="user", content="echo hello")])

        # Second call should have tool_result in messages
        second_msgs = provider.calls[1]["messages"]
        last_msg = second_msgs[-1]
        assert last_msg.role == "user"
        assert isinstance(last_msg.content, list)
        assert any(isinstance(b, ToolResultBlock) for b in last_msg.content)


# --- Scene 3: max_turns ---

class TestMaxTurns:
    async def test_loop_exits_at_max_turns(self):
        # Provider always returns tool_use
        responses = [_tool_use_resp(f"tu_{i}", "echo", {"text": "x"}) for i in range(10)]
        provider = MockProvider(responses)
        registry = ToolRegistry()
        registry.register(EchoTool())
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="sys", content="x", priority=0))
        loop = QueryLoop(provider=provider, tool_registry=registry, prompt_builder=builder, max_turns=3)

        result = await loop.run([Message(role="user", content="go")])

        assert result.reason == "max_turns"
        assert len(result.turns) == 3
        assert len(provider.calls) == 3


# --- Scene 4: Compression threshold ---

class TestCompressionThreshold:
    def test_should_compress_detection(self):
        provider = MockProvider([])
        compressor = ContextCompressor(provider=provider)
        # Short message, huge budget → no compress
        assert compressor.should_compress(
            [Message(role="user", content="Hi")], [], context_budget=100_000
        ) is False
        # Long message, tiny budget → compress
        assert compressor.should_compress(
            [Message(role="user", content="word " * 500)], [], context_budget=50
        ) is True

    def test_tools_schema_counted(self):
        provider = MockProvider([])
        compressor = ContextCompressor(provider=provider)
        big_schemas = [{"name": f"t{i}", "description": "x" * 200, "input_schema": {}} for i in range(10)]
        assert compressor.should_compress(
            [Message(role="user", content="short")], big_schemas, context_budget=50
        ) is True


# --- Full suite run ---

class TestFullSuiteRegression:
    async def test_all_builtin_tools_register(self):
        """All 6 builtin tools can be registered without conflict."""
        from neoagent.tools.builtin.read import ReadTool
        from neoagent.tools.builtin.write import WriteTool
        from neoagent.tools.builtin.edit import EditTool
        from neoagent.tools.builtin.bash import BashTool
        from neoagent.tools.builtin.grep import GrepTool
        from neoagent.tools.builtin.glob import GlobTool

        registry = ToolRegistry()
        for tool_cls in [ReadTool, WriteTool, EditTool, BashTool, GrepTool, GlobTool]:
            registry.register(tool_cls())

        schemas = registry.get_schemas()
        names = {s["name"] for s in schemas}
        assert names == {"read", "write", "edit", "bash", "grep", "glob"}
