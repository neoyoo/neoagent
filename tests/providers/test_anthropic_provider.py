from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from neoagent.core.types import Message, TextBlock, ToolUseBlock
from neoagent.providers.anthropic import AnthropicProvider
from neoagent.providers.base import Provider, Response

class TestProviderABC:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Provider()

    def test_concrete_must_implement_create(self) -> None:
        class BadProvider(Provider):
            def get_context_window(self) -> int:
                return 200_000
        with pytest.raises(TypeError):
            BadProvider()

    def test_concrete_must_implement_get_context_window(self) -> None:
        class BadProvider(Provider):
            async def create(self, system, messages, tools) -> Response:
                ...
        with pytest.raises(TypeError):
            BadProvider()

class TestResponse:
    def test_construction_text_only(self) -> None:
        blocks = [TextBlock(text="hello")]
        resp = Response(content=blocks, stop_reason="end_turn", input_tokens=10, output_tokens=5)
        assert resp.stop_reason == "end_turn"

    def test_tool_use_blocks_property(self) -> None:
        blocks = [
            TextBlock(text="Let me read that."),
            ToolUseBlock(id="tu_1", name="read", input={"path": "/f"}),
            ToolUseBlock(id="tu_2", name="grep", input={"pattern": "TODO"}),
        ]
        resp = Response(content=blocks, stop_reason="tool_use", input_tokens=30, output_tokens=20)
        assert len(resp.tool_use_blocks) == 2

    def test_text_content_property(self) -> None:
        blocks = [TextBlock(text="first"), TextBlock(text="second")]
        resp = Response(content=blocks, stop_reason="end_turn", input_tokens=5, output_tokens=3)
        assert resp.text_content == "first\nsecond"

    def test_text_content_empty(self) -> None:
        blocks = [ToolUseBlock(id="tu_1", name="bash", input={})]
        resp = Response(content=blocks, stop_reason="tool_use", input_tokens=5, output_tokens=3)
        assert resp.text_content == ""

def _make_fake_anthropic_response(stop_reason="end_turn", blocks=None, input_tokens=10, output_tokens=5):
    if blocks is None:
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I'm done."
        blocks = [text_block]
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = blocks
    resp.usage = usage
    return resp

class TestAnthropicProviderConstruction:
    def test_default_model(self) -> None:
        with patch("neoagent.providers.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="sk-test")
            assert provider.model == "claude-sonnet-4-20250514"

    def test_is_provider_subclass(self) -> None:
        with patch("neoagent.providers.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="sk-test")
            assert isinstance(provider, Provider)

    def test_get_context_window(self) -> None:
        with patch("neoagent.providers.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="sk-test")
            assert provider.get_context_window() == 200_000

class TestAnthropicProviderCreate:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = AsyncMock()
        return client

    @pytest.fixture
    def provider(self, mock_client):
        with patch("neoagent.providers.anthropic.AsyncAnthropic", return_value=mock_client):
            p = AnthropicProvider(api_key="sk-test")
        p._client = mock_client
        return p

    async def test_create_text_response(self, provider, mock_client):
        mock_client.messages.create.return_value = _make_fake_anthropic_response()
        messages = [Message(role="user", content="hello")]
        response = await provider.create(system="You are helpful.", messages=messages, tools=[])
        assert isinstance(response, Response)
        assert response.stop_reason == "end_turn"
        assert len(response.content) == 1
        assert isinstance(response.content[0], TextBlock)

    async def test_create_tool_use_response(self, provider, mock_client):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tu_abc"
        tool_block.name = "read"
        tool_block.input = {"path": "/tmp/file.txt"}
        mock_client.messages.create.return_value = _make_fake_anthropic_response(
            stop_reason="tool_use", blocks=[tool_block])
        messages = [Message(role="user", content="read file")]
        response = await provider.create(system="sys", messages=messages, tools=[{"name": "read", "description": "Read", "input_schema": {}}])
        assert response.stop_reason == "tool_use"
        assert len(response.tool_use_blocks) == 1
        assert response.tool_use_blocks[0].name == "read"

    async def test_create_passes_system_and_tools(self, provider, mock_client):
        mock_client.messages.create.return_value = _make_fake_anthropic_response()
        tools = [{"name": "bash", "description": "Run bash", "input_schema": {}}]
        await provider.create(system="Custom.", messages=[Message(role="user", content="hi")], tools=tools)
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Custom."
        assert call_kwargs["tools"] == tools

    async def test_create_message_serialization(self, provider, mock_client):
        mock_client.messages.create.return_value = _make_fake_anthropic_response()
        await provider.create(system="sys", messages=[Message(role="user", content="hello")], tools=[])
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "hello"}]

    async def test_unknown_block_skipped(self, provider, mock_client):
        unknown = MagicMock(); unknown.type = "image"
        text = MagicMock(); text.type = "text"; text.text = "hi"
        mock_client.messages.create.return_value = _make_fake_anthropic_response(blocks=[unknown, text])
        response = await provider.create(system="sys", messages=[Message(role="user", content="x")], tools=[])
        assert len(response.content) == 1
        assert isinstance(response.content[0], TextBlock)
