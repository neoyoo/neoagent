from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from neoagent.core.types import Message, TextBlock, ToolUseBlock, ToolResultBlock
from neoagent.providers.openai import (
    OpenAIProvider, _serialize_messages, _convert_tools, _parse_response,
    _STOP_REASON_MAP,
)
from neoagent.providers.base import Provider, Response

# --- conftest-level stub for openai SDK ---
# The tests/providers/conftest.py already stubs anthropic.
# We need to stub openai similarly if not installed.

class TestOpenAIProviderConstruction:
    def test_default_model(self):
        with patch("neoagent.providers.openai.AsyncOpenAI"):
            p = OpenAIProvider(api_key="sk-test")
            assert p.model == "gpt-4o"

    def test_custom_model(self):
        with patch("neoagent.providers.openai.AsyncOpenAI"):
            p = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
            assert p.model == "gpt-4o-mini"

    def test_is_provider_subclass(self):
        with patch("neoagent.providers.openai.AsyncOpenAI"):
            assert isinstance(OpenAIProvider(api_key="sk-test"), Provider)

    def test_context_window_gpt4o(self):
        with patch("neoagent.providers.openai.AsyncOpenAI"):
            p = OpenAIProvider(api_key="sk-test", model="gpt-4o")
            assert p.get_context_window() == 128_000

    def test_context_window_unknown(self):
        with patch("neoagent.providers.openai.AsyncOpenAI"):
            p = OpenAIProvider(api_key="sk-test", model="future-model")
            assert p.get_context_window() == 128_000

class TestSerializeMessages:
    def test_system_prepended(self):
        msgs = [Message(role="user", content="hi")]
        result = _serialize_messages("You are helpful.", msgs)
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "hi"}

    def test_empty_system(self):
        msgs = [Message(role="user", content="hi")]
        result = _serialize_messages("", msgs)
        assert result[0] == {"role": "user", "content": "hi"}

    def test_tool_result_becomes_tool_role(self):
        msgs = [Message(role="user", content=[
            ToolResultBlock(tool_use_id="tc_1", content="file contents"),
        ])]
        result = _serialize_messages("", msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tc_1"
        assert result[0]["content"] == "file contents"

    def test_assistant_tool_calls(self):
        msgs = [Message(role="assistant", content=[
            TextBlock(text="Let me read."),
            ToolUseBlock(id="tc_1", name="read", input={"path": "/f"}),
        ])]
        result = _serialize_messages("", msgs)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me read."
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "read"

class TestConvertTools:
    def test_converts_schema(self):
        tools = [{"name": "read", "description": "Read file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]
        result = _convert_tools(tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "read"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_empty(self):
        assert _convert_tools([]) == []

class TestStopReasonMap:
    def test_stop(self):
        assert _STOP_REASON_MAP["stop"] == "end_turn"
    def test_tool_calls(self):
        assert _STOP_REASON_MAP["tool_calls"] == "tool_use"
    def test_length(self):
        assert _STOP_REASON_MAP["length"] == "max_tokens"

def _make_fake_openai_response(content="Done.", finish_reason="stop", tool_calls=None, prompt_tokens=10, completion_tokens=5):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp

class TestOpenAIProviderCreate:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock()
        return client

    @pytest.fixture
    def provider(self, mock_client):
        with patch("neoagent.providers.openai.AsyncOpenAI", return_value=mock_client):
            p = OpenAIProvider(api_key="sk-test")
        p._client = mock_client
        return p

    async def test_text_response(self, provider, mock_client):
        mock_client.chat.completions.create.return_value = _make_fake_openai_response()
        resp = await provider.create(system="sys", messages=[Message(role="user", content="hi")], tools=[])
        assert isinstance(resp, Response)
        assert resp.stop_reason == "end_turn"
        assert resp.text_content == "Done."

    async def test_tool_use_response(self, provider, mock_client):
        tc = MagicMock()
        tc.id = "call_abc"
        tc.type = "function"
        tc.function = MagicMock()
        tc.function.name = "read"
        tc.function.arguments = '{"path": "/tmp/f"}'
        mock_client.chat.completions.create.return_value = _make_fake_openai_response(
            content=None, finish_reason="tool_calls", tool_calls=[tc])
        resp = await provider.create(system="sys", messages=[Message(role="user", content="read")], tools=[{"name": "read", "description": "r", "input_schema": {}}])
        assert resp.stop_reason == "tool_use"
        assert len(resp.tool_use_blocks) == 1
        assert resp.tool_use_blocks[0].name == "read"
        assert resp.tool_use_blocks[0].input == {"path": "/tmp/f"}

    async def test_system_passed_as_message(self, provider, mock_client):
        mock_client.chat.completions.create.return_value = _make_fake_openai_response()
        await provider.create(system="Be helpful.", messages=[Message(role="user", content="hi")], tools=[])
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][0] == {"role": "system", "content": "Be helpful."}

    async def test_tools_converted(self, provider, mock_client):
        mock_client.chat.completions.create.return_value = _make_fake_openai_response()
        tools = [{"name": "bash", "description": "Run", "input_schema": {"type": "object"}}]
        await provider.create(system="sys", messages=[Message(role="user", content="go")], tools=tools)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"][0]["type"] == "function"
        assert call_kwargs["tools"][0]["function"]["name"] == "bash"
