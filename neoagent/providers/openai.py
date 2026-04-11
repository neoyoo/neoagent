from __future__ import annotations
import json
from openai import AsyncOpenAI
from neoagent.core.types import ContentBlock, Message, TextBlock, ToolUseBlock, ToolResultBlock
from neoagent.providers.base import Provider, Response

_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o3": 200_000,
    "o4-mini": 200_000,
}
_DEFAULT_CONTEXT_WINDOW = 128_000

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}

def _get_context_window(model: str) -> int:
    for prefix, size in _CONTEXT_WINDOWS.items():
        if model.startswith(prefix):
            return size
    return _DEFAULT_CONTEXT_WINDOW

def _convert_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-style tool schemas to OpenAI function calling format."""
    converted = []
    for tool in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return converted

def _serialize_messages(system: str, messages: list[Message]) -> list[dict]:
    """Convert internal Messages to OpenAI chat format, prepending system message."""
    serialized: list[dict] = []
    if system:
        serialized.append({"role": "system", "content": system})

    for msg in messages:
        if isinstance(msg.content, str):
            serialized.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg.content, list):
            # Check if this is a tool result message (user message with ToolResultBlocks)
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            if tool_results and msg.role == "user":
                # Convert each ToolResultBlock to an OpenAI tool message
                for tr in tool_results:
                    serialized.append({
                        "role": "tool",
                        "tool_call_id": tr.tool_use_id,
                        "content": tr.content,
                    })
            else:
                # Assistant message with possible tool_use blocks
                text_parts = []
                tool_calls = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        })

                entry: dict = {"role": msg.role, "content": "\n".join(text_parts) if text_parts else None}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                serialized.append(entry)

    return serialized

def _parse_response(raw) -> Response:
    """Convert OpenAI ChatCompletion to internal Response."""
    choice = raw.choices[0]
    message = choice.message

    content: list[ContentBlock] = []
    if message.content:
        content.append(TextBlock(text=message.content))

    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                input_data = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                input_data = {}
            content.append(ToolUseBlock(
                id=tc.id,
                name=tc.function.name,
                input=input_data,
            ))

    stop_reason = _STOP_REASON_MAP.get(choice.finish_reason, "end_turn")

    return Response(
        content=content,
        stop_reason=stop_reason,
        input_tokens=raw.usage.prompt_tokens if raw.usage else 0,
        output_tokens=raw.usage.completion_tokens if raw.usage else 0,
    )

class OpenAIProvider(Provider):
    def __init__(self, api_key: str, model: str = "gpt-4o", max_tokens: int = 4096):
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def get_context_window(self) -> int:
        return _get_context_window(self.model)

    async def create(self, system: str, messages: list[Message], tools: list[dict], **kwargs) -> Response:
        serialized = _serialize_messages(system, messages)
        openai_tools = _convert_tools(tools) if tools else None

        create_kwargs: dict = {
            "model": self.model,
            "messages": serialized,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if openai_tools:
            create_kwargs["tools"] = openai_tools

        raw = await self._client.chat.completions.create(**create_kwargs)
        return _parse_response(raw)
