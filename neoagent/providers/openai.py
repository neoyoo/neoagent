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
    def __init__(self, api_key: str, model: str = "gpt-4o", max_tokens: int = 4096, base_url: str | None = None):
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)
        self.model = model
        self.max_tokens = max_tokens

    def get_context_window(self) -> int:
        return _get_context_window(self.model)

    async def create(self, system: str, messages: list[Message], tools: list[dict], **kwargs) -> Response:
        serialized = _serialize_messages(system, messages)
        openai_tools = _convert_tools(tools) if tools else None
        text_delta_callback = kwargs.get("text_delta_callback")

        create_kwargs: dict = {
            "model": self.model,
            "messages": serialized,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if openai_tools:
            create_kwargs["tools"] = openai_tools

        if text_delta_callback is None:
            raw = await self._client.chat.completions.create(**create_kwargs)
            return _parse_response(raw)

        # ── Streaming path ──
        create_kwargs["stream"] = True
        create_kwargs["stream_options"] = {"include_usage": True}

        content_accum = ""
        tool_calls_accum: dict[int, dict] = {}
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0

        stream = await self._client.chat.completions.create(**create_kwargs)
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(delta, "content", None):
                content_accum += delta.content
                text_delta_callback(delta.content)
            if getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    entry = tool_calls_accum.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["arguments"] += tc_delta.function.arguments
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        content: list[ContentBlock] = []
        if content_accum:
            content.append(TextBlock(text=content_accum))
        for entry in tool_calls_accum.values():
            try:
                input_data = json.loads(entry["arguments"]) if entry["arguments"] else {}
            except (json.JSONDecodeError, TypeError):
                input_data = {}
            content.append(ToolUseBlock(id=entry["id"], name=entry["name"], input=input_data))

        return Response(
            content=content,
            stop_reason=_STOP_REASON_MAP.get(finish_reason, "end_turn"),
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
