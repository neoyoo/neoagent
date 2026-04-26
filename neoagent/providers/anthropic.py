from __future__ import annotations
from anthropic import AsyncAnthropic
from neoagent.core.types import ContentBlock, Message, TextBlock, ToolUseBlock, ToolResultBlock
from neoagent.providers.base import Provider, Response

_DEFAULT_CONTEXT_WINDOW = 200_000

def _serialize_messages(messages: list[Message]) -> list[dict]:
    serialized = []
    for msg in messages:
        if isinstance(msg.content, str):
            serialized.append({"role": msg.role, "content": msg.content})
        else:
            blocks = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
                elif isinstance(block, ToolResultBlock):
                    blocks.append({"type": "tool_result", "tool_use_id": block.tool_use_id, "content": block.content, "is_error": block.is_error})
            serialized.append({"role": msg.role, "content": blocks})
    return serialized

def _parse_content_blocks(raw_blocks: list) -> list[ContentBlock]:
    parsed: list[ContentBlock] = []
    for block in raw_blocks:
        if getattr(block, "type", None) == "text":
            parsed.append(TextBlock(text=block.text))
        elif getattr(block, "type", None) == "tool_use":
            parsed.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))
    return parsed

class AnthropicProvider(Provider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", max_tokens: int = 8192, base_url: str | None = None):
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**client_kwargs)
        self.model = model
        self.max_tokens = max_tokens

    def get_context_window(self) -> int:
        return _DEFAULT_CONTEXT_WINDOW

    async def create(self, system: str, messages: list[Message], tools: list[dict], **kwargs) -> Response:
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        text_delta_callback = kwargs.get("text_delta_callback")

        if text_delta_callback is not None:
            async with self._client.messages.stream(
                model=self.model, max_tokens=max_tokens,
                system=system, messages=_serialize_messages(messages), tools=tools,
            ) as stream:
                async for text in stream.text_stream:
                    text_delta_callback(text)
                raw = await stream.get_final_message()
        else:
            raw = await self._client.messages.create(
                model=self.model, max_tokens=max_tokens,
                system=system, messages=_serialize_messages(messages), tools=tools,
            )

        return Response(
            content=_parse_content_blocks(raw.content),
            stop_reason=raw.stop_reason,
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
        )
