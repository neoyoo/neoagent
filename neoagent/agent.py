from __future__ import annotations
from neoagent.config import NeoAgentConfig
from neoagent.core.loop import QueryLoop
from neoagent.core.prompt import PromptBuilder, PromptSection
from neoagent.core.types import ConversationResult, Message, TextBlock
from neoagent.providers.anthropic import AnthropicProvider
from neoagent.tools.base import BaseTool
from neoagent.tools.permission import PermissionChecker
from neoagent.tools.registry import ToolRegistry


class NeoAgent:
    def __init__(self, config: NeoAgentConfig) -> None:
        self._config = config
        self._provider = AnthropicProvider(api_key=config.api_key, model=config.model)
        self._permission = PermissionChecker()
        self._registry = ToolRegistry(max_result_size=config.max_result_size)
        self._prompt_builder = PromptBuilder()
        self._prompt_builder.add_section(PromptSection(
            name="identity",
            content="You are neoagent, a helpful AI assistant.",
            priority=0, is_static=True,
        ))
        self._loop = QueryLoop(
            provider=self._provider,
            tool_registry=self._registry,
            prompt_builder=self._prompt_builder,
            max_turns=config.max_turns,
            context_budget=config.context_budget,
        )

    def register_tool(self, tool: BaseTool) -> None:
        self._registry.register(tool)

    async def chat(self, message: str) -> str:
        messages = [Message(role="user", content=message)]
        result = await self._loop.run(messages)
        if result.turns:
            last = result.turns[-1].response
            if isinstance(last.content, list):
                return "\n".join(b.text for b in last.content if isinstance(b, TextBlock))
            return last.content if isinstance(last.content, str) else ""
        return ""

    async def run(self, messages: list[Message]) -> ConversationResult:
        return await self._loop.run(messages)
