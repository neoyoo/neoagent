from __future__ import annotations
from neoagent.config import NeoAgentConfig
from neoagent.core.loop import QueryLoop
from neoagent.core.prompt import PromptBuilder, PromptSection
from neoagent.core.types import ConversationResult, Message, TextBlock
from neoagent.providers.base import Provider
from neoagent.tools.base import BaseTool
from neoagent.tools.permission import PermissionChecker
from neoagent.tools.registry import ToolRegistry


def _create_provider(config: NeoAgentConfig) -> Provider:
    if config.provider == "openai":
        from neoagent.providers.openai import OpenAIProvider
        return OpenAIProvider(api_key=config.api_key, model=config.model, base_url=config.base_url)
    else:
        from neoagent.providers.anthropic import AnthropicProvider
        return AnthropicProvider(api_key=config.api_key, model=config.model, base_url=config.base_url)


class NeoAgent:
    def __init__(self, config: NeoAgentConfig) -> None:
        self._config = config
        self._provider = _create_provider(config)
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

    def enable_memory(
        self,
        memory_dir: "Path | None" = None,
        project_key: str | None = None,
    ) -> None:
        """Enable persistent memory for this agent."""
        import hashlib
        from pathlib import Path as _Path
        from neoagent.memory.store import MemoryStore
        from neoagent.memory.manager import MemoryManager

        if memory_dir is None:
            key = project_key or hashlib.sha1(str(_Path.cwd()).encode()).hexdigest()[:8]
            memory_dir = _Path.home() / ".neoagent" / "memory" / key

        store = MemoryStore(memory_dir)
        memory_manager = MemoryManager(store, self._provider)

        self._prompt_builder.add_section(PromptSection(
            name="memory",
            content=lambda: memory_manager.build_prompt_section(),
            priority=5,
            is_static=False,
        ))

        self._loop._memory_manager = memory_manager

    def enable_logging(
        self,
        log_dir: "Path | None" = None,
        console: bool = True,
    ) -> "Observer":
        """Enable framework-level logging. Returns the Observer for manual close()."""
        from neoagent.observe import Observer
        from pathlib import Path as _Path

        if log_dir is None:
            log_dir = _Path.cwd() / "logs"

        observer = Observer(log_dir=log_dir, console=console)
        self._loop._observer = observer
        return observer

    def disable_logging(self) -> None:
        """Disable logging and close any open log files."""
        if self._loop._observer:
            self._loop._observer.close()
            self._loop._observer = None
