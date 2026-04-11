from __future__ import annotations
from neoagent.config import NeoAgentConfig
from neoagent.core.loop import QueryLoop
from neoagent.core.prompt import PromptBuilder, PromptSection
from neoagent.core.types import ConversationResult, Message, TextBlock
from neoagent.events import EventBus
from neoagent.providers.base import Provider
from neoagent.session import JsonFileStorage, Session, SessionStorage
from neoagent.tools.base import BaseTool
from neoagent.tools.executor import ToolExecutor
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
    def __init__(self, config: NeoAgentConfig, storage: SessionStorage | None = None) -> None:
        self._config = config
        self._provider = _create_provider(config)
        self._permission = PermissionChecker(auto_approve=config.auto_approve_tools)
        self._registry = ToolRegistry()
        self._event_bus = EventBus()
        self._executor = ToolExecutor(
            registry=self._registry,
            permission_checker=self._permission,
            max_result_size=config.max_result_size,
            event_bus=self._event_bus,
        )
        self._prompt_builder = PromptBuilder()
        self._prompt_builder.add_section(PromptSection(
            name="identity",
            content="You are neoagent, a helpful AI assistant.",
            priority=0, is_static=True,
        ))
        self._loop = QueryLoop(
            provider=self._provider,
            tool_registry=self._registry,
            tool_executor=self._executor,
            prompt_builder=self._prompt_builder,
            max_turns=config.max_turns,
            context_budget=config.context_budget,
            event_bus=self._event_bus,
        )
        self._observer = None
        self._observer_subscriber = None
        # Storage: explicit > config.session_dir > None
        if storage is not None:
            self._storage: SessionStorage | None = storage
        elif config.session_dir is not None:
            self._storage = JsonFileStorage(config.session_dir)
        else:
            self._storage = None

    @property
    def event_bus(self) -> EventBus:
        """Expose EventBus for external subscribers."""
        return self._event_bus

    def register_tool(self, tool: BaseTool) -> None:
        self._registry.register(tool)

    def new_session(self, session_id: str | None = None) -> Session:
        """Create a new empty session."""
        return Session.create(session_id)

    def resume(self, session_id: str) -> Session:
        """Load an existing session from storage.

        Raises RuntimeError if no SessionStorage is configured.
        Raises KeyError if the session_id is not found.
        """
        if self._storage is None:
            raise RuntimeError("No SessionStorage configured. Pass storage= to NeoAgent.")
        return Session.resume(session_id, self._storage)

    async def chat(self, message: str, session: Session | None = None) -> str:
        """Send a message and return the assistant's reply as a string.

        If session=None (default), a temporary session is created and discarded
        after the call (backward-compatible behaviour, no persistence).

        If session is provided, the user message is appended to it, the loop runs,
        and (if storage is configured) the session is auto-saved afterwards.
        """
        _temp = session is None
        if _temp:
            session = Session.create()
        session.messages.append(Message(role="user", content=message))
        result = await self._loop.run(session=session)
        if not _temp and self._storage is not None:
            session.save(self._storage)
        if result.turns:
            last = result.turns[-1].response
            if isinstance(last.content, list):
                return "\n".join(b.text for b in last.content if isinstance(b, TextBlock))
            return last.content if isinstance(last.content, str) else ""
        return ""

    async def run(self, messages: list[Message], session: Session | None = None) -> ConversationResult:
        """Low-level interface: run the loop against a message list.

        If session is None, a transient session is created for this call.
        If session is provided and already has messages, raises ValueError to
        prevent silent state pollution — use session.messages directly, or pass
        a fresh session created with agent.new_session().
        """
        if session is None:
            session = Session.create()
            session.messages = list(messages)
        elif session.messages:
            raise ValueError(
                "Cannot pass both messages and a session with existing messages. "
                "Use session.messages directly, or pass a fresh session."
            )
        else:
            session.messages = list(messages)
        return await self._loop.run(session=session)

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
            key = project_key or hashlib.sha256(str(_Path.cwd()).encode()).hexdigest()[:8]
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
        from neoagent.observe_subscriber import ObserverSubscriber
        from pathlib import Path as _Path

        # Close existing observer if any to avoid file-handle leaks
        if self._observer:
            self._observer.close()

        if log_dir is None:
            log_dir = _Path.cwd() / "logs"

        observer = Observer(log_dir=log_dir, console=console)
        self._observer_subscriber = ObserverSubscriber(observer, self._event_bus)
        self._observer = observer
        return observer

    def disable_logging(self) -> None:
        """Disable logging and close any open log files."""
        if self._observer:
            self._observer.close()
            self._observer = None
            self._observer_subscriber = None
