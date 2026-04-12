from __future__ import annotations
from neoagent.config import NeoAgentConfig
from neoagent.core.loop import QueryLoop
from neoagent.core.prompt import PromptBuilder, PromptSection
from neoagent.core.types import ConversationResult, Message, TextBlock
from neoagent.events import EventBus
from neoagent.hooks import HookManager, HookType, HookHandler
from neoagent.providers.base import Provider
from neoagent.session import JsonFileStorage, Session, SessionStorage
from neoagent.tools.base import BaseTool
from neoagent.tools.executor import ToolExecutor
from neoagent.tools.permission import PermissionChecker
from neoagent.tools.registry import ToolRegistry
from neoagent.tools.deferred import DeferredToolRegistry
from neoagent.tools.builtin.tool_search import ToolSearchTool
from neoagent.mcp.transport import StdioTransport
from neoagent.mcp.client import MCPClient
from neoagent.mcp.tool import create_mcp_tools


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
        self._hook_manager = HookManager()
        self._deferred_registry = DeferredToolRegistry()
        self._mcp_clients: dict[str, object] = {}
        self._executor = ToolExecutor(
            registry=self._registry,
            permission_checker=self._permission,
            max_result_size=config.max_result_size,
            event_bus=self._event_bus,
            hook_manager=self._hook_manager,
        )
        self._prompt_builder = PromptBuilder()
        self._prompt_builder.add_section(PromptSection(
            name="identity",
            content=config.system_prompt or "You are neoagent, a helpful AI assistant.",
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
            hook_manager=self._hook_manager,
            deferred_registry=self._deferred_registry,
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

    async def run(self, messages: list[Message], session: Session | None = None, *, max_turns: int | None = None) -> ConversationResult:
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
        return await self._loop.run(session=session, max_turns=max_turns)

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

        # Detach old subscriber and close old observer to prevent handler leaks
        if self._observer_subscriber is not None:
            self._observer_subscriber.detach(self._event_bus)
            self._observer_subscriber = None
        if self._observer is not None:
            self._observer.close()
            self._observer = None

        if log_dir is None:
            log_dir = _Path.cwd() / "logs"

        observer = Observer(log_dir=log_dir, console=console)
        subscriber = ObserverSubscriber(observer)
        subscriber.attach(self._event_bus)
        self._observer_subscriber = subscriber
        self._observer = observer
        return observer

    def disable_logging(self) -> None:
        """Disable logging and close any open log files."""
        if self._observer_subscriber is not None:
            self._observer_subscriber.detach(self._event_bus)
            self._observer_subscriber = None
        if self._observer is not None:
            self._observer.close()
            self._observer = None

    # ── Hook API ──────────────────────────────────────────────────────────────

    def hook(self, hook_type: HookType, handler: HookHandler, priority: int = 0) -> None:
        """Register *handler* to be called for *hook_type* lifecycle events.

        Lower *priority* values run first (default 0). Same-priority handlers
        run in registration order.
        """
        self._hook_manager.register(hook_type, handler, priority)

    def on(self, hook_type: HookType, priority: int = 0):
        """Decorator form of hook(). Returns the original function unchanged.

        Usage::

            @agent.on("pre_tool_call")
            async def my_handler(event):
                return HookResult.allow()
        """
        def decorator(fn: HookHandler) -> HookHandler:
            self._hook_manager.register(hook_type, fn, priority)
            return fn
        return decorator

    def unhook(self, hook_type: HookType, handler: HookHandler) -> None:
        """Unregister *handler* from *hook_type*. No-op if not registered."""
        self._hook_manager.unregister(hook_type, handler)

    # ── MCP API ──────────────────────────────────────────────────────────────

    async def add_mcp_server(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        """Connect to an MCP server and register its tools.

        Creates a StdioTransport subprocess, performs the MCP initialize
        handshake, fetches the tool list, wraps each as MCPTool, registers
        them to ToolRegistry (for execution) and DeferredToolRegistry (for
        deferred loading).

        On the first call, also auto-registers the tool_search built-in tool
        so the LLM can discover and promote deferred tools.

        Args:
            name:    Logical server name used as tool name prefix (e.g. "github").
            command: Subprocess command (e.g. ["npx", "@anthropic/mcp-server-github"]).
            env:     Optional extra env vars forwarded to the subprocess (merged
                     into the whitelist-filtered env). Use for tokens/secrets.
        """
        transport = StdioTransport(command=command, env=env)
        client = MCPClient(name=name, transport=transport)
        await client.connect()

        tools = await create_mcp_tools(client=client, server_name=name)
        for tool in tools:
            self._registry.register(tool)
            self._deferred_registry.register(tool.name, tool.description)

        self._mcp_clients[name] = client

        # Auto-register tool_search on first MCP server
        if len(self._mcp_clients) == 1:
            tool_search = ToolSearchTool(
                deferred_registry=self._deferred_registry,
                tool_registry=self._registry,
            )
            self._registry.register(tool_search)

    async def remove_mcp_server(self, name: str) -> None:
        """Disconnect from an MCP server and unregister its tools.

        Args:
            name: Logical server name as passed to add_mcp_server().
        """
        client = self._mcp_clients.pop(name, None)
        if client is None:
            return

        # Remove tools with this server's prefix from ToolRegistry and
        # DeferredToolRegistry so that tool_search no longer returns ghost entries.
        prefix = f"{name}__"
        tools_to_remove = {
            tool_name
            for tool_name in self._registry.all_tools()
            if tool_name.startswith(prefix)
        }
        for tool_name in tools_to_remove:
            self._registry.unregister(tool_name)
        self._deferred_registry.remove(tools_to_remove)

        await client.close()

    def list_mcp_servers(self) -> list[str]:
        """Return the names of all connected MCP servers."""
        return list(self._mcp_clients.keys())

    async def close(self) -> None:
        """Close all MCP connections.

        Call this when the agent is being shut down to ensure MCP server
        subprocesses are terminated cleanly.
        """
        for client in list(self._mcp_clients.values()):
            await client.close()
        self._mcp_clients.clear()
