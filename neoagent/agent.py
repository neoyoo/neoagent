from __future__ import annotations
import asyncio
import logging
import os
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from neoagent.config import NeoAgentConfig
from neoagent.core.loop import QueryLoop
from neoagent.core.prompt import LayeredPromptBuilder, PromptBuilder, PromptSection
from neoagent.core.types import ConversationResult, Message, TextBlock
from neoagent.events import EventBus, MessageCreatedEvent, SessionResumeWarningEvent, TurnCompleteEvent
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

if TYPE_CHECKING:
    from neoagent.v2.abc import WorkingMemoryStore
    from neoagent.v2.compressed_store import CompressedMessageStore

logger = logging.getLogger(__name__)


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
            enable_source_wrap=config.enable_source_wrap,
        )
        self._prompt_builder = PromptBuilder()
        self._prompt_builder.add_section(PromptSection(
            name="identity",
            content=config.system_prompt or "You are neoagent, a helpful AI assistant.",
            priority=0, is_static=True,
        ))

        # ── v2: LayeredPromptBuilder — owns ephemeral layers (WM, compressed_history,
        #   memory_context) rendered per-turn via QueryLoop.
        from neoagent.v2.schema import Layer
        self._layered_prompt_builder = LayeredPromptBuilder()
        self._layered_prompt_builder.register_layer_section(
            Layer.IDENTITY,
            PromptSection(
                name="identity",
                content=config.system_prompt or "You are neoagent, a helpful AI assistant.",
                priority=0,
            ),
        )

        # ── v2: WorkingMemoryStore ────────────────────────────────────────────
        if config.wm_store is None:
            from neoagent.v2.stores import InMemoryWorkingMemoryStore
            self._wm_store: "WorkingMemoryStore" = InMemoryWorkingMemoryStore()
        else:
            self._wm_store = config.wm_store

        # ── v2: ContextCompressor with strategy + event_bus ──────────────────
        from neoagent.core.compress import ContextCompressor
        from neoagent.v2.compressed_store import CompressedMessageStore, InMemoryCompressedMessageStore
        self._compressed_store: "CompressedMessageStore" = (
            config.compressed_message_store
            or InMemoryCompressedMessageStore()
        )
        self._compressor = ContextCompressor(
            provider=self._provider,
            strategy=config.compression_strategy,
            event_bus=self._event_bus,
            compressed_store=self._compressed_store,
        )

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
            wm_store=self._wm_store,
            layered_prompt_builder=self._layered_prompt_builder,
            compressed_store=self._compressed_store,
        )
        # Replace the compressor created internally by QueryLoop so that
        # strategy and event_bus are forwarded (QueryLoop creates its own compressor
        # without these fields; we override it here).
        self._loop._compressor = self._compressor

        # ── v2: Security prompt blocks ────────────────────────────────────────
        if config.enable_security_prompt_blocks:
            from neoagent.v2.prompts.security_blocks import build_security_sections
            for sec in build_security_sections(priority=0):
                self._prompt_builder.add_section(PromptSection(
                    name=f"v2_security_{sec.name}",
                    content=sec.content,
                    priority=sec.priority,
                    is_static=sec.is_static,
                ))

        # ── v2: Store memory_provider / memory_review_strategy for Phase 7.2/7.3
        self._memory_provider = config.memory_provider
        self._memory_review_strategy = config.memory_review_strategy

        # Tracks the active session during chat()/run() for NudgeCounter.
        self._current_session: Session | None = None

        # ── Task 7.2: Subscribe to TurnCompleteEvent → NudgeCounter tick / review
        self._event_bus.subscribe(TurnCompleteEvent, self._on_turn_complete_sync)

        self._observer = None
        self._observer_subscriber = None
        # Storage: explicit > config.session_dir > None
        if storage is not None:
            self._storage: SessionStorage | None = storage
        elif config.session_dir is not None:
            self._storage = JsonFileStorage(config.session_dir)
        else:
            self._storage = None

    # ── Task 7.2: NudgeCounter + MemoryReview on TurnCompleteEvent ───────────

    def _on_turn_complete_sync(self, event: TurnCompleteEvent) -> None:
        """Sync EventBus handler: tick nudge_counter; fire async review as a task."""
        session = self._current_session
        if session is None:
            return
        nudge = session.state.nudge_counter
        nudge.tick()
        if not nudge.should_trigger_review():
            return
        if self._memory_review_strategy is None or self._memory_provider is None:
            return
        # Schedule async work without blocking the sync EventBus.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._on_turn_complete_async(session))
        except RuntimeError:
            # No running event loop — skip (test or non-async context)
            pass

    async def _on_turn_complete_async(self, session: Session) -> None:
        """Async part: call MemoryReviewStrategy.review() and MemoryProvider.upsert()."""
        try:
            wm = session.state._current_wm
            entries = await self._memory_review_strategy.review(  # type: ignore[union-attr]
                session_id=session.id,
                user_id="default",
                messages=list(session.messages),
                wm=wm,
            )
            if entries:
                await self._memory_provider.upsert(entries)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("MemoryReview error (soft failure): %s", exc)

    @property
    def event_bus(self) -> EventBus:
        """Expose EventBus for external subscribers."""
        return self._event_bus

    @property
    def wm_store(self) -> "WorkingMemoryStore":
        """Expose the WorkingMemoryStore for inspection and external use."""
        return self._wm_store

    @property
    def compressed_store(self) -> "CompressedMessageStore":
        """Expose the CompressedMessageStore for tool wiring and external use."""
        return self._compressed_store

    def register_tool(self, tool: BaseTool) -> None:
        self._registry.register(tool)

    def new_session(self, session_id: str | None = None) -> Session:
        """Create a new empty session."""
        from neoagent.v2.compressed_store import InMemoryCompressedMessageStore
        session = Session.create(session_id)
        if isinstance(self._compressed_store, InMemoryCompressedMessageStore):
            self._compressed_store.bind_session(session.id, session.state.compressed_messages)
        return session

    def resume(self, session_id: str, validate: bool = True) -> Session:
        """Load an existing session from storage.

        Raises RuntimeError if no SessionStorage is configured.
        Raises KeyError if the session_id is not found.

        If validate=True (default), runs sanity checks and emits
        SessionResumeWarningEvent for any issues found (workspace missing,
        stale session, etc.).
        """
        if self._storage is None:
            raise RuntimeError("No SessionStorage configured. Pass storage= to NeoAgent.")
        session = Session.resume(session_id, self._storage)
        from neoagent.v2.compressed_store import InMemoryCompressedMessageStore
        if isinstance(self._compressed_store, InMemoryCompressedMessageStore):
            self._compressed_store.bind_session(session.id, session.state.compressed_messages)
        if validate:
            self._validate_resume(session)
        return session

    def _validate_resume(self, session: Session) -> None:
        """Run sanity checks on a resumed session; emit warnings via event bus."""
        # Check workspace_path exists if recorded in metadata
        workspace_path = session.metadata.get("workspace_path")
        if workspace_path is not None and not os.path.exists(workspace_path):
            self._event_bus.emit(SessionResumeWarningEvent(
                session_id=session.id,
                reason="workspace_missing",
                details=f"workspace_path does not exist: {workspace_path}",
            ))

        # Check if session is stale (updated_at older than 24h)
        now = datetime.now()
        age = now - session.updated_at
        if age > timedelta(hours=24):
            hours_ago = age.total_seconds() / 3600
            self._event_bus.emit(SessionResumeWarningEvent(
                session_id=session.id,
                reason="stale_session",
                details=f"Session was last updated {hours_ago:.1f}h ago",
            ))

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
        # Increment session-global user-turn counter BEFORE stamping the user message.
        # All Messages created during this chat() call will share this turn value.
        session.state.user_turn_counter += 1
        _current_user_turn = session.state.user_turn_counter
        # Assign msg_id + user_turn so recall_turn / compressor can find this user
        # input later. turn=_current_user_turn is the session-global counter value.
        _user_msg_id = session.state.id_gen.next_msg_id()
        session.messages.append(Message(id=_user_msg_id, turn=_current_user_turn, role="user", content=message))
        self._event_bus.emit(MessageCreatedEvent(
            session_id=session.id,
            msg_id=_user_msg_id,
            turn=_current_user_turn,
            role="user",
            source_type="user_input",
            content=message,
        ))
        if not _temp and self._storage is not None:
            session.bind_storage(self._storage)
        self._current_session = session
        try:
            result = await self._loop.run(session=session, user_turn=_current_user_turn)
        finally:
            self._current_session = None
        # Increment turns_since_last_compression after each successful chat().
        # (Compression resets it to 0 inside compress.py's strategy path;
        #  we only increment here when the call completed without exception.)
        session.state.turns_since_last_compression += 1
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
        _temp = session is None
        if _temp:
            session = Session.create()
            session.messages = list(messages)
        elif session.messages:
            raise ValueError(
                "Cannot pass both messages and a session with existing messages. "
                "Use session.messages directly, or pass a fresh session."
            )
        else:
            session.messages = list(messages)
        if not _temp and self._storage is not None:
            session.bind_storage(self._storage)
        self._current_session = session
        try:
            result = await self._loop.run(session=session, max_turns=max_turns)
        finally:
            self._current_session = None
        if not _temp and self._storage is not None:
            session.save(self._storage)
        return result

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
        # Task 7.3: forward memory_provider so MemoryManager can mirror to provider.
        memory_manager = MemoryManager(
            store, self._provider,
            memory_provider=self._memory_provider,
        )

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

    def enable_metrics(self) -> "MetricsCollector":
        """Enable per-turn and session metrics collection. Returns a MetricsCollector.

        The collector subscribes to ProviderRequestEvent, ProviderResponseEvent,
        ToolCallEvent and TurnCompleteEvent on the EventBus and accumulates
        TurnMetrics for each completed turn.

        Usage::

            collector = agent.enable_metrics()
            await agent.chat("hello")
            sm = collector.get_session_metrics()
            print(sm.total_input_tokens, sm.duration_ms)
        """
        from neoagent.eval.metrics import MetricsCollector
        from neoagent.events import (
            ProviderRequestEvent,
            ProviderResponseEvent,
            ToolCallEvent,
            TurnCompleteEvent,
        )
        collector = MetricsCollector()
        self._event_bus.subscribe(ProviderRequestEvent, collector._on_provider_request)
        self._event_bus.subscribe(ProviderResponseEvent, collector._on_provider_response)
        self._event_bus.subscribe(ToolCallEvent, collector._on_tool_call)
        self._event_bus.subscribe(TurnCompleteEvent, collector._on_turn_complete)
        return collector

    def enable_usage_tracking(self) -> "UsageTracker":
        """Enable token usage tracking. Returns a UsageTracker for inspection.

        The tracker subscribes to ProviderResponseEvent on the EventBus and
        accumulates input/output token counts per model name.

        Usage::

            tracker = agent.enable_usage_tracking()
            await agent.chat("hello")
            print(tracker.total_input_tokens, tracker.total_output_tokens)
        """
        from neoagent.eval.usage import UsageTracker
        from neoagent.events import ProviderResponseEvent
        tracker = UsageTracker()
        tracker._current_model = self._provider.model
        self._event_bus.subscribe(ProviderResponseEvent, tracker._handle_response)
        return tracker

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

        Security: Only pass trusted commands. The command is executed as a
        subprocess — do not pass user-controlled input directly.
        """
        if not command:
            raise ValueError("MCP server command must not be empty")
        # Block obvious shell injection in binary name
        if re.search(r'[;&|`$(){}]', command[0]):
            raise ValueError(
                f"MCP server command contains shell metacharacters: {command[0]!r}"
            )
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
