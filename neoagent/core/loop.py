from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from neoagent.core.compress import ContextCompressor
from neoagent.events import (
    EventBus,
    CompressCheckEvent, CompressDoneEvent, CompressFallbackEvent,
    MemoryExtractEvent,
    MessageCreatedEvent, ToolResultPersistedEvent,
    ProviderRequestEvent, ProviderResponseEvent,
    TurnCompleteEvent,
    WorkingMemoryUpdatedEvent,
    # NOTE: SkillChangeEvent is defined but not yet wired; PromptBuilder needs EventBus access (planned for future).
)

if TYPE_CHECKING:
    from neoagent.memory.manager import MemoryManager
    from neoagent.session import Session, SessionState
    from neoagent.tools.executor import ToolExecutor
    from neoagent.hooks import HookManager
    from neoagent.tools.deferred import DeferredToolRegistry
    from neoagent.v2.abc import WorkingMemoryStore

from neoagent.core.prompt import PromptBuilder, PromptSection
from neoagent.core.types import ConversationResult, Message, TextBlock, ToolCall, ToolResult, ToolResultBlock, Turn
from neoagent.providers.base import Provider, Response
from neoagent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
_DEFAULT_MAX_TOKENS = 8192

class QueryLoop:
    def __init__(self, provider: Provider, tool_registry: ToolRegistry, prompt_builder: PromptBuilder,
                 max_turns: int = 30, context_budget: int = 0,
                 memory_manager: "MemoryManager | None" = None,
                 event_bus: EventBus | None = None,
                 tool_executor: "ToolExecutor | None" = None,
                 hook_manager: "HookManager | None" = None,
                 deferred_registry: "DeferredToolRegistry | None" = None,
                 wm_store: "WorkingMemoryStore | None" = None):
        self._provider = provider
        self._registry = tool_registry
        self._executor = tool_executor
        self._prompt_builder = prompt_builder
        self.max_turns = max_turns
        self.context_budget = context_budget if context_budget > 0 else provider.get_context_window()
        self._compressor = ContextCompressor(provider=provider)
        self._memory_manager = memory_manager
        self._bus = event_bus if event_bus is not None else EventBus()
        self._hook_manager = hook_manager
        self._deferred_registry = deferred_registry
        self._wm_store = wm_store

    async def run(
        self,
        messages: "list[Message] | Session | None" = None,
        *,
        session: "Session | None" = None,
        max_turns: int | None = None,
    ) -> ConversationResult:
        """Run the query loop.

        Accepts either:
          - run(session=session)  — preferred Session-based API
          - run([Message(...)])   — legacy list-based API (for backward compat while agent.py is updated)
        """
        from neoagent.session import Session as _Session

        # Resolve session and msgs
        if session is not None:
            # Preferred: session passed as keyword arg
            _session: _Session = session
            msgs = _session.messages
            session_state: "SessionState | None" = _session.state
        elif isinstance(messages, _Session):
            # session passed positionally
            _session = messages
            msgs = _session.messages
            session_state = _session.state
        else:
            # Legacy list[Message] path — create a transient session for state
            from neoagent.session import Session as _S
            _session = _S.create()
            if messages:
                _session.messages.extend(messages)
            msgs = _session.messages
            session_state = _session.state

        # Propagate session state into ToolSearchTool so promote writes to session scope.
        # Uses a ContextVar to avoid race conditions in concurrent FastAPI requests.
        if self._deferred_registry and session_state is not None:
            from neoagent.tools.builtin.tool_search import set_session_state as _set_ss
            _set_ss(session_state)

        # Set full session so free_tool_result / recall_tool_result can access messages.
        from neoagent.tools.builtin.tool_search import set_current_session as _set_sess
        _set_sess(_session)

        turns: list[Turn] = []
        effective_max_turns = max_turns if max_turns is not None else self.max_turns
        for turn_idx in range(effective_max_turns):
            schemas = self._registry.get_schemas()
            should_compress = self._compressor.should_compress(msgs, schemas, self.context_budget)
            msg_tokens = self._compressor.estimate_tokens(msgs)
            tool_tokens = self._compressor.estimate_tools_tokens(schemas)
            self._bus.emit(CompressCheckEvent(
                msg_tokens=msg_tokens, tool_tokens=tool_tokens,
                budget=self.context_budget, should_compress=should_compress,
            ))
            if should_compress:
                old_summary = session_state.previous_summary if session_state else None
                failures_before = session_state.compression_failures if session_state else 0
                compressed = await self._compressor.compress(msgs, self.context_budget, session_state=session_state)
                # Sync back: replace session messages with compressed list
                _session.messages.clear()
                _session.messages.extend(compressed)
                msgs = _session.messages
                failures_after = session_state.compression_failures if session_state else 0
                if failures_after > failures_before:
                    # LLM compress failed — truncation fallback was used
                    self._bus.emit(CompressFallbackEvent(reason="LLM compression failed; fell back to truncation"))
                elif session_state and session_state.previous_summary:
                    self._bus.emit(CompressDoneEvent(
                        summary=session_state.previous_summary,
                        previous_summary=old_summary,
                    ))

            # Task 4.3: Inject per-turn dynamic sections via add_section (no raw string concat bypass).
            # Sections are added before build() and removed in try/finally to prevent accumulation.
            # priority=-100 ensures they sort after all regular sections (appear at the end).
            _deferred_injected = False
            _freed_injected = False

            # MCP deferred loading: filter schemas + inject deferred names into system prompt.
            # A tool managed by the deferred registry is hidden until promoted.
            # Promotion is tracked at two levels (OR logic for visibility):
            #   1. Session-scoped: session_state.promoted_tools (preferred, per-session isolation)
            #   2. Global: DeferredToolRegistry.is_deferred() == False (backward compat)
            # A tool is visible when EITHER condition indicates it has been promoted.
            if self._deferred_registry:
                session_promoted = session_state.promoted_tools if session_state else set()

                def _is_hidden(tool_name: str) -> bool:
                    if tool_name not in self._deferred_registry._all:
                        return False  # not a managed deferred tool
                    # Visible if promoted in this session OR promoted globally
                    return (
                        tool_name not in session_promoted
                        and self._deferred_registry.is_deferred(tool_name)
                    )

                schemas = [s for s in schemas if not _is_hidden(s["name"])]

                # Show names of all tools that are still deferred in this session
                deferred_names = sorted(
                    name for name in self._deferred_registry._all
                    if _is_hidden(name)
                )
                if deferred_names:
                    deferred_section_content = (
                        "<deferred-tools>\n"
                        + "\n".join(deferred_names)
                        + "\n</deferred-tools>"
                    )
                    self._prompt_builder.add_section(
                        PromptSection(name="_deferred", content=deferred_section_content, priority=-100, is_static=False)
                    )
                    _deferred_injected = True

            # Apply freed-tool rewriting before sending to provider (non-mutating).
            if session_state and session_state.freed_tool_results:
                msgs_for_llm = _apply_freed_to_messages(
                    msgs,
                    session_state.freed_tool_results,
                    session_state.recalled_this_turn,
                )
                freed_section_content = _render_freed_section(session_state.freed_tool_results)
                if freed_section_content:
                    self._prompt_builder.add_section(
                        PromptSection(name="_freed", content=freed_section_content, priority=-100, is_static=False)
                    )
                    _freed_injected = True
            else:
                msgs_for_llm = msgs

            try:
                system = self._prompt_builder.build()
            finally:
                # Always clean up per-turn sections regardless of success/failure
                if _deferred_injected:
                    self._prompt_builder.remove_section("_deferred")
                if _freed_injected:
                    self._prompt_builder.remove_section("_freed")

            self._bus.emit(ProviderRequestEvent(
                system=system, messages=tuple(msgs_for_llm),
                tools=tuple(schemas), turn=turn_idx,
            ))

            # pre_provider_call hook
            _system, _msgs, _schemas = system, msgs_for_llm, schemas
            if self._hook_manager:
                from neoagent.hooks import PreProviderCallEvent
                pre_event = PreProviderCallEvent(
                    system=system, messages=list(msgs), tools=list(schemas)
                )
                pre_result = await self._hook_manager.run_pre("pre_provider_call", pre_event)
                if pre_result.action == "deny":
                    reason = pre_result.reason or "denied by hook"
                    denial_msg = f"[Hook denied: {reason}]"
                    turn = Turn(
                        response=Message(role="assistant", content=[TextBlock(text=denial_msg)]),
                        tool_calls=[], tool_results=[], stop_reason="end_turn",
                    )
                    turns.append(turn)
                    return ConversationResult(turns=turns, reason="completed")
                if pre_result.action == "modify" and pre_result.modified_data:
                    _system = pre_result.modified_data.get("system", system)
                    _msgs = pre_result.modified_data.get("messages", msgs)
                    _schemas = pre_result.modified_data.get("tools", schemas)

            response = await self._provider.create(system=_system, messages=list(_msgs), tools=_schemas, max_tokens=_DEFAULT_MAX_TOKENS)
            if response.stop_reason == "max_tokens":
                response = await self._retry_with_higher_max(_system, list(_msgs), _schemas)

            # post_provider_call hook (after potential retry)
            if self._hook_manager and response.stop_reason != "max_tokens":
                from neoagent.hooks import PostProviderCallEvent
                post_event = PostProviderCallEvent(
                    response=response,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
                post_result = await self._hook_manager.run_post("post_provider_call", post_event)
                if post_result.action == "modify" and post_result.modified_data:
                    response = post_result.modified_data.get("response", response)
            if response.stop_reason == "max_tokens":
                # Still truncated after retry — treat as end_turn to avoid corrupt tool calls
                turn = Turn(
                    response=Message(role="assistant", content=response.content),
                    tool_calls=[], tool_results=[], stop_reason="max_tokens",
                )
                turns.append(turn)
                if session_state:
                    session_state.total_input_tokens += response.input_tokens
                    session_state.total_output_tokens += response.output_tokens
                return ConversationResult(turns=turns, reason="completed")
            self._bus.emit(ProviderResponseEvent(
                content=tuple(response.content), stop_reason=response.stop_reason,
                input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                turn=turn_idx,
            ))
            if session_state:
                session_state.total_input_tokens += response.input_tokens
                session_state.total_output_tokens += response.output_tokens
            tool_calls = [ToolCall(id=b.id, name=b.name, input=b.input) for b in response.tool_use_blocks]
            if response.stop_reason == "end_turn" or not tool_calls:
                turn = Turn(response=Message(role="assistant", content=response.content), tool_calls=[], tool_results=[], stop_reason="end_turn")
                turns.append(turn)
                # Task 4.5: Snapshot _current_wm + emit WorkingMemoryUpdatedEvent
                # BEFORE TurnCompleteEvent (WM snapshot is a turn-ending side-effect).
                if (
                    self._wm_store is not None
                    and session_state is not None
                    and session_state._current_wm is not None
                ):
                    _wm = session_state._current_wm
                    await self._wm_store.save(_session.id, _wm)
                    from neoagent.session import _wm_to_dict
                    self._bus.emit(WorkingMemoryUpdatedEvent(
                        session_id=_session.id,
                        version=_wm.version,
                        at_turn=_wm.at_turn,
                        wm_json=_wm_to_dict(_wm),
                        updated_by=_wm.updated_by,
                    ))
                self._bus.emit(TurnCompleteEvent(
                    turn_index=turn_idx,
                    stop_reason=turn.stop_reason,
                    tool_call_count=len(turn.tool_calls),
                ))
                if self._memory_manager:
                    current_tokens = self._compressor.estimate_tokens(msgs)
                    tool_calls_before = session_state.memory_tool_calls if session_state else 0
                    token_baseline_before = session_state.memory_token_baseline if session_state else 0
                    triggered, items_stored = await self._memory_manager.maybe_extract(
                        msgs, current_tokens, session_state=session_state
                    )
                    token_delta = max(0, current_tokens - token_baseline_before)
                    # NOTE: filenames not populated yet — extractor returns text summaries, not file references.
                    self._bus.emit(MemoryExtractEvent(
                        triggered=triggered,
                        tool_calls=tool_calls_before,
                        token_delta=token_delta,
                        items_stored=items_stored,
                    ))
                # Task 4.4: Emit MessageCreatedEvent for assistant_reply (end_turn path)
                if session_state:
                    self._bus.emit(MessageCreatedEvent(
                        session_id=_session.id,
                        msg_id=session_state.id_gen.next_msg_id(),
                        turn=turn_idx,
                        role="assistant",
                        source_type="assistant_reply",
                        content=list(response.content),
                    ))
                msgs.append(Message(role="assistant", content=response.content))
                if session_state:
                    session_state.recalled_this_turn.clear()
                _session.save_if_storage()  # per-turn auto-save (end_turn)
                return ConversationResult(turns=turns, reason="completed")
            if self._executor is None:
                raise RuntimeError(
                    "QueryLoop has no ToolExecutor but model requested tool calls. "
                    "Pass tool_executor= to QueryLoop."
                )
            results = await self._executor.execute(tool_calls)
            if self._memory_manager:
                self._memory_manager.record_tool_calls(len(tool_calls), session_state=session_state)
            assistant_msg = Message(role="assistant", content=response.content)
            # Task 4.4: Emit MessageCreatedEvent for assistant_reply (tool_use path)
            if session_state:
                self._bus.emit(MessageCreatedEvent(
                    session_id=_session.id,
                    msg_id=session_state.id_gen.next_msg_id(),
                    turn=turn_idx,
                    role="assistant",
                    source_type="assistant_reply",
                    content=list(response.content),
                ))
            msgs.append(assistant_msg)
            tool_result_blocks = [ToolResultBlock(tool_use_id=r.call_id, content=r.output, is_error=r.is_error) for r in results]
            result_msg = Message(role="user", content=tool_result_blocks)
            msgs.append(result_msg)
            # Record tool_use_id → tool_name mapping (used by free_tool_result tool and events).
            # Must be populated before emitting ToolResultPersistedEvent.
            if session_state:
                for call in tool_calls:
                    session_state.tool_use_to_tool_name[call.id] = call.name
            # Task 4.4: Emit MessageCreatedEvent (tool_result) + ToolResultPersistedEvent for each result
            if session_state:
                self._bus.emit(MessageCreatedEvent(
                    session_id=_session.id,
                    msg_id=session_state.id_gen.next_msg_id(),
                    turn=turn_idx,
                    role="user",
                    source_type="tool_result",
                    content=list(tool_result_blocks),
                ))
                for r in results:
                    tool_name_for_event = session_state.tool_use_to_tool_name.get(r.call_id, r.call_id)
                    self._bus.emit(ToolResultPersistedEvent(
                        session_id=_session.id,
                        tool_use_id=r.call_id,
                        turn=turn_idx,
                        tool_name=tool_name_for_event,
                        output=r.output,
                        size_bytes=len(r.output.encode("utf-8")),
                        is_error=r.is_error,
                    ))
            turn = Turn(response=assistant_msg, tool_calls=tool_calls, tool_results=results, stop_reason="tool_use")
            turns.append(turn)
            self._bus.emit(TurnCompleteEvent(
                turn_index=turn_idx,
                stop_reason=turn.stop_reason,
                tool_call_count=len(turn.tool_calls),
            ))

            # Clear recalled-this-turn so freed rewriting applies again next turn.
            if session_state:
                session_state.recalled_this_turn.clear()

            _session.save_if_storage()  # per-turn auto-save (tool_use)
        return ConversationResult(turns=turns, reason="max_turns")

    async def _retry_with_higher_max(self, system, messages, tools):
        higher = _DEFAULT_MAX_TOKENS * 2
        return await self._provider.create(system=system, messages=messages, tools=tools, max_tokens=higher)


def _apply_freed_to_messages(
    messages: list[Message],
    freed: "dict[str, object]",
    recalled: "set[str]",
) -> list[Message]:
    """Return copy of messages with freed tool_results replaced by placeholders (except recalled)."""
    from neoagent.session import FreedToolResult
    out: list[Message] = []
    for msg in messages:
        if not isinstance(msg.content, list):
            out.append(msg)
            continue
        new_blocks = []
        changed = False
        for block in msg.content:
            if (
                isinstance(block, ToolResultBlock)
                and block.tool_use_id in freed
                and block.tool_use_id not in recalled
            ):
                info: FreedToolResult = freed[block.tool_use_id]
                placeholder = (
                    f"[freed: tool_use_id={info.id}, tool={info.tool_name}, "
                    f"size={info.size}B, preview={info.preview!r}]"
                )
                new_blocks.append(ToolResultBlock(
                    tool_use_id=block.tool_use_id,
                    content=placeholder,
                    is_error=block.is_error,
                ))
                changed = True
            else:
                new_blocks.append(block)
        if changed:
            out.append(Message(role=msg.role, content=new_blocks))
        else:
            out.append(msg)
    return out


def _render_freed_section(freed: "dict[str, object]") -> str:
    if not freed:
        return ""
    from neoagent.session import FreedToolResult
    lines = ["## Freed Tool Results (recoverable)", ""]
    for info in freed.values():
        assert isinstance(info, FreedToolResult)
        lines.append(f"- `{info.id}` · {info.tool_name} · {info.size}B · {info.preview!r}")
    lines.append("")
    lines.append("Call `recall_tool_result(tool_use_id)` to view full content for the current turn.")
    return "\n".join(lines)
