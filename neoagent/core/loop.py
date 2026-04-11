from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Callable
from neoagent.core.compress import ContextCompressor
from neoagent.events import (
    EventBus,
    CompressCheckEvent, CompressDoneEvent,
    ProviderRequestEvent, ProviderResponseEvent,
    ToolCallEvent, ToolResultEvent,
    TurnCompleteEvent,
)

if TYPE_CHECKING:
    from neoagent.memory.manager import MemoryManager
    from neoagent.session import Session, SessionState

from neoagent.core.prompt import PromptBuilder
from neoagent.core.types import ConversationResult, Message, ToolCall, ToolResult, ToolResultBlock, ToolUseBlock, Turn
from neoagent.providers.base import Provider, Response
from neoagent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
_DEFAULT_MAX_TOKENS = 8192
_MAX_RETRY_TOKENS = 16384

class QueryLoop:
    def __init__(self, provider: Provider, tool_registry: ToolRegistry, prompt_builder: PromptBuilder,
                 max_turns: int = 30, context_budget: int = 0, on_turn: Callable[[Turn], None] | None = None,
                 memory_manager: "MemoryManager | None" = None,
                 event_bus: EventBus | None = None):
        self._provider = provider
        self._registry = tool_registry
        self._prompt_builder = prompt_builder
        self.max_turns = max_turns
        self.context_budget = context_budget if context_budget > 0 else provider.get_context_window()
        self._on_turn = on_turn
        self._compressor = ContextCompressor(provider=provider)
        self._memory_manager = memory_manager
        self._bus = event_bus if event_bus is not None else EventBus()

    async def run(
        self,
        messages: "list[Message] | Session | None" = None,
        *,
        session: "Session | None" = None,
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

        turns: list[Turn] = []
        for turn_idx in range(self.max_turns):
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
                compressed = await self._compressor.compress(msgs, self.context_budget, session_state=session_state)
                # Sync back: replace session messages with compressed list
                _session.messages.clear()
                _session.messages.extend(compressed)
                msgs = _session.messages
                if session_state and session_state.previous_summary:
                    self._bus.emit(CompressDoneEvent(
                        summary=session_state.previous_summary,
                        previous_summary=old_summary,
                    ))
            system = self._prompt_builder.build()
            self._bus.emit(ProviderRequestEvent(
                system=system, messages=tuple(msgs),
                tools=tuple(schemas), turn=turn_idx,
            ))
            response = await self._provider.create(system=system, messages=msgs, tools=schemas, max_tokens=_DEFAULT_MAX_TOKENS)
            if response.stop_reason == "max_tokens":
                response = await self._retry_with_higher_max(system, msgs, schemas)
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
                if self._on_turn:
                    self._on_turn(turn)
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
                self._bus.emit(TurnCompleteEvent(
                    turn_index=turn_idx,
                    stop_reason=turn.stop_reason,
                    tool_call_count=len(turn.tool_calls),
                ))
                if self._on_turn:
                    self._on_turn(turn)
                if self._memory_manager:
                    current_tokens = self._compressor.estimate_tokens(msgs)
                    await self._memory_manager.maybe_extract(msgs, current_tokens, session_state=session_state)
                return ConversationResult(turns=turns, reason="completed")
            for tc in tool_calls:
                self._bus.emit(ToolCallEvent(name=tc.name, input_data=tc.input, call_id=tc.id))
            results = await self._registry.execute(tool_calls)
            for tc, r in zip(tool_calls, results):
                self._bus.emit(ToolResultEvent(name=tc.name, call_id=r.call_id,
                                               output=r.output, is_error=r.is_error))
            if self._memory_manager:
                self._memory_manager.record_tool_calls(len(tool_calls), session_state=session_state)
            assistant_msg = Message(role="assistant", content=response.content)
            msgs.append(assistant_msg)
            result_msg = Message(role="user", content=[ToolResultBlock(tool_use_id=r.call_id, content=r.output, is_error=r.is_error) for r in results])
            msgs.append(result_msg)
            turn = Turn(response=assistant_msg, tool_calls=tool_calls, tool_results=results, stop_reason="tool_use")
            turns.append(turn)
            self._bus.emit(TurnCompleteEvent(
                turn_index=turn_idx,
                stop_reason=turn.stop_reason,
                tool_call_count=len(turn.tool_calls),
            ))
            if self._on_turn:
                self._on_turn(turn)
        return ConversationResult(turns=turns, reason="max_turns")

    async def _retry_with_higher_max(self, system, messages, tools):
        higher = min(_MAX_RETRY_TOKENS, _DEFAULT_MAX_TOKENS * 2)
        return await self._provider.create(system=system, messages=messages, tools=tools, max_tokens=higher)
