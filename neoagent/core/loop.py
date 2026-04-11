from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Callable
from neoagent.core.compress import ContextCompressor

if TYPE_CHECKING:
    from neoagent.memory.manager import MemoryManager
    from neoagent.observe import Observer

from neoagent.core.prompt import PromptBuilder
from neoagent.core.types import ConversationResult, Message, ToolCall, ToolResult, ToolResultBlock, ToolUseBlock, Turn
from neoagent.providers.base import Provider, Response
from neoagent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
_MAX_TOKENS_RETRY_FRACTION = 0.5
_DEFAULT_MAX_TOKENS = 8096

class QueryLoop:
    def __init__(self, provider: Provider, tool_registry: ToolRegistry, prompt_builder: PromptBuilder,
                 max_turns: int = 30, context_budget: int = 0, on_turn: Callable[[Turn], None] | None = None,
                 memory_manager: "MemoryManager | None" = None,
                 observer: "Observer | None" = None):
        self._provider = provider
        self._registry = tool_registry
        self._prompt_builder = prompt_builder
        self.max_turns = max_turns
        self.context_budget = context_budget if context_budget > 0 else provider.get_context_window()
        self._on_turn = on_turn
        self._compressor = ContextCompressor(provider=provider)
        self._memory_manager = memory_manager
        self._observer = observer

    async def run(self, messages: list[Message]) -> ConversationResult:
        msgs = list(messages)
        turns: list[Turn] = []
        for turn_idx in range(self.max_turns):
            schemas = self._registry.get_schemas()
            should_compress = self._compressor.should_compress(msgs, schemas, self.context_budget)
            if self._observer:
                msg_tokens = self._compressor.estimate_tokens(msgs)
                tool_tokens = self._compressor.estimate_tools_tokens(schemas)
                self._observer.on_compress_check(msg_tokens, tool_tokens, self.context_budget, should_compress)
            if should_compress:
                old_summary = self._compressor._previous_summary
                msgs = await self._compressor.compress(msgs, self.context_budget)
                if self._observer and self._compressor._previous_summary:
                    self._observer.on_compress_done(self._compressor._previous_summary, old_summary)
            system = self._prompt_builder.build()
            if self._observer:
                self._observer.on_provider_request(system, msgs, schemas, turn=turn_idx)
            response = await self._provider.create(system=system, messages=msgs, tools=schemas, max_tokens=_DEFAULT_MAX_TOKENS)
            if response.stop_reason == "max_tokens":
                response = await self._retry_with_lower_max(system, msgs, schemas)
            if self._observer:
                self._observer.on_provider_response(
                    response.content, response.stop_reason,
                    response.input_tokens, response.output_tokens, turn=turn_idx,
                )
            tool_calls = [ToolCall(id=b.id, name=b.name, input=b.input) for b in response.tool_use_blocks]
            if response.stop_reason == "end_turn" or not tool_calls:
                turn = Turn(response=Message(role="assistant", content=response.content), tool_calls=[], tool_results=[], stop_reason="end_turn")
                turns.append(turn)
                if self._on_turn:
                    self._on_turn(turn)
                if self._memory_manager:
                    current_tokens = self._compressor.estimate_tokens(msgs)
                    await self._memory_manager.maybe_extract(msgs, current_tokens)
                return ConversationResult(turns=turns, reason="completed")
            if self._observer:
                for tc in tool_calls:
                    self._observer.on_tool_call(tc.name, tc.input)
            results = await self._registry.execute(tool_calls)
            if self._observer:
                for tc, r in zip(tool_calls, results):
                    self._observer.on_tool_result(tc.name, r.output[:300], r.is_error)
            if self._memory_manager:
                self._memory_manager.record_tool_calls(len(tool_calls))
            assistant_msg = Message(role="assistant", content=response.content)
            msgs.append(assistant_msg)
            result_msg = Message(role="user", content=[ToolResultBlock(tool_use_id=r.call_id, content=r.output, is_error=r.is_error) for r in results])
            msgs.append(result_msg)
            turn = Turn(response=assistant_msg, tool_calls=tool_calls, tool_results=results, stop_reason="tool_use")
            turns.append(turn)
            if self._on_turn:
                self._on_turn(turn)
        return ConversationResult(turns=turns, reason="max_turns")

    async def _retry_with_lower_max(self, system, messages, tools):
        lower = max(1, int(_DEFAULT_MAX_TOKENS * _MAX_TOKENS_RETRY_FRACTION))
        return await self._provider.create(system=system, messages=messages, tools=tools, max_tokens=lower)
