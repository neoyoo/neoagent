from __future__ import annotations
import asyncio
import copy
import logging
from types import MappingProxyType
from typing import TYPE_CHECKING

from neoagent.core.types import ToolCall, ToolResult
from neoagent.tools.permission import PermissionChecker
from neoagent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from neoagent.events import EventBus
    from neoagent.hooks import HookManager

logger = logging.getLogger(__name__)


class ToolExecutor:
    """工具执行层：权限检查、并发路由、结果截断、事件发射。

    与 ToolRegistry 解耦：ToolRegistry 管 schema，ToolExecutor 管运行时。
    同一个 ToolExecutor 可配合不同 registry（本地工具/MCP 工具）使用。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_checker: PermissionChecker | None = None,
        max_result_size: int = 50000,
        event_bus: "EventBus | None" = None,
        hook_manager: "HookManager | None" = None,
    ) -> None:
        self._registry = registry
        self._permission = permission_checker or PermissionChecker()
        self.max_result_size = max_result_size
        self._bus = event_bus
        self._hook_manager = hook_manager

    async def execute(self, calls: list[ToolCall]) -> list[ToolResult]:
        safe, unsafe = self._partition_by_concurrency(calls)
        results: dict[int, ToolResult] = {}

        if safe:
            safe_results = await asyncio.gather(
                *[self._run_one(idx, call) for idx, call in safe]
            )
            for idx, result in safe_results:
                results[idx] = result

        for idx, call in unsafe:
            _, result = await self._run_one(idx, call)
            results[idx] = result

        return [results[i] for i in range(len(calls))]

    def _partition_by_concurrency(self, calls: list[ToolCall]) -> tuple[list, list]:
        safe, unsafe = [], []
        for idx, call in enumerate(calls):
            tool = self._registry.get_tool(call.name)
            if tool and tool.is_concurrent_safe:
                safe.append((idx, call))
            else:
                unsafe.append((idx, call))
        return safe, unsafe

    async def _run_one(self, idx: int, call: ToolCall) -> tuple[int, ToolResult]:
        tool = self._registry.get_tool(call.name)
        if tool is None:
            result = ToolResult(call_id=call.id, output=f"Tool not found: {call.name}", is_error=True)
            self._emit_result(call, result)
            return idx, result

        # Emit ToolCallEvent before execution; wrap in MappingProxyType to prevent handler mutation
        if self._bus:
            from neoagent.events import ToolCallEvent
            payload = MappingProxyType(copy.deepcopy(call.input))
            self._bus.emit(ToolCallEvent(name=call.name, input_data=payload, call_id=call.id))

        try:
            validated_input = tool.input_model.model_validate(call.input)
            allowed = await self._permission.check(tool, validated_input)
            if not allowed:
                msg = (
                    f"Permission denied: tool '{tool.name}' is disabled"
                    if tool.permission == "deny"
                    else f"Permission denied: tool '{tool.name}' requires user approval"
                )
                result = ToolResult(call_id=call.id, output=msg, is_error=True)
                self._emit_result(call, result)
                return idx, result

            # pre_tool_call hook (runs AFTER permission check)
            if self._hook_manager:
                from neoagent.hooks import PreToolCallEvent
                pre_event = PreToolCallEvent(
                    tool_name=call.name,
                    tool_input=dict(call.input),
                    call_id=call.id,
                )
                pre_result = await self._hook_manager.run_pre("pre_tool_call", pre_event)
                if pre_result.action == "deny":
                    reason = pre_result.reason or "denied by hook"
                    result = ToolResult(call_id=call.id, output=f"Hook denied: {reason}", is_error=True)
                    self._emit_result(call, result)
                    return idx, result
                if pre_result.action == "modify" and pre_result.modified_data:
                    # Apply modified tool_input to call for execution
                    new_input = pre_result.modified_data.get("tool_input", call.input)
                    call = ToolCall(id=call.id, name=call.name, input=new_input)
                    validated_input = tool.input_model.model_validate(call.input)

            result = await tool.execute(validated_input)
            result.call_id = call.id
            if len(result.output) > self.max_result_size:
                result.output = result.output[:self.max_result_size] + "\n[truncated]"

            # post_tool_call hook
            if self._hook_manager:
                from neoagent.hooks import PostToolCallEvent
                post_event = PostToolCallEvent(
                    tool_name=call.name,
                    tool_input=dict(call.input),
                    call_id=call.id,
                    result=result.output,
                    is_error=result.is_error,
                )
                post_result = await self._hook_manager.run_post("post_tool_call", post_event)
                if post_result.action == "modify" and post_result.modified_data:
                    new_output = post_result.modified_data.get("result", result.output)
                    result = ToolResult(call_id=call.id, output=new_output, is_error=result.is_error)

        except Exception as e:
            result = ToolResult(call_id=call.id, output=str(e), is_error=True)

        self._emit_result(call, result)
        return idx, result

    def _emit_result(self, call: ToolCall, result: ToolResult) -> None:
        if self._bus:
            from neoagent.events import ToolResultEvent
            self._bus.emit(ToolResultEvent(
                name=call.name, call_id=call.id,
                output=result.output, is_error=result.is_error,
            ))
