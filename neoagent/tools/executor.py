from __future__ import annotations
import asyncio
import copy
import logging
from typing import TYPE_CHECKING

from neoagent.core.types import ToolCall, ToolResult
from neoagent.tools.permission import PermissionChecker
from neoagent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from neoagent.events import EventBus

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
    ) -> None:
        self._registry = registry
        self._permission = permission_checker or PermissionChecker()
        self.max_result_size = max_result_size
        self._bus = event_bus

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

        # Emit ToolCallEvent before execution; deepcopy to prevent handler mutation
        if self._bus:
            from neoagent.events import ToolCallEvent
            self._bus.emit(ToolCallEvent(name=call.name, input_data=copy.deepcopy(call.input), call_id=call.id))

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
            else:
                result = await tool.execute(validated_input)
                result.call_id = call.id
                if len(result.output) > self.max_result_size:
                    result.output = result.output[:self.max_result_size] + "\n[truncated]"
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
