from __future__ import annotations
import asyncio
from neoagent.core.types import ToolCall, ToolResult
from neoagent.tools.base import BaseTool
from neoagent.tools.permission import PermissionChecker


class ToolRegistry:
    def __init__(
        self,
        max_result_size: int = 50000,
        permission_checker: PermissionChecker | None = None,
    ):
        self._tools: dict[str, BaseTool] = {}
        self.max_result_size = max_result_size
        self._permission_checker = permission_checker or PermissionChecker()

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def get_schemas(self) -> list[dict]:
        return [t.get_schema() for t in self._tools.values() if t.permission != "deny"]

    async def execute(self, calls: list[ToolCall]) -> list[ToolResult]:
        safe, unsafe = self._partition_by_concurrency(calls)
        results: dict[int, ToolResult] = {}

        # Run concurrent-safe calls in parallel
        if safe:
            safe_results = await asyncio.gather(
                *[self._run_one(idx, call) for idx, call in safe]
            )
            for idx, result in safe_results:
                results[idx] = result

        # Run unsafe calls serially
        for idx, call in unsafe:
            _, result = await self._run_one(idx, call)
            results[idx] = result

        return [results[i] for i in range(len(calls))]

    def _partition_by_concurrency(self, calls: list[ToolCall]) -> tuple[list, list]:
        safe = []
        unsafe = []
        for idx, call in enumerate(calls):
            tool = self._tools.get(call.name)
            if tool and tool.is_concurrent_safe:
                safe.append((idx, call))
            else:
                unsafe.append((idx, call))
        return safe, unsafe

    async def _run_one(self, idx: int, call: ToolCall) -> tuple[int, ToolResult]:
        tool = self._tools.get(call.name)
        if tool is None:
            return idx, ToolResult(call_id=call.id, output=f"Tool not found: {call.name}", is_error=True)
        try:
            validated_input = tool.input_model.model_validate(call.input)
            allowed = await self._permission_checker.check(tool, validated_input)
            if not allowed:
                if tool.permission == "deny":
                    msg = f"Permission denied: tool '{tool.name}' is disabled"
                else:
                    msg = f"Permission denied: tool '{tool.name}' requires user approval"
                return idx, ToolResult(call_id=call.id, output=msg, is_error=True)
            result = await tool.execute(validated_input)
            result.call_id = call.id
            if len(result.output) > self.max_result_size:
                result.output = result.output[:self.max_result_size] + "\n[truncated]"
            return idx, result
        except Exception as e:
            return idx, ToolResult(call_id=call.id, output=str(e), is_error=True)
