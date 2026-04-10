from __future__ import annotations
import asyncio
import pytest
from pydantic import BaseModel
from neoagent.tools.base import BaseTool
from neoagent.tools.registry import ToolRegistry
from neoagent.core.types import ToolCall, ToolResult

class AddInput(BaseModel):
    a: int
    b: int

class AddTool(BaseTool):
    name: str = "add"
    description: str = "Add two integers"
    input_model: type[BaseModel] = AddInput
    permission: str = "auto"
    is_concurrent_safe: bool = True
    async def execute(self, input: BaseModel) -> ToolResult:
        return ToolResult(call_id="x", output=str(input.a + input.b))

class SlowTool(BaseTool):
    name: str = "slow"
    description: str = "Slow tool for concurrency test"
    input_model: type[BaseModel] = AddInput
    permission: str = "auto"
    is_concurrent_safe: bool = True
    async def execute(self, input: BaseModel) -> ToolResult:
        await asyncio.sleep(0.1)
        return ToolResult(call_id="x", output="slow_done")

class UnsafeTool(BaseTool):
    name: str = "unsafe"
    description: str = "Not concurrent safe"
    input_model: type[BaseModel] = AddInput
    permission: str = "auto"
    is_concurrent_safe: bool = False
    async def execute(self, input: BaseModel) -> ToolResult:
        return ToolResult(call_id="x", output="unsafe_done")

class DenyTool(BaseTool):
    name: str = "denied"
    description: str = "Denied tool"
    input_model: type[BaseModel] = AddInput
    permission: str = "deny"
    async def execute(self, input: BaseModel) -> ToolResult:
        return ToolResult(call_id="x", output="should not run")

class ErrorTool(BaseTool):
    name: str = "error"
    description: str = "Always errors"
    input_model: type[BaseModel] = AddInput
    permission: str = "auto"
    async def execute(self, input: BaseModel) -> ToolResult:
        raise RuntimeError("boom")

class TestToolRegistryRegister:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(AddTool())
        assert reg.get_tool("add") is not None

    def test_get_nonexistent(self):
        reg = ToolRegistry()
        assert reg.get_tool("nope") is None

    def test_duplicate_name_raises(self):
        reg = ToolRegistry()
        reg.register(AddTool())
        with pytest.raises(ValueError):
            reg.register(AddTool())

class TestToolRegistrySchemas:
    def test_get_schemas_returns_registered(self):
        reg = ToolRegistry()
        reg.register(AddTool())
        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "add"

    def test_deny_tool_excluded_from_schemas(self):
        reg = ToolRegistry()
        reg.register(AddTool())
        reg.register(DenyTool())
        schemas = reg.get_schemas()
        names = [s["name"] for s in schemas]
        assert "add" in names
        assert "denied" not in names

class TestToolRegistryExecute:
    @pytest.mark.asyncio
    async def test_execute_single_tool(self):
        reg = ToolRegistry()
        reg.register(AddTool())
        calls = [ToolCall(id="c1", name="add", input={"a": 2, "b": 3})]
        results = await reg.execute(calls)
        assert len(results) == 1
        assert results[0].output == "5"
        assert results[0].call_id == "c1"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        calls = [ToolCall(id="c1", name="unknown", input={})]
        results = await reg.execute(calls)
        assert len(results) == 1
        assert results[0].is_error is True
        assert "unknown" in results[0].output.lower() or "not found" in results[0].output.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self):
        reg = ToolRegistry()
        reg.register(ErrorTool())
        calls = [ToolCall(id="c1", name="error", input={"a": 1, "b": 2})]
        results = await reg.execute(calls)
        assert results[0].is_error is True
        assert "boom" in results[0].output

    @pytest.mark.asyncio
    async def test_concurrent_safe_tools_run_parallel(self):
        reg = ToolRegistry()
        reg.register(SlowTool())
        # Two slow calls should run in parallel (~0.1s not ~0.2s)
        calls = [
            ToolCall(id="c1", name="slow", input={"a": 1, "b": 1}),
            ToolCall(id="c2", name="slow", input={"a": 2, "b": 2}),
        ]
        import time
        start = time.monotonic()
        results = await reg.execute(calls)
        elapsed = time.monotonic() - start
        assert len(results) == 2
        assert elapsed < 0.18  # parallel, not serial

    @pytest.mark.asyncio
    async def test_unsafe_tools_run_serial(self):
        reg = ToolRegistry()
        reg.register(UnsafeTool())
        calls = [
            ToolCall(id="c1", name="unsafe", input={"a": 1, "b": 1}),
            ToolCall(id="c2", name="unsafe", input={"a": 2, "b": 2}),
        ]
        results = await reg.execute(calls)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_result_truncation(self):
        class BigTool(BaseTool):
            name: str = "big"
            description: str = "Returns big output"
            input_model: type[BaseModel] = AddInput
            permission: str = "auto"
            async def execute(self, input: BaseModel) -> ToolResult:
                return ToolResult(call_id="x", output="x" * 100000)
        reg = ToolRegistry(max_result_size=1000)
        reg.register(BigTool())
        calls = [ToolCall(id="c1", name="big", input={"a": 1, "b": 1})]
        results = await reg.execute(calls)
        assert len(results[0].output) <= 1100  # some overhead for truncation notice
        assert "[truncated]" in results[0].output
