from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from neoagent.tools.executor import ToolExecutor
from neoagent.tools.registry import ToolRegistry
from neoagent.tools.permission import PermissionChecker
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolCall, ToolResult
from neoagent.events import EventBus, ToolCallEvent, ToolResultEvent


class EchoInput(BaseModel):
    text: str


class EchoTool(BaseTool):
    name: str = "echo"
    description: str = "echoes input"
    input_model: type[BaseModel] = EchoInput
    permission: str = "auto"
    is_concurrent_safe: bool = True

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, EchoInput)
        return ToolResult(call_id="", output=input.text)


class DenyTool(BaseTool):
    name: str = "deny_tool"
    description: str = "always denied"
    input_model: type[BaseModel] = EchoInput
    permission: str = "deny"

    async def execute(self, input: BaseModel) -> ToolResult:
        return ToolResult(call_id="", output="should not reach")


class BrokenTool(BaseTool):
    name: str = "broken"
    description: str = "always throws"
    input_model: type[BaseModel] = EchoInput
    permission: str = "auto"

    async def execute(self, input: BaseModel) -> ToolResult:
        raise RuntimeError("tool is broken")


@pytest.fixture
def registry_with_echo() -> ToolRegistry:
    r = ToolRegistry()
    r.register(EchoTool())
    return r


@pytest.fixture
def executor(registry_with_echo: ToolRegistry) -> ToolExecutor:
    return ToolExecutor(
        registry=registry_with_echo,
        permission_checker=PermissionChecker(auto_approve=True),
        max_result_size=100,
    )


# ── Basic execution ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_single_call(executor: ToolExecutor):
    results = await executor.execute([ToolCall(id="c1", name="echo", input={"text": "hello"})])
    assert len(results) == 1
    assert results[0].output == "hello"
    assert results[0].call_id == "c1"
    assert not results[0].is_error


@pytest.mark.asyncio
async def test_execute_unknown_tool(executor: ToolExecutor):
    results = await executor.execute([ToolCall(id="c2", name="nonexistent", input={})])
    assert results[0].is_error
    assert "not found" in results[0].output


@pytest.mark.asyncio
async def test_execute_denied_tool():
    r = ToolRegistry()
    r.register(DenyTool())
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker())
    results = await ex.execute([ToolCall(id="c3", name="deny_tool", input={"text": "x"})])
    assert results[0].is_error
    assert "Permission denied" in results[0].output


@pytest.mark.asyncio
async def test_execute_broken_tool():
    r = ToolRegistry()
    r.register(BrokenTool())
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True))
    results = await ex.execute([ToolCall(id="c4", name="broken", input={"text": "x"})])
    assert results[0].is_error
    assert "tool is broken" in results[0].output


# ── Truncation ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_output_truncated_at_max_result_size():
    r = ToolRegistry()
    r.register(EchoTool())
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), max_result_size=10)
    results = await ex.execute([ToolCall(id="c5", name="echo", input={"text": "x" * 50})])
    assert len(results[0].output) <= 10 + len("\n[truncated]")
    assert "[truncated]" in results[0].output


# ── Concurrency ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_safe_tools_run_in_parallel():
    """Multiple concurrent-safe calls should produce correct results (order preserved)."""
    r = ToolRegistry()
    r.register(EchoTool())
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True))
    calls = [
        ToolCall(id=f"c{i}", name="echo", input={"text": str(i)})
        for i in range(5)
    ]
    results = await ex.execute(calls)
    for i, r in enumerate(results):
        assert r.output == str(i)


# ── EventBus integration ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_executor_emits_tool_call_event():
    r = ToolRegistry()
    r.register(EchoTool())
    bus = EventBus()
    received = []
    bus.subscribe(ToolCallEvent, lambda e: received.append(e))
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), event_bus=bus)
    await ex.execute([ToolCall(id="c6", name="echo", input={"text": "hi"})])
    assert len(received) == 1
    assert received[0].name == "echo"
    assert received[0].call_id == "c6"


@pytest.mark.asyncio
async def test_executor_emits_tool_result_event():
    r = ToolRegistry()
    r.register(EchoTool())
    bus = EventBus()
    received = []
    bus.subscribe(ToolResultEvent, lambda e: received.append(e))
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), event_bus=bus)
    await ex.execute([ToolCall(id="c7", name="echo", input={"text": "world"})])
    assert len(received) == 1
    assert received[0].output == "world"
    assert not received[0].is_error


@pytest.mark.asyncio
async def test_executor_emits_result_event_on_error():
    r = ToolRegistry()
    r.register(BrokenTool())
    bus = EventBus()
    received = []
    bus.subscribe(ToolResultEvent, lambda e: received.append(e))
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), event_bus=bus)
    await ex.execute([ToolCall(id="c8", name="broken", input={"text": "x"})])
    assert received[0].is_error


# ── Fix 1: ToolCallEvent input_data is a deepcopy ────────────────────────────

@pytest.mark.asyncio
async def test_tool_call_event_input_data_is_deepcopy():
    """Handler mutation of ToolCallEvent.input_data must not affect actual execution."""
    r = ToolRegistry()
    r.register(EchoTool())
    bus = EventBus()

    original_input = {"text": "original"}
    mutated_in_handler: dict = {}

    def mutating_handler(event: ToolCallEvent):
        # Mutate the event's input_data — must NOT affect execution
        event.input_data["text"] = "mutated"
        mutated_in_handler.update(event.input_data)

    bus.subscribe(ToolCallEvent, mutating_handler)
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), event_bus=bus)
    results = await ex.execute([ToolCall(id="c9", name="echo", input=original_input)])

    # Handler received the event (and mutated its copy)
    assert mutated_in_handler["text"] == "mutated"
    # But actual execution used the original value
    assert results[0].output == "original"
    assert not results[0].is_error


@pytest.mark.asyncio
async def test_tool_call_event_input_data_is_not_same_object():
    """ToolCallEvent.input_data must be a different object from call.input."""
    r = ToolRegistry()
    r.register(EchoTool())
    bus = EventBus()
    event_input_ids: list[int] = []
    bus.subscribe(ToolCallEvent, lambda e: event_input_ids.append(id(e.input_data)))

    call_input = {"text": "hello"}
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), event_bus=bus)
    await ex.execute([ToolCall(id="c10", name="echo", input=call_input)])

    assert len(event_input_ids) == 1
    assert event_input_ids[0] != id(call_input)
