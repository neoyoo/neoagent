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
from neoagent.hooks import HookManager, HookResult, PreToolCallEvent, PostToolCallEvent


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


# ── Fix 1: ToolCallEvent input_data is MappingProxyType (immutable) ──────────

@pytest.mark.asyncio
async def test_tool_call_event_input_data_is_immutable():
    """ToolCallEvent.input_data must be a MappingProxyType — handler mutation raises TypeError."""
    from types import MappingProxyType
    r = ToolRegistry()
    r.register(EchoTool())
    bus = EventBus()

    received_events: list[ToolCallEvent] = []
    bus.subscribe(ToolCallEvent, lambda e: received_events.append(e))

    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), event_bus=bus)
    await ex.execute([ToolCall(id="c9", name="echo", input={"text": "original"})])

    assert len(received_events) == 1
    assert isinstance(received_events[0].input_data, MappingProxyType)
    # Mutation must raise TypeError
    with pytest.raises(TypeError):
        received_events[0].input_data["text"] = "mutated"  # type: ignore[index]


@pytest.mark.asyncio
async def test_tool_call_event_input_data_does_not_affect_execution():
    """Even though input_data is immutable, actual tool execution uses the original input."""
    r = ToolRegistry()
    r.register(EchoTool())
    bus = EventBus()
    bus.subscribe(ToolCallEvent, lambda e: None)  # subscriber exists but can't mutate

    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), event_bus=bus)
    results = await ex.execute([ToolCall(id="c9b", name="echo", input={"text": "original"})])

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


# ── Hook integration: pre_tool_call ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_pre_tool_call_deny_blocks_execution():
    """pre_tool_call hook returning deny must prevent tool execution."""
    r = ToolRegistry()
    r.register(EchoTool())
    hook_mgr = HookManager()

    async def block_all(event: PreToolCallEvent) -> HookResult:
        return HookResult.deny("test block")

    hook_mgr.register("pre_tool_call", block_all)

    ex = ToolExecutor(
        registry=r,
        permission_checker=PermissionChecker(auto_approve=True),
        hook_manager=hook_mgr,
    )
    results = await ex.execute([ToolCall(id="h1", name="echo", input={"text": "hello"})])
    assert results[0].is_error
    assert "test block" in results[0].output


@pytest.mark.asyncio
async def test_pre_tool_call_deny_does_not_call_execute():
    """When pre hook denies, tool.execute() must never be called."""
    r = ToolRegistry()
    execute_called = []

    class TrackingEcho(BaseTool):
        name: str = "track_echo"
        description: str = "tracks execute calls"
        input_model: type[BaseModel] = EchoInput
        permission: str = "auto"
        is_concurrent_safe: bool = True

        async def execute(self, input: BaseModel) -> ToolResult:
            execute_called.append(True)
            return ToolResult(call_id="", output=input.text)

    r.register(TrackingEcho())
    hook_mgr = HookManager()

    async def denier(event):
        return HookResult.deny("blocked")

    hook_mgr.register("pre_tool_call", denier)
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), hook_manager=hook_mgr)
    await ex.execute([ToolCall(id="h2", name="track_echo", input={"text": "x"})])
    assert execute_called == []


@pytest.mark.asyncio
async def test_pre_tool_call_modify_changes_input():
    """pre_tool_call modify must replace the tool input before execution."""
    r = ToolRegistry()
    r.register(EchoTool())
    hook_mgr = HookManager()

    async def replace_input(event: PreToolCallEvent) -> HookResult:
        return HookResult.modify({"tool_input": {"text": "REPLACED"}})

    hook_mgr.register("pre_tool_call", replace_input)
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), hook_manager=hook_mgr)
    results = await ex.execute([ToolCall(id="h3", name="echo", input={"text": "original"})])
    assert results[0].output == "REPLACED"
    assert not results[0].is_error


@pytest.mark.asyncio
async def test_pre_tool_call_allow_proceeds_normally():
    """pre_tool_call allow must not change behavior."""
    r = ToolRegistry()
    r.register(EchoTool())
    hook_mgr = HookManager()

    async def allow_all(event):
        return HookResult.allow()

    hook_mgr.register("pre_tool_call", allow_all)
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), hook_manager=hook_mgr)
    results = await ex.execute([ToolCall(id="h4", name="echo", input={"text": "unchanged"})])
    assert results[0].output == "unchanged"


# ── Hook integration: post_tool_call ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_post_tool_call_modify_changes_output():
    """post_tool_call modify must replace the result.output."""
    r = ToolRegistry()
    r.register(EchoTool())
    hook_mgr = HookManager()

    async def replace_output(event: PostToolCallEvent) -> HookResult:
        return HookResult.modify({"result": "OVERRIDDEN"})

    hook_mgr.register("post_tool_call", replace_output)
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), hook_manager=hook_mgr)
    results = await ex.execute([ToolCall(id="h5", name="echo", input={"text": "hello"})])
    assert results[0].output == "OVERRIDDEN"


@pytest.mark.asyncio
async def test_post_tool_call_deny_is_ignored():
    """post_tool_call deny must not change the result (execution already done)."""
    r = ToolRegistry()
    r.register(EchoTool())
    hook_mgr = HookManager()

    async def deny_post(event):
        return HookResult.deny("too late")

    hook_mgr.register("post_tool_call", deny_post)
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True), hook_manager=hook_mgr)
    results = await ex.execute([ToolCall(id="h6", name="echo", input={"text": "real"})])
    # Result should be the actual output, not an error
    assert results[0].output == "real"
    assert not results[0].is_error


@pytest.mark.asyncio
async def test_no_hook_manager_still_works():
    """ToolExecutor with no hook_manager must behave identically to before."""
    r = ToolRegistry()
    r.register(EchoTool())
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(auto_approve=True))
    results = await ex.execute([ToolCall(id="h7", name="echo", input={"text": "baseline"})])
    assert results[0].output == "baseline"
    assert not results[0].is_error


@pytest.mark.asyncio
async def test_pre_hook_runs_after_permission_check():
    """pre_tool_call hook must NOT run if PermissionChecker denies first."""
    r = ToolRegistry()
    r.register(DenyTool())  # permission="deny" tool
    hook_mgr = HookManager()
    hook_called = []

    async def hook_observer(event):
        hook_called.append(True)
        return HookResult.allow()

    hook_mgr.register("pre_tool_call", hook_observer)
    ex = ToolExecutor(registry=r, permission_checker=PermissionChecker(), hook_manager=hook_mgr)
    results = await ex.execute([ToolCall(id="h8", name="deny_tool", input={"text": "x"})])

    # PermissionChecker denied — hook should not run
    assert hook_called == []
    assert results[0].is_error
    assert "Permission denied" in results[0].output


# ── Security: permission re-check after hook modify ───────────────────────────

@pytest.mark.asyncio
async def test_pre_tool_call_modify_reruns_permission_check():
    """After pre_tool_call modify, PermissionChecker must run again on modified input.

    Security: a hook must not be able to bypass permission enforcement by
    substituting input after the initial check has already passed.
    """
    r = ToolRegistry()
    r.register(EchoTool())
    hook_mgr = HookManager()

    # Count how many times permission check is called
    check_call_count = []

    class CountingPermissionChecker(PermissionChecker):
        async def check(self, tool, input):
            check_call_count.append(dict(input))
            return await super().check(tool, input)

    async def modifying_hook(event):
        return HookResult.modify({"tool_input": {"text": "MODIFIED"}})

    hook_mgr.register("pre_tool_call", modifying_hook)
    ex = ToolExecutor(
        registry=r,
        permission_checker=CountingPermissionChecker(auto_approve=True),
        hook_manager=hook_mgr,
    )
    results = await ex.execute([ToolCall(id="sec1", name="echo", input={"text": "original"})])

    # Permission must be checked twice: once before and once after hook modify
    assert len(check_call_count) == 2
    # First check on original input, second on modified input
    assert check_call_count[0] == {"text": "original"}
    assert check_call_count[1] == {"text": "MODIFIED"}
    # Execution uses the modified input
    assert results[0].output == "MODIFIED"
    assert not results[0].is_error


@pytest.mark.asyncio
async def test_pre_tool_call_modify_permission_denied_on_modified_input():
    """If PermissionChecker denies the modified input, the call must be blocked."""

    class AskEchoTool(BaseTool):
        """Echo tool with permission='ask' so checker outcome controls access."""
        name: str = "ask_echo"
        description: str = "echo with ask permission"
        input_model: type[EchoInput] = EchoInput
        permission: str = "ask"
        is_concurrent_safe: bool = True

        async def execute(self, input) -> ToolResult:
            return ToolResult(call_id="", output=input.text)

    r = ToolRegistry()
    r.register(AskEchoTool())
    hook_mgr = HookManager()

    call_index = [0]

    async def first_approve_then_deny(tool_name, tool_desc, input_dict):
        idx = call_index[0]
        call_index[0] += 1
        # Approve the first check (original input), deny the second (modified input)
        return idx == 0

    async def modifying_hook(event):
        return HookResult.modify({"tool_input": {"text": "DANGEROUS"}})

    hook_mgr.register("pre_tool_call", modifying_hook)
    ex = ToolExecutor(
        registry=r,
        permission_checker=PermissionChecker(ask_callback=first_approve_then_deny),
        hook_manager=hook_mgr,
    )
    results = await ex.execute([ToolCall(id="sec2", name="ask_echo", input={"text": "safe"})])

    # Second permission check (on modified input) must deny the call
    assert results[0].is_error
    assert "Permission denied" in results[0].output
    # Tool must NOT have executed — no output from the tool itself
    assert "DANGEROUS" not in results[0].output
