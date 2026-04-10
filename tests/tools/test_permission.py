from __future__ import annotations
import pytest
from pydantic import BaseModel
from neoagent.tools.base import BaseTool
from neoagent.tools.permission import PermissionChecker
from neoagent.core.types import ToolResult

class DummyInput(BaseModel):
    x: str

class AutoTool(BaseTool):
    name: str = "auto_tool"
    description: str = "auto permission"
    input_model: type[BaseModel] = DummyInput
    permission: str = "auto"
    async def execute(self, input: BaseModel) -> ToolResult:
        return ToolResult(call_id="x", output="ok")

class AskTool(BaseTool):
    name: str = "ask_tool"
    description: str = "ask permission"
    input_model: type[BaseModel] = DummyInput
    permission: str = "ask"
    async def execute(self, input: BaseModel) -> ToolResult:
        return ToolResult(call_id="x", output="ok")

class DenyTool(BaseTool):
    name: str = "deny_tool"
    description: str = "deny permission"
    input_model: type[BaseModel] = DummyInput
    permission: str = "deny"
    async def execute(self, input: BaseModel) -> ToolResult:
        return ToolResult(call_id="x", output="ok")

class TestPermissionChecker:
    @pytest.mark.asyncio
    async def test_auto_returns_true(self):
        checker = PermissionChecker()
        result = await checker.check(AutoTool(), DummyInput(x="hi"))
        assert result is True

    @pytest.mark.asyncio
    async def test_deny_returns_false(self):
        checker = PermissionChecker()
        result = await checker.check(DenyTool(), DummyInput(x="hi"))
        assert result is False

    @pytest.mark.asyncio
    async def test_ask_no_callback_defaults_true(self):
        checker = PermissionChecker()
        result = await checker.check(AskTool(), DummyInput(x="hi"))
        assert result is True

    @pytest.mark.asyncio
    async def test_ask_with_callback_approved(self):
        async def approve(name, desc, data):
            return True
        checker = PermissionChecker(ask_callback=approve)
        result = await checker.check(AskTool(), DummyInput(x="hi"))
        assert result is True

    @pytest.mark.asyncio
    async def test_ask_with_callback_denied(self):
        async def deny(name, desc, data):
            return False
        checker = PermissionChecker(ask_callback=deny)
        result = await checker.check(AskTool(), DummyInput(x="hi"))
        assert result is False

    @pytest.mark.asyncio
    async def test_ask_callback_receives_correct_args(self):
        received = {}
        async def capture(name, desc, data):
            received["name"] = name
            received["desc"] = desc
            received["data"] = data
            return True
        checker = PermissionChecker(ask_callback=capture)
        await checker.check(AskTool(), DummyInput(x="hello"))
        assert received["name"] == "ask_tool"
        assert received["desc"] == "ask permission"
        assert received["data"] == {"x": "hello"}

    @pytest.mark.asyncio
    async def test_auto_does_not_call_callback(self):
        called = False
        async def should_not_call(name, desc, data):
            nonlocal called
            called = True
            return True
        checker = PermissionChecker(ask_callback=should_not_call)
        await checker.check(AutoTool(), DummyInput(x="hi"))
        assert called is False

    @pytest.mark.asyncio
    async def test_deny_does_not_call_callback(self):
        called = False
        async def should_not_call(name, desc, data):
            nonlocal called
            called = True
            return True
        checker = PermissionChecker(ask_callback=should_not_call)
        await checker.check(DenyTool(), DummyInput(x="hi"))
        assert called is False
