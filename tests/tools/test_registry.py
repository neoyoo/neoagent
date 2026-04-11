from __future__ import annotations
import pytest
from pydantic import BaseModel
from neoagent.tools.base import BaseTool
from neoagent.tools.registry import ToolRegistry
from neoagent.core.types import ToolResult


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


class DenyTool(BaseTool):
    name: str = "denied"
    description: str = "Denied tool"
    input_model: type[BaseModel] = AddInput
    permission: str = "deny"

    async def execute(self, input: BaseModel) -> ToolResult:
        return ToolResult(call_id="x", output="should not run")


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


class TestAllTools:
    def test_all_tools_returns_dict(self):
        reg = ToolRegistry()
        reg.register(AddTool())
        reg.register(DenyTool())
        all_tools = reg.all_tools()
        assert "add" in all_tools
        assert "denied" in all_tools
        assert len(all_tools) == 2

    def test_all_tools_empty_registry(self):
        reg = ToolRegistry()
        assert reg.all_tools() == {}


# ── Registry purity tests (Task 6) ───────────────────────────────────────────

def test_registry_has_no_execute_method():
    """After Task 6, ToolRegistry must not have execute()."""
    from neoagent.tools.registry import ToolRegistry
    r = ToolRegistry()
    assert not hasattr(r, 'execute'), \
        "execute() must be removed from ToolRegistry — it belongs in ToolExecutor"


def test_registry_has_no_max_result_size():
    """After Task 6, ToolRegistry must not own max_result_size."""
    from neoagent.tools.registry import ToolRegistry
    r = ToolRegistry()
    assert not hasattr(r, 'max_result_size'), \
        "max_result_size must be in ToolExecutor, not ToolRegistry"


def test_registry_has_no_permission_checker():
    """After Task 6, ToolRegistry must not own permission_checker."""
    from neoagent.tools.registry import ToolRegistry
    r = ToolRegistry()
    assert not hasattr(r, '_permission_checker'), \
        "_permission_checker must be in ToolExecutor, not ToolRegistry"
