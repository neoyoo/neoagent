from __future__ import annotations
import pytest
from pydantic import BaseModel
from neoagent.tools.base import BaseTool
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
        assert isinstance(input, AddInput)
        return ToolResult(call_id="test-id", output=str(input.a + input.b))

class TestBaseTool:
    def test_get_schema_has_required_keys(self):
        tool = AddTool()
        schema = tool.get_schema()
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema

    def test_get_schema_name_matches(self):
        tool = AddTool()
        assert tool.get_schema()["name"] == "add"

    def test_get_schema_description_matches(self):
        tool = AddTool()
        assert tool.get_schema()["description"] == "Add two integers"

    def test_get_schema_input_schema_has_properties(self):
        tool = AddTool()
        props = tool.get_schema()["input_schema"]["properties"]
        assert "a" in props
        assert "b" in props

    def test_get_schema_input_schema_type_is_object(self):
        tool = AddTool()
        assert tool.get_schema()["input_schema"]["type"] == "object"

    def test_permission_default_is_ask(self):
        class MinInput(BaseModel):
            x: str
        class MinTool(BaseTool):
            name: str = "min"
            description: str = "minimal"
            input_model: type[BaseModel] = MinInput
            async def execute(self, input: BaseModel) -> ToolResult:
                return ToolResult(call_id="x", output="ok")
        assert MinTool().permission == "ask"

    def test_is_concurrent_safe_default_is_false(self):
        class MinInput(BaseModel):
            x: str
        class MinTool(BaseTool):
            name: str = "min2"
            description: str = "minimal2"
            input_model: type[BaseModel] = MinInput
            async def execute(self, input: BaseModel) -> ToolResult:
                return ToolResult(call_id="x", output="ok")
        assert MinTool().is_concurrent_safe is False

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self):
        tool = AddTool()
        result = await tool.execute(AddInput(a=3, b=4))
        assert isinstance(result, ToolResult)
        assert result.output == "7"
        assert result.is_error is False

    def test_returns_external_content_default_is_false(self):
        tool = AddTool()
        assert tool.returns_external_content is False

    def test_returns_external_content_can_be_overridden(self):
        class ExternalTool(BaseTool):
            name: str = "external"
            description: str = "External content tool"
            input_model: type[BaseModel] = AddInput
            returns_external_content: bool = True

            async def execute(self, input: BaseModel) -> ToolResult:
                return ToolResult(call_id="x", output="data")

        assert ExternalTool().returns_external_content is True

    def test_returns_external_content_default_not_inherited_as_true(self):
        class NormalTool(BaseTool):
            name: str = "normal"
            description: str = "Normal tool"
            input_model: type[BaseModel] = AddInput

            async def execute(self, input: BaseModel) -> ToolResult:
                return ToolResult(call_id="x", output="ok")

        assert NormalTool().returns_external_content is False
