from __future__ import annotations
import pytest
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from neoagent.mcp.tool import MCPTool, create_mcp_tools, _schema_to_pydantic, _json_schema_to_type
from neoagent.mcp.client import MCPClient, MCPToolInfo
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult


# ── Helper: mock MCPClient ────────────────────────────────────────────────────

def _make_mock_client(name: str = "github") -> MCPClient:
    client = MagicMock(spec=MCPClient)
    client.name = name
    client.call_tool = AsyncMock(return_value="tool result text")
    return client


def _make_tool_info(
    name: str = "create_issue",
    description: str = "Create a GitHub issue",
    input_schema: dict | None = None,
) -> MCPToolInfo:
    return MCPToolInfo(
        name=name,
        description=description,
        input_schema=input_schema or {
            "type": "object",
            "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
            "required": ["title"],
        },
    )


# ── MCPTool is a BaseTool ────────────────────────────────────────────────────

def test_mcptool_is_basetool():
    client = _make_mock_client()
    info = _make_tool_info()
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    assert isinstance(tool, BaseTool)


# ── MCPTool name format ───────────────────────────────────────────────────────

def test_mcptool_name_format():
    """name must be '{server_name}__{tool_name}'."""
    client = _make_mock_client("github")
    info = _make_tool_info("create_issue")
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    assert tool.name == "github__create_issue"


def test_mcptool_name_with_different_server():
    client = _make_mock_client("filesystem")
    info = _make_tool_info("read_file")
    tool = MCPTool(client=client, server_name="filesystem", tool_info=info)
    assert tool.name == "filesystem__read_file"


# ── MCPTool description ───────────────────────────────────────────────────────

def test_mcptool_description():
    client = _make_mock_client()
    info = _make_tool_info(description="Create a GitHub issue")
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    assert tool.description == "Create a GitHub issue"


# ── MCPTool permission and concurrency ───────────────────────────────────────

def test_mcptool_permission_is_ask():
    """MCP tools default to 'ask' permission (require user confirmation)."""
    client = _make_mock_client()
    info = _make_tool_info()
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    assert tool.permission == "ask"


def test_mcptool_is_concurrent_safe_true():
    client = _make_mock_client()
    info = _make_tool_info()
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    assert tool.is_concurrent_safe is True


# ── MCPTool dynamic input_model ───────────────────────────────────────────────

def test_mcptool_input_model_is_pydantic_model():
    """input_model must be a Pydantic BaseModel subclass."""
    client = _make_mock_client()
    info = _make_tool_info()
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    assert issubclass(tool.input_model, BaseModel)


def test_mcptool_input_model_has_fields_from_schema():
    """input_model must expose fields declared in the JSON Schema."""
    client = _make_mock_client()
    info = _make_tool_info(input_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "number": {"type": "integer"},
        },
        "required": ["title"],
    })
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    fields = tool.input_model.model_fields
    assert "title" in fields
    assert "number" in fields


def test_mcptool_input_model_empty_schema_still_valid():
    """Empty schema → input_model with no required fields (accepts anything via kwargs)."""
    client = _make_mock_client()
    info = _make_tool_info(input_schema={"type": "object", "properties": {}})
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    # Should not raise
    instance = tool.input_model()
    assert isinstance(instance, BaseModel)


# ── MCPTool.execute() ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mcptool_execute_delegates_to_client():
    """execute() must call client.call_tool with tool_name (without prefix) and args."""
    client = _make_mock_client("github")
    info = _make_tool_info("create_issue", input_schema={
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    })
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    validated = tool.input_model.model_validate({"title": "Bug report"})
    result = await tool.execute(validated)

    # Must call with the MCP server's original tool name (without server prefix)
    client.call_tool.assert_called_once_with("create_issue", {"title": "Bug report"})
    assert isinstance(result, ToolResult)
    assert result.output == "tool result text"
    assert not result.is_error


@pytest.mark.asyncio
async def test_mcptool_execute_returns_toolresult():
    client = _make_mock_client()
    info = _make_tool_info(input_schema={"type": "object", "properties": {}})
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    validated = tool.input_model()
    result = await tool.execute(validated)
    assert isinstance(result, ToolResult)


@pytest.mark.asyncio
async def test_mcptool_execute_error_from_client():
    """If client.call_tool raises, execute() must return is_error=True ToolResult."""
    client = _make_mock_client()
    client.call_tool = AsyncMock(side_effect=RuntimeError("MCP server error"))
    info = _make_tool_info(input_schema={"type": "object", "properties": {}})
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    validated = tool.input_model()
    result = await tool.execute(validated)
    assert result.is_error is True
    assert "MCP server error" in result.output


# ── create_mcp_tools() helper ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_mcp_tools_returns_list_of_mcptool():
    """create_mcp_tools() must return one MCPTool per tool from list_tools()."""
    client = MagicMock(spec=MCPClient)
    client.name = "github"
    client.list_tools = AsyncMock(return_value=[
        _make_tool_info("create_issue"),
        _make_tool_info("list_repos"),
    ])

    tools = await create_mcp_tools(client=client, server_name="github")

    assert len(tools) == 2
    assert all(isinstance(t, MCPTool) for t in tools)
    assert tools[0].name == "github__create_issue"
    assert tools[1].name == "github__list_repos"


@pytest.mark.asyncio
async def test_create_mcp_tools_empty_server():
    """create_mcp_tools() with no tools returns empty list."""
    client = MagicMock(spec=MCPClient)
    client.name = "empty"
    client.list_tools = AsyncMock(return_value=[])
    tools = await create_mcp_tools(client=client, server_name="empty")
    assert tools == []


def test_mcptool_get_schema_uses_prefixed_name():
    """get_schema() (inherited from BaseTool) must use the prefixed tool name."""
    client = _make_mock_client("github")
    info = _make_tool_info("create_issue")
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    schema = tool.get_schema()
    assert schema["name"] == "github__create_issue"
    assert "input_schema" in schema


# ── _schema_to_pydantic(): enhanced JSON Schema support (Fix 8) ───────────────

def test_schema_typed_array_string():
    """array with items type string → list[str]."""
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["tags"],
    }
    model = _schema_to_pydantic("test_tool", schema)
    fields = model.model_fields
    assert "tags" in fields
    # Validate that list[str] is accepted
    instance = model.model_validate({"tags": ["a", "b", "c"]})
    assert instance.tags == ["a", "b", "c"]


def test_schema_typed_array_integer():
    """array with items type integer → list[int]."""
    schema = {
        "type": "object",
        "properties": {
            "ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["ids"],
    }
    model = _schema_to_pydantic("test_tool", schema)
    instance = model.model_validate({"ids": [1, 2, 3]})
    assert instance.ids == [1, 2, 3]


def test_schema_untyped_array_fallback_to_list():
    """array without items → plain list (no item type constraint)."""
    schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array"},
        },
    }
    model = _schema_to_pydantic("test_tool", schema)
    instance = model.model_validate({"items": [1, "two", True]})
    assert instance.items == [1, "two", True]


def test_schema_nested_object_creates_pydantic_model():
    """object type with nested properties → nested Pydantic model."""
    schema = {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "number": {"type": "integer"},
                },
                "required": ["street"],
            },
        },
        "required": ["address"],
    }
    model = _schema_to_pydantic("test_tool", schema)
    instance = model.model_validate({"address": {"street": "Main St", "number": 42}})
    assert instance.address.street == "Main St"
    assert instance.address.number == 42


def test_schema_nested_object_without_properties_falls_back_to_dict():
    """object type with no properties → dict fallback."""
    schema = {
        "type": "object",
        "properties": {
            "metadata": {"type": "object"},
        },
    }
    model = _schema_to_pydantic("test_tool", schema)
    instance = model.model_validate({"metadata": {"key": "value"}})
    assert instance.metadata == {"key": "value"}


def test_schema_enum_creates_literal():
    """enum field → Literal[...] type annotation."""
    schema = {
        "type": "object",
        "properties": {
            "color": {"enum": ["red", "green", "blue"]},
        },
        "required": ["color"],
    }
    model = _schema_to_pydantic("test_tool", schema)
    # Valid enum value should be accepted
    instance = model.model_validate({"color": "red"})
    assert instance.color == "red"


def test_schema_field_with_default():
    """field with 'default' key → optional with that default."""
    schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 10},
        },
    }
    model = _schema_to_pydantic("test_tool", schema)
    # No value provided → default kicks in
    instance = model.model_validate({})
    assert instance.limit == 10


def test_schema_required_field_no_default():
    """Required field without default → must be provided."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
        },
        "required": ["name"],
    }
    model = _schema_to_pydantic("test_tool", schema)
    import pytest
    with pytest.raises(Exception):
        model.model_validate({})  # should fail — required field missing


def test_schema_unknown_type_falls_back_to_any():
    """Unknown JSON Schema type → Any (accept anything)."""
    schema = {
        "type": "object",
        "properties": {
            "data": {"type": "unknown_custom_type"},
        },
    }
    model = _schema_to_pydantic("test_tool", schema)
    instance = model.model_validate({"data": {"arbitrary": "value"}})
    assert instance.data == {"arbitrary": "value"}


def test_schema_empty_enum_falls_back_to_any():
    """Empty enum list → Any fallback."""
    schema = {
        "type": "object",
        "properties": {
            "val": {"enum": []},
        },
    }
    # Should not raise
    model = _schema_to_pydantic("test_tool", schema)
    assert model is not None


def test_mcptool_with_typed_array_schema():
    """MCPTool built from a schema with typed arrays validates correctly."""
    client = _make_mock_client("github")
    info = _make_tool_info("list_users", input_schema={
        "type": "object",
        "properties": {
            "group_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["group_ids"],
    })
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    instance = tool.input_model.model_validate({"group_ids": [1, 2, 3]})
    assert instance.group_ids == [1, 2, 3]


def test_mcptool_with_enum_schema():
    """MCPTool built from a schema with enum validates correctly."""
    client = _make_mock_client("github")
    info = _make_tool_info("set_status", input_schema={
        "type": "object",
        "properties": {
            "status": {"enum": ["open", "closed", "pending"]},
        },
        "required": ["status"],
    })
    tool = MCPTool(client=client, server_name="github", tool_info=info)
    instance = tool.input_model.model_validate({"status": "open"})
    assert instance.status == "open"
