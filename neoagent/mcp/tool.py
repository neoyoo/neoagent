from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Literal, get_args
from pydantic import BaseModel, Field, create_model
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult

if TYPE_CHECKING:
    from neoagent.mcp.client import MCPClient, MCPToolInfo

logger = logging.getLogger(__name__)

# Counter for generating unique nested model names
_nested_model_counter = 0


def _unique_model_name(base: str) -> str:
    global _nested_model_counter
    _nested_model_counter += 1
    return f"{base}_{_nested_model_counter}"


def _json_schema_to_type(field_schema: dict, model_name_prefix: str) -> Any:
    """Recursively convert a JSON Schema dict to a Python/Pydantic type annotation.

    Supported conversions:
    - ``{"type": "string/integer/number/boolean"}`` → str/int/float/bool
    - ``{"type": "array", "items": <schema>}`` → list[<item_type>]
    - ``{"type": "object", "properties": {...}}`` → nested Pydantic model
    - ``{"enum": [...]}`` → Literal[...]
    - Fallback: Any
    """
    # Enum: Literal[val1, val2, ...]
    if "enum" in field_schema:
        enum_vals = tuple(field_schema["enum"])
        if enum_vals:
            return Literal[enum_vals]  # type: ignore[valid-type]
        return Any

    json_type = field_schema.get("type")

    if json_type == "string":
        return str
    if json_type == "integer":
        return int
    if json_type == "number":
        return float
    if json_type == "boolean":
        return bool

    if json_type == "array":
        items_schema = field_schema.get("items")
        if items_schema and isinstance(items_schema, dict):
            item_type = _json_schema_to_type(items_schema, model_name_prefix + "_item")
            return list[item_type]  # type: ignore[valid-type]
        return list

    if json_type == "object":
        nested_props = field_schema.get("properties")
        if nested_props and isinstance(nested_props, dict):
            return _build_pydantic_model(
                _unique_model_name(model_name_prefix),
                field_schema,
            )
        return dict

    return Any


def _build_pydantic_model(model_name: str, input_schema: dict) -> type[BaseModel]:
    """Build a Pydantic model from a JSON Schema object definition.

    Supports scalar types, typed arrays, nested objects, enums, and defaults.
    Falls back to Any for unknown types.
    """
    properties: dict = input_schema.get("properties", {})
    required: set[str] = set(input_schema.get("required", []))

    field_definitions: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        py_type = _json_schema_to_type(field_schema, f"{model_name}_{field_name}")
        has_default = "default" in field_schema
        default_val = field_schema.get("default")
        description = field_schema.get("description", "")

        if field_name in required:
            if description:
                field_definitions[field_name] = (py_type, Field(..., description=description))
            else:
                field_definitions[field_name] = (py_type, ...)
        else:
            if has_default:
                field_definitions[field_name] = (
                    py_type,
                    Field(default_val, description=description) if description else default_val,
                )
            else:
                field_definitions[field_name] = (
                    py_type | None,
                    Field(None, description=description) if description else None,
                )

    return create_model(model_name, **field_definitions)  # type: ignore[call-overload]


def _schema_to_pydantic(tool_name: str, input_schema: dict) -> type[BaseModel]:
    """Build a Pydantic model class from a JSON Schema dict for an MCP tool.

    Supports scalar types, typed arrays (list[str] etc.), nested objects as
    recursive Pydantic models, enums as Literal[...], and field defaults.
    Falls back to Any for truly unknown schemas.
    """
    model_name = f"MCPInput_{tool_name.replace('__', '_').replace('-', '_')}"
    return _build_pydantic_model(model_name, input_schema)


class MCPTool(BaseTool):
    """Wraps an MCP tool as a BaseTool so it can run through ToolExecutor.

    name format: "{server_name}__{tool_name}" to prevent registry collisions.
    permission defaults to "ask" — MCP tools require user confirmation.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    permission: str = "ask"
    is_concurrent_safe: bool = True

    def __init__(
        self,
        client: "MCPClient",
        server_name: str,
        tool_info: "MCPToolInfo",
    ) -> None:
        self._client = client
        self._tool_name = tool_info.name  # original name for MCP call (no prefix)
        # Public BaseTool attributes
        self.name = f"{server_name}__{tool_info.name}"
        self.description = tool_info.description
        self.input_model = _schema_to_pydantic(self.name, tool_info.input_schema)

    async def execute(self, input: BaseModel) -> ToolResult:
        """Delegate to MCPClient.call_tool() and return a ToolResult."""
        args = input.model_dump(exclude_none=True)
        try:
            text = await self._client.call_tool(self._tool_name, args)
            return ToolResult(call_id="", output=text)
        except Exception as exc:
            logger.warning("MCPTool '%s': call failed: %s", self.name, exc)
            return ToolResult(call_id="", output=str(exc), is_error=True)


async def create_mcp_tools(client: "MCPClient", server_name: str) -> list[MCPTool]:
    """Fetch tools from MCPClient and return them as MCPTool instances.

    Args:
        client:      Connected MCPClient.
        server_name: Logical name for this server (used as tool name prefix).

    Returns:
        list[MCPTool] — one per tool reported by the server.
    """
    tool_infos = await client.list_tools()
    return [MCPTool(client=client, server_name=server_name, tool_info=info) for info in tool_infos]
