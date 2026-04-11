from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any
from pydantic import BaseModel, create_model
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult

if TYPE_CHECKING:
    from neoagent.mcp.client import MCPClient, MCPToolInfo

logger = logging.getLogger(__name__)

# Map JSON Schema type strings to Python/Pydantic types
_JSON_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_pydantic(tool_name: str, input_schema: dict) -> type[BaseModel]:
    """Build a Pydantic model class from a JSON Schema dict.

    Supports 'properties' with scalar types and 'required' list.
    Unknown or complex types fall back to Any.
    """
    properties: dict = input_schema.get("properties", {})
    required: set[str] = set(input_schema.get("required", []))

    field_definitions: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")
        py_type = _JSON_TYPE_MAP.get(json_type, Any)
        if field_name in required:
            field_definitions[field_name] = (py_type, ...)
        else:
            field_definitions[field_name] = (py_type | None, None)

    # create_model("name", field=(type, default), ...) — positional arg is model name
    model_name = f"MCPInput_{tool_name.replace('__', '_').replace('-', '_')}"
    return create_model(model_name, **field_definitions)  # type: ignore[call-overload]


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
