from __future__ import annotations
import logging
from dataclasses import dataclass
from neoagent.mcp.transport import MCPTransport

logger = logging.getLogger(__name__)

_MCP_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class MCPToolInfo:
    """Metadata about a single tool exposed by an MCP server."""
    name: str
    description: str
    input_schema: dict  # JSON Schema for the tool's input


class MCPClient:
    """Manages the connection lifecycle and JSON-RPC protocol with one MCP server.

    Usage::

        transport = StdioTransport(command=["npx", "@anthropic/mcp-server-github"])
        client = MCPClient(name="github", transport=transport)
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("create_issue", {"title": "Bug"})
        await client.close()
    """

    def __init__(self, name: str, transport: MCPTransport) -> None:
        self.name = name
        self._transport = transport
        self._next_id = 1

    def _next_request_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    async def connect(self) -> None:
        """Establish transport and perform the MCP initialize handshake.

        Sends:
          1. initialize request
          2. notifications/initialized notification (no id, no response expected)
        """
        await self._transport.connect()

        # Step 1: initialize request
        init_id = self._next_request_id()
        await self._transport.send({
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "neoagent", "version": "1.0"},
            },
        })

        # Wait for initialize response
        response = await self._transport.receive()
        if "error" in response:
            raise RuntimeError(
                f"MCPClient '{self.name}': initialize failed: {response['error']}"
            )
        logger.debug("MCPClient '%s': initialize OK", self.name)

        # Step 2: send initialized notification (no id = notification, no response)
        await self._transport.send({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        })

    async def list_tools(self) -> list[MCPToolInfo]:
        """Fetch the list of tools from the MCP server.

        Returns:
            list[MCPToolInfo] — one entry per tool exposed by the server.
        """
        req_id = self._next_request_id()
        await self._transport.send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/list",
            "params": {},
        })
        response = await self._transport.receive()
        if "error" in response:
            raise RuntimeError(
                f"MCPClient '{self.name}': tools/list failed: {response['error']}"
            )
        raw_tools: list[dict] = response.get("result", {}).get("tools", [])
        return [
            MCPToolInfo(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in raw_tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool and return its result as a string.

        Concatenates all text-type content blocks in the response.

        Args:
            tool_name: Name of the tool as returned by list_tools().
            arguments: Dict matching the tool's input schema.

        Returns:
            String result (concatenated text content blocks).
        """
        req_id = self._next_request_id()
        await self._transport.send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        })
        response = await self._transport.receive()
        if "error" in response:
            raise RuntimeError(
                f"MCPClient '{self.name}': tools/call '{tool_name}' failed: {response['error']}"
            )
        content_blocks: list[dict] = response.get("result", {}).get("content", [])
        texts = [b["text"] for b in content_blocks if b.get("type") == "text"]
        return "".join(texts)

    async def close(self) -> None:
        """Close the transport connection."""
        await self._transport.close()
        logger.debug("MCPClient '%s': closed", self.name)
