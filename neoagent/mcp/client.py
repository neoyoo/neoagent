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
        # Buffer for responses received out-of-order: id → message dict
        self._pending: dict[int, dict] = {}

    def _next_request_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    async def _request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and return the matching response.

        Handles interleaved notifications (no "id" field) by skipping them
        with a DEBUG log. Handles out-of-order responses by buffering them
        in ``self._pending`` until the matching id arrives.

        Args:
            method: JSON-RPC method name.
            params: Parameters dict.

        Returns:
            The response dict with matching id.
        """
        req_id = self._next_request_id()

        # Check if the response was already buffered from a previous read
        if req_id in self._pending:
            return self._pending.pop(req_id)

        await self._transport.send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        })

        # Loop until we get the response for our request id
        while True:
            msg = await self._transport.receive()

            # Notifications have no "id" field — log and skip
            if "id" not in msg:
                logger.debug(
                    "MCPClient '%s': received notification method=%s, skipping",
                    self.name, msg.get("method", "<unknown>"),
                )
                continue

            msg_id = msg["id"]
            if msg_id == req_id:
                return msg

            # Response for a different (future) request — buffer it
            logger.debug(
                "MCPClient '%s': buffering out-of-order response id=%s (waiting for id=%s)",
                self.name, msg_id, req_id,
            )
            self._pending[msg_id] = msg

    async def connect(self) -> None:
        """Establish transport and perform the MCP initialize handshake.

        Sends:
          1. initialize request
          2. notifications/initialized notification (no id, no response expected)
        """
        await self._transport.connect()

        # Step 1: initialize request (uses _request for id correlation)
        response = await self._request(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "neoagent", "version": "1.0"},
            },
        )
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
        response = await self._request("tools/list", {})
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
        response = await self._request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
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
