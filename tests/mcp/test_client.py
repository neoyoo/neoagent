from __future__ import annotations
import json
import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch, call
from neoagent.mcp.client import MCPClient, MCPToolInfo
from neoagent.mcp.transport import MCPTransport


# ── Test double: InMemoryTransport ───────────────────────────────────────────

class InMemoryTransport(MCPTransport):
    """In-memory transport for testing: caller programs the response sequence."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self._sent: list[dict] = []
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def send(self, message: dict) -> None:
        self._sent.append(message)

    async def receive(self) -> dict:
        if not self._responses:
            raise ConnectionError("no more responses")
        return self._responses.pop(0)

    async def close(self) -> None:
        self._connected = False


def _make_initialize_response(request_id: int = 1) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "test-server", "version": "1.0"},
        },
    }


def _make_tools_list_response(tools: list[dict], request_id: int = 2) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"tools": tools},
    }


def _make_tool_call_response(content: list[dict], request_id: int = 3) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"content": content},
    }


# ── MCPToolInfo dataclass ────────────────────────────────────────────────────

def test_mcptoolinfo_fields():
    info = MCPToolInfo(
        name="create_issue",
        description="Create a GitHub issue",
        input_schema={"type": "object", "properties": {}},
    )
    assert info.name == "create_issue"
    assert info.description == "Create a GitHub issue"
    assert info.input_schema == {"type": "object", "properties": {}}


# ── MCPClient construction ───────────────────────────────────────────────────

def test_mcpclient_stores_name():
    t = InMemoryTransport([])
    client = MCPClient(name="github", transport=t)
    assert client.name == "github"


def test_mcpclient_stores_transport():
    t = InMemoryTransport([])
    client = MCPClient(name="github", transport=t)
    assert client._transport is t


# ── MCPClient.connect(): initialize handshake ────────────────────────────────

@pytest.mark.asyncio
async def test_mcpclient_connect_sends_initialize():
    """connect() must send an initialize request."""
    t = InMemoryTransport([_make_initialize_response(1)])
    client = MCPClient(name="test", transport=t)
    await client.connect()

    assert len(t._sent) >= 1
    init_msg = t._sent[0]
    assert init_msg["method"] == "initialize"
    assert init_msg["jsonrpc"] == "2.0"


@pytest.mark.asyncio
async def test_mcpclient_connect_sends_initialized_notification():
    """connect() must send initialized notification after initialize response."""
    t = InMemoryTransport([_make_initialize_response(1)])
    client = MCPClient(name="test", transport=t)
    await client.connect()

    # Should have sent: initialize request + initialized notification
    assert len(t._sent) >= 2
    notification = t._sent[1]
    assert notification["method"] == "notifications/initialized"
    # Notifications have no id field
    assert "id" not in notification


@pytest.mark.asyncio
async def test_mcpclient_connect_calls_transport_connect():
    """connect() must call transport.connect() before sending."""
    t = InMemoryTransport([_make_initialize_response(1)])
    client = MCPClient(name="test", transport=t)
    await client.connect()
    assert t._connected is True


# ── MCPClient.list_tools() ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mcpclient_list_tools_returns_mcptoolinfo_list():
    """list_tools() must return list[MCPToolInfo]."""
    tools_payload = [
        {
            "name": "create_issue",
            "description": "Create a GitHub issue",
            "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}},
        },
        {
            "name": "list_repos",
            "description": "List repositories",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]
    t = InMemoryTransport([
        _make_initialize_response(1),
        _make_tools_list_response(tools_payload, 2),
    ])
    client = MCPClient(name="github", transport=t)
    await client.connect()
    tools = await client.list_tools()

    assert len(tools) == 2
    assert all(isinstance(tool, MCPToolInfo) for tool in tools)
    assert tools[0].name == "create_issue"
    assert tools[1].name == "list_repos"


@pytest.mark.asyncio
async def test_mcpclient_list_tools_maps_input_schema():
    """list_tools() must map inputSchema → input_schema on MCPToolInfo."""
    tools_payload = [
        {
            "name": "my_tool",
            "description": "desc",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "integer"}}},
        }
    ]
    t = InMemoryTransport([
        _make_initialize_response(1),
        _make_tools_list_response(tools_payload, 2),
    ])
    client = MCPClient(name="srv", transport=t)
    await client.connect()
    tools = await client.list_tools()

    assert tools[0].input_schema == {"type": "object", "properties": {"x": {"type": "integer"}}}


@pytest.mark.asyncio
async def test_mcpclient_list_tools_empty():
    """list_tools() with no tools must return an empty list."""
    t = InMemoryTransport([
        _make_initialize_response(1),
        _make_tools_list_response([], 2),
    ])
    client = MCPClient(name="srv", transport=t)
    await client.connect()
    tools = await client.list_tools()
    assert tools == []


# ── MCPClient.call_tool() ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mcpclient_call_tool_returns_text():
    """call_tool() must return the concatenated text content from the response."""
    content = [{"type": "text", "text": "Issue #42 created successfully."}]
    t = InMemoryTransport([
        _make_initialize_response(1),
        _make_tool_call_response(content, 2),
    ])
    client = MCPClient(name="github", transport=t)
    await client.connect()
    result = await client.call_tool("create_issue", {"title": "bug"})
    assert result == "Issue #42 created successfully."


@pytest.mark.asyncio
async def test_mcpclient_call_tool_multiple_text_blocks():
    """call_tool() must concatenate multiple text content blocks."""
    content = [
        {"type": "text", "text": "Created."},
        {"type": "text", "text": " URL: https://example.com/42"},
    ]
    t = InMemoryTransport([
        _make_initialize_response(1),
        _make_tool_call_response(content, 2),
    ])
    client = MCPClient(name="github", transport=t)
    await client.connect()
    result = await client.call_tool("create_issue", {"title": "bug"})
    assert "Created." in result
    assert "https://example.com/42" in result


@pytest.mark.asyncio
async def test_mcpclient_call_tool_sends_correct_method():
    """call_tool() must send tools/call with correct params."""
    content = [{"type": "text", "text": "ok"}]
    t = InMemoryTransport([
        _make_initialize_response(1),
        _make_tool_call_response(content, 2),
    ])
    client = MCPClient(name="srv", transport=t)
    await client.connect()
    await client.call_tool("my_tool", {"arg": "val"})

    # The second sent message (after initialize + initialized) should be the tools/call
    call_msg = t._sent[2]  # sent[0]=initialize, sent[1]=initialized notification, sent[2]=tools/call
    assert call_msg["method"] == "tools/call"
    assert call_msg["params"]["name"] == "my_tool"
    assert call_msg["params"]["arguments"] == {"arg": "val"}


# ── MCPClient.close() ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mcpclient_close_calls_transport_close():
    """close() must call transport.close()."""
    t = InMemoryTransport([_make_initialize_response(1)])
    client = MCPClient(name="srv", transport=t)
    await client.connect()
    await client.close()
    assert t._connected is False


# ── MCPClient._request(): id correlation and notification handling ────────────

@pytest.mark.asyncio
async def test_mcpclient_request_skips_notification_no_id():
    """_request() must skip messages with no 'id' field (notifications) and wait for
    the matching response."""
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {"progress": 50},
    }
    # Server sends: notification (no id), then the real response (id=1)
    t = InMemoryTransport([
        _make_initialize_response(1),
        # After connect, list_tools sends request id=2; server sends a notification first
        notification,
        _make_tools_list_response([], request_id=2),
    ])
    client = MCPClient(name="srv", transport=t)
    await client.connect()
    tools = await client.list_tools()
    # The notification was skipped; list_tools still got the correct response
    assert tools == []


@pytest.mark.asyncio
async def test_mcpclient_request_skips_multiple_notifications():
    """_request() must skip multiple interleaved notifications before the response."""
    notif1 = {"jsonrpc": "2.0", "method": "notifications/log", "params": {"message": "a"}}
    notif2 = {"jsonrpc": "2.0", "method": "notifications/log", "params": {"message": "b"}}
    content = [{"type": "text", "text": "result"}]
    t = InMemoryTransport([
        _make_initialize_response(1),
        notif1,
        notif2,
        _make_tool_call_response(content, request_id=2),
    ])
    client = MCPClient(name="srv", transport=t)
    await client.connect()
    result = await client.call_tool("do_thing", {})
    assert result == "result"


@pytest.mark.asyncio
async def test_mcpclient_request_buffers_wrong_id_response():
    """_request() must buffer a response with a non-matching id and return it when
    the correct request comes in later."""
    tools_payload = [{"name": "t1", "description": "d1", "inputSchema": {}}]
    content = [{"type": "text", "text": "call result"}]

    # Simulate out-of-order: call_tool response (id=3) arrives before list_tools response (id=2)
    t = InMemoryTransport([
        _make_initialize_response(1),
        # list_tools sends id=2; server returns id=3 first (out-of-order), then id=2
        _make_tool_call_response(content, request_id=3),
        _make_tools_list_response(tools_payload, request_id=2),
        # call_tool sends id=3; already buffered — no more receives needed
    ])
    client = MCPClient(name="srv", transport=t)
    await client.connect()

    # list_tools sends id=2; gets id=3 first (buffered), then id=2 → returns correct tools
    tools = await client.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "t1"

    # call_tool sends id=3; already buffered from previous read → returns immediately
    result = await client.call_tool("t1", {})
    assert result == "call result"


@pytest.mark.asyncio
async def test_mcpclient_buffered_response_used_on_next_request():
    """A response buffered during one request must be returned on the next matching request."""
    # Scenario: list_tools (id=2) gets a response with id=3 first.
    # Then call_tool (id=3) should pick up the buffered response without another receive().
    content = [{"type": "text", "text": "buffered result"}]
    tools_payload = [{"name": "my_tool", "description": "desc", "inputSchema": {}}]

    transport_responses = [
        _make_initialize_response(1),
        # list_tools request (id=2): server returns id=3 (out of order), then id=2
        _make_tool_call_response(content, request_id=3),
        _make_tools_list_response(tools_payload, request_id=2),
        # call_tool request (id=3): already buffered, no receive needed
    ]
    t = InMemoryTransport(transport_responses)
    client = MCPClient(name="srv", transport=t)
    await client.connect()

    # list_tools: buffers id=3, returns id=2 result
    tools = await client.list_tools()
    assert tools[0].name == "my_tool"

    # call_tool: id=3 was buffered; _pending has it, no transport.receive() call needed
    assert 3 in client._pending

    result = await client.call_tool("my_tool", {})
    assert result == "buffered result"
    # Buffer cleared after use
    assert 3 not in client._pending
