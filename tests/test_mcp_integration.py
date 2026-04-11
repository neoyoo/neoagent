from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel
from neoagent.config import NeoAgentConfig
from neoagent.agent import NeoAgent
from neoagent.core.loop import QueryLoop
from neoagent.core.prompt import PromptBuilder
from neoagent.tools.registry import ToolRegistry
from neoagent.tools.deferred import DeferredToolRegistry
from neoagent.tools.builtin.tool_search import ToolSearchTool
from neoagent.tools.base import BaseTool
from neoagent.core.types import Message, TextBlock, ToolUseBlock, ToolResult
from neoagent.providers.base import Response


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_provider():
    mock = MagicMock()
    mock.get_context_window.return_value = 200_000
    return mock


def _make_response(text="Hello", stop_reason="end_turn"):
    return Response(
        content=[TextBlock(text=text)],
        stop_reason=stop_reason,
        input_tokens=10,
        output_tokens=5,
    )


def _make_tool_response(tool_name: str, tool_input: dict, call_id: str = "c1") -> Response:
    return Response(
        content=[ToolUseBlock(id=call_id, name=tool_name, input=tool_input)],
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=5,
    )


# ── NeoAgent: add_mcp_server / list_mcp_servers / remove_mcp_server ──────────

@patch("neoagent.agent._create_provider")
def test_agent_has_deferred_registry(mock_create):
    mock_create.return_value = _make_mock_provider()
    agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
    assert hasattr(agent, "_deferred_registry")
    from neoagent.tools.deferred import DeferredToolRegistry
    assert isinstance(agent._deferred_registry, DeferredToolRegistry)


@patch("neoagent.agent._create_provider")
def test_agent_list_mcp_servers_initially_empty(mock_create):
    mock_create.return_value = _make_mock_provider()
    agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
    assert agent.list_mcp_servers() == []


@pytest.mark.asyncio
@patch("neoagent.agent._create_provider")
async def test_agent_add_mcp_server_registers_tools(mock_create):
    """add_mcp_server() must register MCPTools into ToolRegistry and DeferredToolRegistry."""
    from neoagent.mcp.client import MCPToolInfo
    mock_create.return_value = _make_mock_provider()

    mock_client = MagicMock()
    mock_client.name = "github"
    mock_client.connect = AsyncMock()
    mock_client.list_tools = AsyncMock(return_value=[
        MCPToolInfo("create_issue", "Create a GitHub issue", {"type": "object", "properties": {}}),
        MCPToolInfo("list_repos", "List repos", {"type": "object", "properties": {}}),
    ])
    mock_client.close = AsyncMock()

    with patch("neoagent.agent.MCPClient", return_value=mock_client):
        with patch("neoagent.agent.StdioTransport"):
            agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
            await agent.add_mcp_server("github", command=["npx", "server-github"])

    assert agent.list_mcp_servers() == ["github"]
    # Tools registered in ToolRegistry
    assert agent._registry.get_tool("github__create_issue") is not None
    assert agent._registry.get_tool("github__list_repos") is not None
    # Tools in DeferredToolRegistry
    assert agent._deferred_registry.is_deferred("github__create_issue")
    assert agent._deferred_registry.is_deferred("github__list_repos")


@pytest.mark.asyncio
@patch("neoagent.agent._create_provider")
async def test_agent_add_mcp_server_registers_tool_search_on_first_server(mock_create):
    """First add_mcp_server() must auto-register tool_search in ToolRegistry."""
    from neoagent.mcp.client import MCPToolInfo
    mock_create.return_value = _make_mock_provider()

    mock_client = MagicMock()
    mock_client.name = "github"
    mock_client.connect = AsyncMock()
    mock_client.list_tools = AsyncMock(return_value=[
        MCPToolInfo("create_issue", "desc", {"type": "object", "properties": {}}),
    ])
    mock_client.close = AsyncMock()

    with patch("neoagent.agent.MCPClient", return_value=mock_client):
        with patch("neoagent.agent.StdioTransport"):
            agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
            await agent.add_mcp_server("github", command=["npx", "server-github"])

    tool_search = agent._registry.get_tool("tool_search")
    assert tool_search is not None
    assert isinstance(tool_search, ToolSearchTool)


@pytest.mark.asyncio
@patch("neoagent.agent._create_provider")
async def test_agent_add_second_mcp_server_no_duplicate_tool_search(mock_create):
    """Adding a second MCP server must not re-register tool_search (would raise ValueError)."""
    from neoagent.mcp.client import MCPToolInfo
    mock_create.return_value = _make_mock_provider()

    def _make_client(name):
        c = MagicMock()
        c.name = name
        c.connect = AsyncMock()
        c.list_tools = AsyncMock(return_value=[
            MCPToolInfo(f"{name}_tool", "desc", {"type": "object", "properties": {}}),
        ])
        c.close = AsyncMock()
        return c

    with patch("neoagent.agent.MCPClient", side_effect=[_make_client("s1"), _make_client("s2")]):
        with patch("neoagent.agent.StdioTransport"):
            agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
            await agent.add_mcp_server("s1", command=["cmd1"])
            await agent.add_mcp_server("s2", command=["cmd2"])  # must not raise

    assert set(agent.list_mcp_servers()) == {"s1", "s2"}


@pytest.mark.asyncio
@patch("neoagent.agent._create_provider")
async def test_agent_remove_mcp_server(mock_create):
    """remove_mcp_server() must close client and unregister its tools."""
    from neoagent.mcp.client import MCPToolInfo
    mock_create.return_value = _make_mock_provider()

    mock_client = MagicMock()
    mock_client.name = "github"
    mock_client.connect = AsyncMock()
    mock_client.list_tools = AsyncMock(return_value=[
        MCPToolInfo("create_issue", "desc", {"type": "object", "properties": {}}),
    ])
    mock_client.close = AsyncMock()

    with patch("neoagent.agent.MCPClient", return_value=mock_client):
        with patch("neoagent.agent.StdioTransport"):
            agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
            await agent.add_mcp_server("github", command=["npx", "server-github"])
            await agent.remove_mcp_server("github")

    mock_client.close.assert_called_once()
    assert agent.list_mcp_servers() == []
    # Tool must be unregistered from ToolRegistry
    assert agent._registry.get_tool("github__create_issue") is None


@pytest.mark.asyncio
@patch("neoagent.agent._create_provider")
async def test_agent_remove_mcp_server_cleans_deferred_registry(mock_create):
    """remove_mcp_server() must also remove tools from DeferredToolRegistry.

    Without this, tool_search can still return ghost entries for removed servers.
    """
    from neoagent.mcp.client import MCPToolInfo
    mock_create.return_value = _make_mock_provider()

    mock_client = MagicMock()
    mock_client.name = "github"
    mock_client.connect = AsyncMock()
    mock_client.list_tools = AsyncMock(return_value=[
        MCPToolInfo("create_issue", "Create a GitHub issue", {"type": "object", "properties": {}}),
        MCPToolInfo("list_repos", "List repositories", {"type": "object", "properties": {}}),
    ])
    mock_client.close = AsyncMock()

    with patch("neoagent.agent.MCPClient", return_value=mock_client):
        with patch("neoagent.agent.StdioTransport"):
            agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
            await agent.add_mcp_server("github", command=["npx", "server-github"])

            # Tools must be in deferred registry before removal
            assert agent._deferred_registry.is_deferred("github__create_issue")
            assert agent._deferred_registry.is_deferred("github__list_repos")

            await agent.remove_mcp_server("github")

    # After removal, deferred registry must have no ghost entries
    assert not agent._deferred_registry.is_deferred("github__create_issue")
    assert not agent._deferred_registry.is_deferred("github__list_repos")
    # search must not return removed tools
    results = agent._deferred_registry.search("github")
    assert results == []


@pytest.mark.asyncio
@patch("neoagent.agent._create_provider")
async def test_agent_close_closes_all_mcp_clients(mock_create):
    """agent.close() must close all MCP clients."""
    from neoagent.mcp.client import MCPToolInfo
    mock_create.return_value = _make_mock_provider()

    clients = []
    def _make_client(name):
        c = MagicMock()
        c.name = name
        c.connect = AsyncMock()
        c.list_tools = AsyncMock(return_value=[
            MCPToolInfo(f"{name}_tool", "desc", {"type": "object", "properties": {}}),
        ])
        c.close = AsyncMock()
        clients.append(c)
        return c

    with patch("neoagent.agent.MCPClient", side_effect=[_make_client("s1"), _make_client("s2")]):
        with patch("neoagent.agent.StdioTransport"):
            agent = NeoAgent(NeoAgentConfig(api_key="sk-test"))
            await agent.add_mcp_server("s1", command=["cmd1"])
            await agent.add_mcp_server("s2", command=["cmd2"])
            await agent.close()

    for c in clients:
        c.close.assert_called_once()


# ── QueryLoop: schema filtering and deferred injection ─────────────────────────

@pytest.mark.asyncio
async def test_queryloop_filters_deferred_schemas():
    """Deferred tools must not appear in the tools list sent to the LLM."""
    from pydantic import create_model

    class _FakeTool(BaseTool):
        name: str = "deferred__tool"
        description: str = "a deferred tool"
        input_model: type[BaseModel] = create_model("I", x=(str, ""))
        permission: str = "auto"
        is_concurrent_safe: bool = True
        async def execute(self, input: BaseModel) -> ToolResult:
            return ToolResult(call_id="", output="ok")

    registry = ToolRegistry()
    registry.register(_FakeTool())

    deferred = DeferredToolRegistry()
    deferred.register("deferred__tool", "a deferred tool")

    captured_tools: list[list[dict]] = []
    mock_provider = MagicMock()
    mock_provider.get_context_window.return_value = 200_000

    async def fake_create(system, messages, tools, max_tokens):
        captured_tools.append(list(tools))
        return _make_response()

    mock_provider.create = fake_create
    prompt_builder = PromptBuilder()

    loop = QueryLoop(
        provider=mock_provider,
        tool_registry=registry,
        prompt_builder=prompt_builder,
        deferred_registry=deferred,
    )

    msgs = [Message(role="user", content="hello")]
    await loop.run(messages=msgs)

    assert len(captured_tools) == 1
    tool_names = [t["name"] for t in captured_tools[0]]
    assert "deferred__tool" not in tool_names


@pytest.mark.asyncio
async def test_queryloop_injects_deferred_names_into_system_prompt():
    """Deferred tool names must appear in the system prompt sent to the LLM."""
    from pydantic import create_model

    class _FakeTool(BaseTool):
        name: str = "github__create_issue"
        description: str = "desc"
        input_model: type[BaseModel] = create_model("I", x=(str, ""))
        permission: str = "auto"
        is_concurrent_safe: bool = True
        async def execute(self, input: BaseModel) -> ToolResult:
            return ToolResult(call_id="", output="ok")

    registry = ToolRegistry()
    registry.register(_FakeTool())
    deferred = DeferredToolRegistry()
    deferred.register("github__create_issue", "desc")

    captured_systems: list[str] = []
    mock_provider = MagicMock()
    mock_provider.get_context_window.return_value = 200_000

    async def fake_create(system, messages, tools, max_tokens):
        captured_systems.append(system)
        return _make_response()

    mock_provider.create = fake_create
    prompt_builder = PromptBuilder()

    loop = QueryLoop(
        provider=mock_provider,
        tool_registry=registry,
        prompt_builder=prompt_builder,
        deferred_registry=deferred,
    )

    await loop.run(messages=[Message(role="user", content="hi")])

    assert len(captured_systems) == 1
    assert "<deferred-tools>" in captured_systems[0]
    assert "github__create_issue" in captured_systems[0]
    assert "</deferred-tools>" in captured_systems[0]


@pytest.mark.asyncio
async def test_queryloop_no_deferred_registry_no_injection():
    """Without deferred_registry, system prompt must NOT have <deferred-tools> section."""
    captured_systems: list[str] = []
    mock_provider = MagicMock()
    mock_provider.get_context_window.return_value = 200_000

    async def fake_create(system, messages, tools, max_tokens):
        captured_systems.append(system)
        return _make_response()

    mock_provider.create = fake_create
    prompt_builder = PromptBuilder()
    loop = QueryLoop(
        provider=mock_provider,
        tool_registry=ToolRegistry(),
        prompt_builder=prompt_builder,
    )

    await loop.run(messages=[Message(role="user", content="hi")])
    assert "<deferred-tools>" not in captured_systems[0]


@pytest.mark.asyncio
async def test_queryloop_promoted_tool_visible_next_turn():
    """After tool_search promotes a tool, it must appear in schemas on the next LLM call."""
    from pydantic import create_model

    class _SearchableTool(BaseTool):
        name: str = "github__create_issue"
        description: str = "Create issue"
        input_model: type[BaseModel] = create_model("GI", title=(str, ...))
        permission: str = "auto"
        is_concurrent_safe: bool = True
        async def execute(self, input: BaseModel) -> ToolResult:
            return ToolResult(call_id="", output="done")

    class _SearchTool(BaseTool):
        name: str = "tool_search"
        description: str = "search tools"
        input_model: type[BaseModel] = create_model("TSI", query=(str, ""))
        permission: str = "auto"
        is_concurrent_safe: bool = True
        def __init__(self, deferred, registry):
            self._d = deferred
            self._r = registry
        async def execute(self, input: BaseModel) -> ToolResult:
            self._d.promote({"github__create_issue"})
            return ToolResult(call_id="", output=json.dumps([]))

    registry = ToolRegistry()
    deferred = DeferredToolRegistry()
    searchable = _SearchableTool()
    registry.register(searchable)
    deferred.register("github__create_issue", "Create issue")
    search = _SearchTool(deferred, registry)
    registry.register(search)

    call_count = 0
    captured_tools_per_turn: list[list[str]] = []

    mock_provider = MagicMock()
    mock_provider.get_context_window.return_value = 200_000

    async def fake_create(system, messages, tools, max_tokens):
        nonlocal call_count
        call_count += 1
        captured_tools_per_turn.append([t["name"] for t in tools])
        if call_count == 1:
            # First turn: request tool_search
            return _make_tool_response("tool_search", {"query": "select:github__create_issue"}, "c1")
        # Second turn: end
        return _make_response("done")

    mock_provider.create = AsyncMock(side_effect=fake_create)
    prompt_builder = PromptBuilder()

    loop = QueryLoop(
        provider=mock_provider,
        tool_registry=registry,
        prompt_builder=prompt_builder,
        deferred_registry=deferred,
        tool_executor=None,  # will be set below
    )
    from neoagent.tools.executor import ToolExecutor
    from neoagent.tools.permission import PermissionChecker
    executor = ToolExecutor(registry=registry, permission_checker=PermissionChecker(auto_approve=True))
    loop._executor = executor

    await loop.run(messages=[Message(role="user", content="help")])

    # Turn 1: tool_search visible, github__create_issue not
    assert "tool_search" in captured_tools_per_turn[0]
    assert "github__create_issue" not in captured_tools_per_turn[0]
    # Turn 2: github__create_issue now visible (promoted)
    assert "github__create_issue" in captured_tools_per_turn[1]
