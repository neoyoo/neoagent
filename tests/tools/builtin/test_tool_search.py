from __future__ import annotations
import json
import pytest
from unittest.mock import MagicMock, AsyncMock
from pydantic import BaseModel
from neoagent.tools.builtin.tool_search import ToolSearchTool, ToolSearchInput
from neoagent.tools.deferred import DeferredToolRegistry, ToolIndex
from neoagent.tools.registry import ToolRegistry
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult


# ── Helpers ───────────────────────────────────────────────────────────────────

class _SimpleTool(BaseTool):
    name: str
    description: str
    input_model: type[BaseModel]
    permission: str = "auto"
    is_concurrent_safe: bool = True

    def __init__(self, name: str, description: str = "a tool"):
        from pydantic import create_model
        self.name = name
        self.description = description
        self.input_model = create_model(f"Input_{name}", query=(str, ""))

    async def execute(self, input: BaseModel) -> ToolResult:
        return ToolResult(call_id="", output="ok")


def _make_setup(tool_names: list[str]) -> tuple[DeferredToolRegistry, ToolRegistry, ToolSearchTool]:
    deferred = DeferredToolRegistry()
    registry = ToolRegistry()

    for name in tool_names:
        tool = _SimpleTool(name=name, description=f"Description of {name}")
        registry.register(tool)
        deferred.register(name, f"Description of {name}")

    search_tool = ToolSearchTool(deferred_registry=deferred, tool_registry=registry)
    return deferred, registry, search_tool


# ── ToolSearchInput ────────────────────────────────────────────────────────────

def test_toolsearchinput_has_query_field():
    inp = ToolSearchInput(query="test")
    assert inp.query == "test"


# ── ToolSearchTool class attributes ──────────────────────────────────────────

def test_toolsearch_name():
    _, _, t = _make_setup([])
    assert t.name == "tool_search"


def test_toolsearch_permission_is_auto():
    _, _, t = _make_setup([])
    assert t.permission == "auto"


def test_toolsearch_is_concurrent_safe():
    _, _, t = _make_setup([])
    assert t.is_concurrent_safe is True


def test_toolsearch_is_basetool():
    _, _, t = _make_setup([])
    assert isinstance(t, BaseTool)


# ── execute(): returns matching schemas ──────────────────────────────────────

@pytest.mark.asyncio
async def test_toolsearch_returns_json():
    """execute() must return valid JSON in ToolResult.output."""
    _, _, t = _make_setup(["github__create_issue"])
    inp = ToolSearchInput(query="github__create_issue")
    result = await t.execute(inp)
    assert not result.is_error
    data = json.loads(result.output)
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_toolsearch_returns_schema_for_matching_tool():
    """execute() must include the full schema of matched tools."""
    _, reg, t = _make_setup(["github__create_issue", "github__list_repos"])
    inp = ToolSearchInput(query="select:github__create_issue")
    result = await t.execute(inp)
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["name"] == "github__create_issue"
    assert "input_schema" in data[0]


@pytest.mark.asyncio
async def test_toolsearch_keyword_matches_multiple():
    """Keyword search returns multiple matching tools."""
    _, _, t = _make_setup(["github__create_issue", "github__list_repos", "filesystem__read_file"])
    inp = ToolSearchInput(query="github")
    result = await t.execute(inp)
    data = json.loads(result.output)
    names = {item["name"] for item in data}
    assert "github__create_issue" in names
    assert "github__list_repos" in names
    assert "filesystem__read_file" not in names


@pytest.mark.asyncio
async def test_toolsearch_no_match_returns_empty_list():
    """No matching tools → return empty JSON array."""
    _, _, t = _make_setup(["github__create_issue"])
    inp = ToolSearchInput(query="xyz_no_match_at_all")
    result = await t.execute(inp)
    data = json.loads(result.output)
    assert data == []
    assert not result.is_error


# ── execute(): promotes matched tools ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_toolsearch_promotes_matched_tools():
    """After execute(), matched tools must be promoted (no longer deferred)."""
    deferred, _, t = _make_setup(["github__create_issue", "github__list_repos"])
    inp = ToolSearchInput(query="select:github__create_issue")
    await t.execute(inp)
    # create_issue promoted, list_repos still deferred
    assert not deferred.is_deferred("github__create_issue")
    assert deferred.is_deferred("github__list_repos")


@pytest.mark.asyncio
async def test_toolsearch_no_match_no_promotion():
    """No match → no tools promoted."""
    deferred, _, t = _make_setup(["github__create_issue"])
    inp = ToolSearchInput(query="xyz_no_match")
    await t.execute(inp)
    assert deferred.is_deferred("github__create_issue")


@pytest.mark.asyncio
async def test_toolsearch_promotes_all_keyword_matches():
    """Keyword search promotes all matched tools."""
    deferred, _, t = _make_setup(["github__create_issue", "github__list_repos", "filesystem__read_file"])
    inp = ToolSearchInput(query="github")
    await t.execute(inp)
    assert not deferred.is_deferred("github__create_issue")
    assert not deferred.is_deferred("github__list_repos")
    assert deferred.is_deferred("filesystem__read_file")


# ── execute(): schema from ToolRegistry ──────────────────────────────────────

@pytest.mark.asyncio
async def test_toolsearch_schema_comes_from_tool_registry():
    """Schema in output must match what ToolRegistry.get_tool().get_schema() returns."""
    deferred, registry, t = _make_setup(["github__create_issue"])
    inp = ToolSearchInput(query="select:github__create_issue")
    result = await t.execute(inp)
    data = json.loads(result.output)
    expected_schema = registry.get_tool("github__create_issue").get_schema()
    assert data[0] == expected_schema


@pytest.mark.asyncio
async def test_toolsearch_tool_not_in_registry_skipped():
    """If DeferredToolRegistry has a name not in ToolRegistry, it is silently skipped."""
    deferred = DeferredToolRegistry()
    deferred.register("ghost__tool", "A ghost tool not in registry")
    registry = ToolRegistry()  # empty
    t = ToolSearchTool(deferred_registry=deferred, tool_registry=registry)
    inp = ToolSearchInput(query="ghost")
    result = await t.execute(inp)
    data = json.loads(result.output)
    assert data == []
