from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING
from pydantic import BaseModel, Field
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult
from neoagent.tools.deferred import DeferredToolRegistry
from neoagent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from neoagent.session import SessionState

logger = logging.getLogger(__name__)


class ToolSearchInput(BaseModel):
    """Input for the tool_search built-in tool."""
    query: str = Field(
        description=(
            "Search query for deferred MCP tools. Supported formats:\n"
            "  - 'select:name1,name2' — exact name match\n"
            "  - '+keyword rest'      — name must contain keyword\n"
            "  - 'keyword'            — regex search on name and description"
        )
    )


class ToolSearchTool(BaseTool):
    """Built-in tool: search deferred MCP tools and promote them for the next turn.

    When the model calls this tool, it:
    1. Searches DeferredToolRegistry for matching tool names.
    2. Promotes the matched tools in the current session (session-scoped visibility).
    3. Returns only name + description (not full schema) to avoid leaking internal
       system details before the tool is used.  The full schema becomes visible in
       the LLM tool list on the next turn after promote.

    Permission: "auto" — searching itself has no dangerous side effects.
    """

    name: str = "tool_search"
    description: str = (
        "Search available deferred MCP tools by keyword or exact name. "
        "Matched tools will become available for calling in the next turn. "
        "Use 'select:tool_name' for exact match. "
        "Use a keyword to search by name or description."
    )
    input_model: type[BaseModel] = ToolSearchInput
    permission: str = "auto"
    is_concurrent_safe: bool = True

    def __init__(
        self,
        deferred_registry: DeferredToolRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        self._deferred = deferred_registry
        self._registry = tool_registry
        # Set by QueryLoop before each turn; tracks per-session promoted tools.
        self._session_state: "SessionState | None" = None

    async def execute(self, input: BaseModel) -> ToolResult:
        """Search + promote in session + return name/description as JSON array."""
        assert isinstance(input, ToolSearchInput)
        query = input.query.strip()

        # 1. Search deferred index (searches all registered-but-unregistered-globally tools)
        matches = self._deferred.search(query)
        logger.debug("tool_search: query=%r matched %d tools", query, len(matches))

        # 2. Verify tools exist in ToolRegistry, skip ghost entries
        promoted_names: set[str] = set()
        results: list[dict] = []
        for idx in matches:
            tool = self._registry.get_tool(idx.name)
            if tool is None:
                logger.warning(
                    "tool_search: '%s' found in deferred index but not in ToolRegistry; skipping",
                    idx.name,
                )
                continue
            # Return only name + description — full schema exposed after promote on next turn
            results.append({"name": idx.name, "description": idx.description})
            promoted_names.add(idx.name)

        # 3. Promote matched tools in session state (session-scoped) and global registry
        if promoted_names:
            if self._session_state is not None:
                self._session_state.promoted_tools.update(promoted_names)
            # Also promote in global registry for backward compat (loop uses session state
            # as the source of truth when session state is available)
            self._deferred.promote(promoted_names)

        # 4. Return compact JSON (no indent — reduces token usage)
        return ToolResult(call_id="", output=json.dumps(results))
