from __future__ import annotations
import json
import logging
from pydantic import BaseModel, Field
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult
from neoagent.tools.deferred import DeferredToolRegistry
from neoagent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolSearchInput(BaseModel):
    """Input for the tool_search built-in tool."""
    query: str = Field(
        description=(
            "Search query for deferred tools. Supported formats:\n"
            "  - 'select:name1,name2' — exact name match\n"
            "  - '+keyword rest'      — name must contain keyword\n"
            "  - 'keyword'            — regex search on name and description"
        )
    )


class ToolSearchTool(BaseTool):
    """Built-in tool: search deferred tools and return their full schemas.

    When the model calls this tool, it:
    1. Searches DeferredToolRegistry for matching tool names.
    2. Retrieves full schemas from ToolRegistry.
    3. Promotes the matched tools (they become visible in the next LLM turn).
    4. Returns the schemas as JSON.

    Permission: "auto" — searching has no side effects beyond promoting schemas.
    """

    name: str = "tool_search"
    description: str = (
        "Search available tools by keyword or exact name. "
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

    async def execute(self, input: BaseModel) -> ToolResult:
        """Search + promote + return schemas as JSON array."""
        assert isinstance(input, ToolSearchInput)
        query = input.query.strip()

        # 1. Search deferred index
        matches = self._deferred.search(query)
        logger.debug("tool_search: query=%r matched %d tools", query, len(matches))

        # 2. Fetch full schemas from ToolRegistry, skip ghosts
        schemas: list[dict] = []
        promoted_names: set[str] = set()
        for idx in matches:
            tool = self._registry.get_tool(idx.name)
            if tool is None:
                logger.warning(
                    "tool_search: '%s' found in deferred index but not in ToolRegistry; skipping",
                    idx.name,
                )
                continue
            schemas.append(tool.get_schema())
            promoted_names.add(idx.name)

        # 3. Promote matched tools
        if promoted_names:
            self._deferred.promote(promoted_names)

        # 4. Return JSON
        return ToolResult(call_id="", output=json.dumps(schemas, indent=2))
