from __future__ import annotations
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolIndex:
    """Lightweight index entry for a deferred tool."""
    name: str
    description: str


class DeferredToolRegistry:
    """Manages the deferred tool loading index (DeerFlow pattern).

    Tools are registered here when first loaded from an MCP server.
    They remain "deferred" until explicitly promoted via promote().
    Deferred tools are hidden from the LLM's tool list but searchable
    via the tool_search built-in tool.

    Lifecycle:
        register() → is_deferred() == True
        promote() → is_deferred() == False (LLM sees schema next turn)
        reset() → clears all state (for new sessions)
    """

    def __init__(self) -> None:
        # All registered tools: name → ToolIndex
        self._all: dict[str, ToolIndex] = {}
        # Names of tools still in deferred state (not yet promoted)
        self._deferred: set[str] = set()

    def register(self, tool_name: str, description: str) -> None:
        """Add a tool to the deferred index. Idempotent."""
        self._all[tool_name] = ToolIndex(name=tool_name, description=description)
        self._deferred.add(tool_name)

    def promote(self, tool_names: set[str]) -> None:
        """Remove tools from deferred state so their schema becomes LLM-visible next turn.

        Silently ignores names that were never registered.
        """
        self._deferred -= tool_names
        logger.debug("DeferredToolRegistry: promoted %s", tool_names)

    def is_deferred(self, tool_name: str) -> bool:
        """Return True if tool is registered and not yet promoted."""
        return tool_name in self._deferred

    def get_deferred_names(self) -> list[str]:
        """Return sorted list of all currently deferred tool names."""
        return sorted(self._deferred)

    def search(self, query: str) -> list[ToolIndex]:
        """Search the deferred tool index.

        Three query modes:
        - ``"select:name1,name2"`` — exact name match, returns those tools if deferred
        - ``"+keyword rest"`` — name must contain keyword (case-insensitive)
        - ``"keyword"`` — regex search across name + description (case-insensitive)

        Returns only currently-deferred tools.
        """
        if query.startswith("select:"):
            return self._search_select(query[len("select:"):])
        if query.startswith("+"):
            return self._search_plus_keyword(query[1:])
        return self._search_regex(query)

    def reset(self) -> None:
        """Clear all state. Call when starting a new session."""
        self._all.clear()
        self._deferred.clear()

    # ── private search implementations ───────────────────────────────────────

    def _search_select(self, names_str: str) -> list[ToolIndex]:
        """Exact match by comma-separated names. Only returns deferred entries."""
        names = [n.strip() for n in names_str.split(",") if n.strip()]
        return [self._all[n] for n in names if n in self._all and n in self._deferred]

    def _search_plus_keyword(self, rest: str) -> list[ToolIndex]:
        """name must contain the first word (the keyword after '+')."""
        parts = rest.split(None, 1)
        if not parts:
            return []
        keyword = parts[0].lower()
        return [
            idx for idx in self._all.values()
            if idx.name in self._deferred and keyword in idx.name.lower()
        ]

    def _search_regex(self, pattern: str) -> list[ToolIndex]:
        """Case-insensitive regex search across name and description."""
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Treat invalid regex as literal substring
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        return [
            idx for idx in self._all.values()
            if idx.name in self._deferred and (
                rx.search(idx.name) or rx.search(idx.description)
            )
        ]
