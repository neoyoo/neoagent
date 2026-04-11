from __future__ import annotations
from neoagent.tools.base import BaseTool


class ToolRegistry:
    """Pure data structure: register tools, serve schemas.

    Execution logic (permission, concurrency, truncation) lives in ToolExecutor.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def get_schemas(self) -> list[dict]:
        """deny 的工具不暴露给模型。"""
        return [t.get_schema() for t in self._tools.values() if t.permission != "deny"]

    def unregister(self, name: str) -> None:
        """Remove a tool by name. No-op if the tool is not registered."""
        self._tools.pop(name, None)

    def all_tools(self) -> dict[str, BaseTool]:
        return dict(self._tools)
