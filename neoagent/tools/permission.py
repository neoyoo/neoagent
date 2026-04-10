from __future__ import annotations
from typing import Awaitable, Callable
from pydantic import BaseModel
from neoagent.tools.base import BaseTool

AskCallback = Callable[[str, str, dict], Awaitable[bool]]

class PermissionChecker:
    def __init__(self, ask_callback: AskCallback | None = None):
        self._ask_callback = ask_callback

    async def check(self, tool: BaseTool, input: BaseModel) -> bool:
        if tool.permission == "auto":
            return True
        if tool.permission == "deny":
            return False
        # permission == "ask"
        if self._ask_callback is None:
            return True
        return await self._ask_callback(
            tool.name, tool.description, input.model_dump()
        )
