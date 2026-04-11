from __future__ import annotations
import logging
from typing import Awaitable, Callable
from pydantic import BaseModel
from neoagent.tools.base import BaseTool

AskCallback = Callable[[str, str, dict], Awaitable[bool]]

logger = logging.getLogger(__name__)


class PermissionChecker:
    def __init__(
        self,
        ask_callback: AskCallback | None = None,
        auto_approve: bool = True,
    ):
        self._ask_callback = ask_callback
        self._auto_approve = auto_approve

    async def check(self, tool: BaseTool, input: BaseModel) -> bool:
        if tool.permission == "auto":
            return True
        if tool.permission == "deny":
            return False
        # permission == "ask"
        if self._ask_callback is None:
            if self._auto_approve:
                logger.warning(
                    "Auto-approving tool '%s' (non-interactive mode, no ask_callback set)",
                    tool.name,
                )
                return True
            return False
        return await self._ask_callback(
            tool.name, tool.description, input.model_dump()
        )
