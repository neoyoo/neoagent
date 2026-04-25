from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from neoagent.core.types import ToolResult, ToolResultBlock
from neoagent.tools.base import BaseTool
from neoagent.tools.builtin.tool_search import _current_session

if TYPE_CHECKING:
    pass


class FreeToolResultInput(BaseModel):
    tool_use_ids: list[str] = Field(description="IDs of past tool calls whose results should be collapsed.")


class FreeToolResultTool(BaseTool):
    name = "free_tool_result"
    description = (
        "Collapse one or more previous tool results to save context. "
        "Call this after you've extracted what you need from a large tool result. "
        "The results remain recoverable via recall_tool_result. "
        "Input: list of tool_use_ids from past tool calls."
    )
    input_model = FreeToolResultInput
    permission = "auto"
    is_concurrent_safe = True

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, FreeToolResultInput)
        session = _current_session.get()
        if session is None:
            return ToolResult(call_id="", output="Error: no active session context.", is_error=True)

        freed: list[str] = []
        not_found: list[str] = []
        already_freed: list[str] = []

        for use_id in input.tool_use_ids:
            if use_id in session.state.freed_tool_results:
                already_freed.append(use_id)
                continue

            original_content: str | None = None
            for msg in session.messages:
                if not isinstance(msg.content, list):
                    continue
                for block in msg.content:
                    if isinstance(block, ToolResultBlock) and block.tool_use_id == use_id:
                        original_content = block.content
                        break
                if original_content is not None:
                    break

            if original_content is None:
                not_found.append(use_id)
                continue

            tool_name = session.state.tool_use_to_tool_name.get(use_id, "unknown")

            snippet = original_content.replace("\n", " ").strip()
            preview = snippet[:80] + "…" if len(snippet) > 80 else snippet

            from neoagent.session import FreedToolResult
            session.state.freed_tool_results[use_id] = FreedToolResult(
                id=use_id,
                tool_name=tool_name,
                size=len(original_content.encode()),
                preview=preview,
                original_content=original_content,
            )
            freed.append(use_id)

        return ToolResult(
            call_id="",
            output=json.dumps({"freed": freed, "not_found": not_found, "already_freed": already_freed}),
        )
