from __future__ import annotations

from pydantic import BaseModel, Field

from neoagent.core.types import ToolResult
from neoagent.tools.base import BaseTool
from neoagent.tools.builtin.tool_search import _current_session


class RecallToolResultInput(BaseModel):
    tool_use_id: str = Field(description="The tool_use_id of a freed tool result to restore for this turn.")


class RecallToolResultTool(BaseTool):
    name = "recall_tool_result"
    description = (
        "Recall the full content of a previously freed tool result. "
        "The content becomes visible to you in this turn only. "
        "To view it again later, call this tool again with the same id."
    )
    input_model = RecallToolResultInput
    permission = "auto"
    is_concurrent_safe = True

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, RecallToolResultInput)
        session = _current_session.get()
        if session is None:
            return ToolResult(call_id="", output="Error: no active session context.", is_error=True)

        freed = session.state.freed_tool_results.get(input.tool_use_id)
        if freed is None:
            return ToolResult(
                call_id="",
                output=f"Error: tool_use_id {input.tool_use_id!r} not found in freed results.",
                is_error=True,
            )

        session.state.recalled_this_turn.add(input.tool_use_id)
        return ToolResult(call_id="", output=freed.original_content)
