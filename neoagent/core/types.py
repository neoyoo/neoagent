from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict

class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False

ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]

@dataclass
class ToolCall:
    id: str
    name: str
    input: dict

@dataclass
class ToolResult:
    call_id: str
    output: str
    is_error: bool = False

class Turn(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    response: Message
    tool_calls: list[ToolCall] = []
    tool_results: list[ToolResult] = []
    stop_reason: Literal["end_turn", "tool_use", "max_tokens"]

class ConversationResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    turns: list[Turn] = []
    reason: Literal["completed", "max_turns"]
