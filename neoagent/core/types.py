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
    # id: session-local msg_id assigned by SessionIdGenerator ("m1", "m2", …).
    # Optional for backward compatibility — older messages and Anthropic-native
    # construction won't have it; recall_turn / compression rely on it where set.
    id: str | None = None
    # turn: session-local turn index (0-based, matches TurnCompleteEvent.turn_index).
    # Assigned by QueryLoop at the same time as `id`. Optional for backward compat.
    turn: int | None = None
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
