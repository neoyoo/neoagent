from __future__ import annotations
import copy
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from neoagent.core.types import Message, TextBlock, ToolUseBlock, ToolResultBlock


@dataclass
class SessionState:
    """Session 内所有跨轮次共享的可变状态。与 Session 一起持久化。"""
    previous_summary: str | None = None
    compression_failures: int = 0
    memory_tool_calls: int = 0
    memory_token_baseline: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


@dataclass
class Session:
    id: str
    messages: list[Message]
    state: SessionState
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(cls, session_id: str | None = None) -> "Session":
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        return cls(
            id=session_id or str(uuid.uuid4()),
            messages=[],
            state=SessionState(),
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def resume(cls, session_id: str, storage: "SessionStorage") -> "Session":
        return storage.load(session_id)

    def save(self, storage: "SessionStorage") -> None:
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        storage.save(self)

    def fork(self, new_id: str | None = None) -> "Session":
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        return Session(
            id=new_id or str(uuid.uuid4()),
            messages=copy.deepcopy(self.messages),
            state=copy.deepcopy(self.state),
            created_at=now,
            updated_at=now,
        )


@runtime_checkable
class SessionStorage(Protocol):
    def save(self, session: Session) -> None: ...
    def load(self, session_id: str) -> Session: ...
    def list_ids(self) -> list[str]: ...
    def delete(self, session_id: str) -> None: ...


class JsonFileStorage:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        path = (self._base_dir / f"{session_id}.json").resolve()
        if not path.is_relative_to(self._base_dir.resolve()):
            raise ValueError(f"Invalid session id: {session_id!r}")
        return path

    def save(self, session: Session) -> None:
        data = _session_to_dict(session)
        self._path(session.id).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.exists():
            raise KeyError(f"Session not found: {session_id}")
        data = json.loads(path.read_text())
        return _session_from_dict(data)

    def list_ids(self) -> list[str]:
        return [p.stem for p in self._base_dir.glob("*.json")]

    def delete(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)


def _message_to_dict(msg: Message) -> dict:
    if isinstance(msg.content, str):
        return {"role": msg.role, "content": msg.content}
    blocks = []
    for b in msg.content:
        if isinstance(b, TextBlock):
            blocks.append({"type": "text", "text": b.text})
        elif isinstance(b, ToolUseBlock):
            blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        elif isinstance(b, ToolResultBlock):
            blocks.append({"type": "tool_result", "tool_use_id": b.tool_use_id,
                           "content": b.content, "is_error": b.is_error})
    return {"role": msg.role, "content": blocks}


def _message_from_dict(d: dict) -> Message:
    content = d["content"]
    if isinstance(content, str):
        return Message(role=d["role"], content=content)
    blocks = []
    for b in content:
        t = b["type"]
        if t == "text":
            blocks.append(TextBlock(text=b["text"]))
        elif t == "tool_use":
            blocks.append(ToolUseBlock(id=b["id"], name=b["name"], input=b["input"]))
        elif t == "tool_result":
            blocks.append(ToolResultBlock(tool_use_id=b["tool_use_id"],
                                          content=b["content"], is_error=b.get("is_error", False)))
    return Message(role=d["role"], content=blocks)


def _session_to_dict(session: Session) -> dict:
    return {
        "id": session.id,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "state": {
            "previous_summary": session.state.previous_summary,
            "compression_failures": session.state.compression_failures,
            "memory_tool_calls": session.state.memory_tool_calls,
            "memory_token_baseline": session.state.memory_token_baseline,
            "total_input_tokens": session.state.total_input_tokens,
            "total_output_tokens": session.state.total_output_tokens,
        },
        "messages": [_message_to_dict(m) for m in session.messages],
    }


def _session_from_dict(data: dict) -> Session:
    state_d = data.get("state", {})
    state = SessionState(
        previous_summary=state_d.get("previous_summary"),
        compression_failures=state_d.get("compression_failures", 0),
        memory_tool_calls=state_d.get("memory_tool_calls", 0),
        memory_token_baseline=state_d.get("memory_token_baseline", 0),
        total_input_tokens=state_d.get("total_input_tokens", 0),
        total_output_tokens=state_d.get("total_output_tokens", 0),
    )
    messages = [_message_from_dict(m) for m in data.get("messages", [])]
    return Session(
        id=data["id"],
        messages=messages,
        state=state,
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
    )
