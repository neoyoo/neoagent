from __future__ import annotations
import copy
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from neoagent.core.types import Message, TextBlock, ToolUseBlock, ToolResultBlock
from neoagent.v2.schema import WorkingMemory
from neoagent.v2.id_gen import SessionIdGenerator
from neoagent.v2.nudge import NudgeCounter


@dataclass
class FreedToolResult:
    id: str
    tool_name: str
    size: int
    preview: str
    original_content: str


@dataclass
class SessionState:
    """Session 内所有跨轮次共享的可变状态。与 Session 一起持久化。"""
    previous_summary: str | None = None
    compression_failures: int = 0
    memory_tool_calls: int = 0
    memory_token_baseline: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # Tracks which deferred tools have been promoted in this session.
    # DeferredToolRegistry holds the global index; this set controls per-session visibility.
    promoted_tools: set[str] = field(default_factory=set)
    freed_tool_results: dict[str, FreedToolResult] = field(default_factory=dict)
    # IDs recalled this turn — freed rewriting skips these so LLM sees full content.
    recalled_this_turn: set[str] = field(default_factory=set)
    # Maps tool_use_id → tool_name (used by free_tool_result for lookup/display).
    tool_use_to_tool_name: dict[str, str] = field(default_factory=dict)
    # v2 fields ──────────────────────────────────────────────────────────────
    # In-memory current WorkingMemory; None until framework_init sets it.
    # Snapshot to WorkingMemoryStore is handled in Phase 4 Batch 4.
    _current_wm: WorkingMemory | None = field(default=None, repr=False)
    # Session-local monotonic id generator (m1 / cm_1 / d1).
    id_gen: SessionIdGenerator = field(default_factory=SessionIdGenerator)
    # Turn counter; triggers MemoryReview every threshold turns (Phase 7).
    nudge_counter: NudgeCounter = field(default_factory=NudgeCounter)


@dataclass
class Session:
    id: str
    messages: list[Message]
    state: SessionState
    created_at: datetime
    updated_at: datetime
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # _storage is not a dataclass field to avoid init/repr/serialization
        # complications with underscore-prefixed names. We set it here.
        object.__setattr__(self, "_storage", None)

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

    def save_if_storage(self) -> None:
        """Save to bound storage if one has been set via bind_storage(). No-op otherwise."""
        storage = object.__getattribute__(self, "_storage")
        if storage is not None:
            self.save(storage)

    def bind_storage(self, storage: "SessionStorage") -> None:
        """Bind a storage backend so save_if_storage() can be called without explicit args."""
        object.__setattr__(self, "_storage", storage)

    def fork(self, new_id: str | None = None) -> "Session":
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        return Session(
            id=new_id or str(uuid.uuid4()),
            messages=copy.deepcopy(self.messages),
            state=copy.deepcopy(self.state),
            metadata=copy.deepcopy(self.metadata),
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
        import os
        data = _session_to_dict(session)
        content = json.dumps(data, ensure_ascii=False, indent=2)
        path = self._path(session.id)
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, content.encode("utf-8"))
        finally:
            os.close(fd)

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

    def cleanup(self, max_age_days: int = 30, max_sessions: int = 100) -> list[str]:
        import time
        now = time.time()
        max_age_secs = max_age_days * 86400
        paths = sorted(self._base_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        removed: list[str] = []
        surviving = []
        for p in paths:
            if now - p.stat().st_mtime > max_age_secs:
                p.unlink(missing_ok=True)
                removed.append(p.stem)
            else:
                surviving.append(p)
        if len(surviving) > max_sessions:
            to_remove = surviving[:len(surviving) - max_sessions]
            for p in to_remove:
                p.unlink(missing_ok=True)
                removed.append(p.stem)
        return removed


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


def _wm_to_dict(wm: WorkingMemory) -> dict:
    """Serialize a WorkingMemory to a JSON-safe dict."""
    return {
        "session_id": wm.session_id,
        "version": wm.version,
        "at_turn": wm.at_turn,
        "constraints_and_preferences": list(wm.constraints_and_preferences),
        "progress": wm.progress,
        "key_decisions": list(wm.key_decisions),
        "relevant_files": list(wm.relevant_files),
        "next_steps": list(wm.next_steps),
        "critical_context": wm.critical_context,
        "updated_by": wm.updated_by,
        "updated_at": wm.updated_at.isoformat(),
    }


def _wm_from_dict(d: dict) -> WorkingMemory:
    """Deserialize a WorkingMemory from a dict produced by _wm_to_dict."""
    return WorkingMemory(
        session_id=d["session_id"],
        version=d["version"],
        at_turn=d["at_turn"],
        constraints_and_preferences=list(d.get("constraints_and_preferences", [])),
        progress=d.get("progress", ""),
        key_decisions=list(d.get("key_decisions", [])),
        relevant_files=list(d.get("relevant_files", [])),
        next_steps=list(d.get("next_steps", [])),
        critical_context=d.get("critical_context", ""),
        updated_by=d["updated_by"],
        updated_at=datetime.fromisoformat(d["updated_at"]),
    )


def _session_to_dict(session: Session) -> dict:
    wm = session.state._current_wm
    id_gen = session.state.id_gen
    nc = session.state.nudge_counter
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
            "promoted_tools": sorted(session.state.promoted_tools),
            "freed_tool_results": {
                k: {
                    "id": v.id,
                    "tool_name": v.tool_name,
                    "size": v.size,
                    "preview": v.preview,
                    "original_content": v.original_content,
                }
                for k, v in session.state.freed_tool_results.items()
            },
            "tool_use_to_tool_name": session.state.tool_use_to_tool_name,
            # v2 fields
            "_current_wm": _wm_to_dict(wm) if wm is not None else None,
            "id_gen": {
                "_msg_counter": id_gen._msg_counter,
                "_batch_counter": id_gen._batch_counter,
                "_decision_counter": id_gen._decision_counter,
            },
            "nudge_counter": {
                "threshold": nc.threshold,
                "_count": nc._count,
            },
        },
        "messages": [_message_to_dict(m) for m in session.messages],
        "metadata": session.metadata,
    }


def _session_from_dict(data: dict) -> Session:
    state_d = data.get("state", {})
    _freed_raw = state_d.get("freed_tool_results", {})
    _freed = {
        k: FreedToolResult(
            id=v["id"],
            tool_name=v["tool_name"],
            size=v["size"],
            preview=v["preview"],
            original_content=v["original_content"],
        )
        for k, v in _freed_raw.items()
    }

    # v2: _current_wm — None if missing or null (backward compat)
    _wm_raw = state_d.get("_current_wm", None)
    _current_wm: WorkingMemory | None = _wm_from_dict(_wm_raw) if _wm_raw is not None else None

    # v2: id_gen — fresh instance if missing (backward compat)
    _id_gen_raw = state_d.get("id_gen", None)
    if _id_gen_raw is not None:
        id_gen = SessionIdGenerator()
        id_gen._msg_counter = _id_gen_raw["_msg_counter"]
        id_gen._batch_counter = _id_gen_raw["_batch_counter"]
        id_gen._decision_counter = _id_gen_raw["_decision_counter"]
    else:
        id_gen = SessionIdGenerator()

    # v2: nudge_counter — fresh instance if missing (backward compat)
    _nc_raw = state_d.get("nudge_counter", None)
    if _nc_raw is not None:
        nudge_counter = NudgeCounter(threshold=_nc_raw["threshold"])
        nudge_counter._count = _nc_raw["_count"]
    else:
        nudge_counter = NudgeCounter()

    state = SessionState(
        previous_summary=state_d.get("previous_summary"),
        compression_failures=state_d.get("compression_failures", 0),
        memory_tool_calls=state_d.get("memory_tool_calls", 0),
        memory_token_baseline=state_d.get("memory_token_baseline", 0),
        total_input_tokens=state_d.get("total_input_tokens", 0),
        total_output_tokens=state_d.get("total_output_tokens", 0),
        promoted_tools=set(state_d.get("promoted_tools", [])),
        freed_tool_results=_freed,
        tool_use_to_tool_name=state_d.get("tool_use_to_tool_name", {}),
        _current_wm=_current_wm,
        id_gen=id_gen,
        nudge_counter=nudge_counter,
    )
    messages = [_message_from_dict(m) for m in data.get("messages", [])]
    return Session(
        id=data["id"],
        messages=messages,
        state=state,
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
        metadata=data.get("metadata", {}),
    )
