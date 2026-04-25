"""One-off demo: rebuild a Session JSON from logs/<id>/events.jsonl so the
old conversation can be resumed via `python examples/v2_repl.py --resume <id>`.

Usage:
    python examples/migrate_log_to_session.py <session_id>
    python examples/migrate_log_to_session.py --list

Caveats:
- Only events emitted by REPL are replayed. State not in events.jsonl
  (compression_failures, total_*_tokens, freed_tool_results, etc.) defaults
  to 0/empty — affects observability counters only, not LLM behaviour.
- created_at is set to now (original timestamp not in events).
- id_gen counters are bumped past max msg/batch ids seen, so newly
  allocated ids after resume won't collide with existing ones.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neoagent.core.types import Message, TextBlock, ToolUseBlock, ToolResultBlock  # noqa: E402
from neoagent.session import Session, SessionState, JsonFileStorage  # noqa: E402
from neoagent.v2.id_gen import SessionIdGenerator  # noqa: E402
from neoagent.v2.schema import Batch, BatchMember, WorkingMemory  # noqa: E402


def _content_from_event(content):
    """events.jsonl stores content as raw dict/list; convert back to typed blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    blocks = []
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            blocks.append(TextBlock(type="text", text=b.get("text", "")))
        elif t == "tool_use":
            blocks.append(ToolUseBlock(
                type="tool_use", id=b["id"], name=b["name"], input=b.get("input", {}),
            ))
        elif t == "tool_result":
            blocks.append(ToolResultBlock(
                type="tool_result",
                tool_use_id=b["tool_use_id"],
                content=b.get("content", ""),
                is_error=b.get("is_error", False),
            ))
    return blocks


def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return datetime.now()


def replay(events_path: Path) -> Session:
    all_messages: dict[str, Message] = {}
    message_order: list[str] = []
    batches: list[Batch] = []
    latest_wm_data: dict | None = None
    session_id: str | None = None
    max_repl_turn = 0
    last_compression_repl_turn = 0
    max_msg_n = 0
    max_batch_n = 0

    with events_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            tag = ev.get("tag")
            data = ev.get("data", {})
            ts = ev.get("ts")
            repl_turn = ev.get("repl_turn", 0)
            max_repl_turn = max(max_repl_turn, repl_turn)
            session_id = session_id or data.get("session_id")

            if tag == "MessageCreated":
                msg_id = data.get("msg_id")
                if not msg_id or msg_id in all_messages:
                    continue
                msg = Message(
                    id=msg_id,
                    turn=data.get("turn"),
                    role=data["role"],
                    content=_content_from_event(data.get("content", "")),
                )
                all_messages[msg_id] = msg
                message_order.append(msg_id)
                if msg_id.startswith("m"):
                    try:
                        max_msg_n = max(max_msg_n, int(msg_id[1:]))
                    except ValueError:
                        pass

            elif tag == "BatchCreated":
                batch_id = data.get("batch_id")
                if not batch_id:
                    continue
                members = [
                    BatchMember(
                        id=m["id"],
                        role=m["role"],
                        preview=m.get("preview", ""),
                        turn=m.get("turn"),
                    )
                    for m in data.get("members", [])
                ]
                created = _parse_iso(ts)
                batch = Batch(
                    session_id=session_id or "unknown",
                    batch_id=batch_id,
                    turns_from=data.get("turns_from", 0),
                    turns_to=data.get("turns_to", 0),
                    time_from=created,
                    time_to=created,
                    summary=data.get("summary"),
                    members=members,
                    trigger="migrated",
                    created_at=created,
                )
                batches.append(batch)
                last_compression_repl_turn = repl_turn
                if batch_id.startswith("cm_"):
                    try:
                        max_batch_n = max(max_batch_n, int(batch_id[3:]))
                    except ValueError:
                        pass

            elif tag == "WMUpdated":
                latest_wm_data = data.get("wm_json") or latest_wm_data

    if not session_id:
        raise RuntimeError("session_id not found in any event")

    compressed_ids = {m.id for batch in batches for m in batch.members}
    live_messages = [all_messages[mid] for mid in message_order if mid not in compressed_ids]
    compressed_messages = [all_messages[mid] for mid in message_order if mid in compressed_ids]

    wm: WorkingMemory | None = None
    if latest_wm_data:
        wm = WorkingMemory(
            session_id=latest_wm_data["session_id"],
            version=latest_wm_data["version"],
            at_turn=latest_wm_data["at_turn"],
            constraints_and_preferences=latest_wm_data.get("constraints_and_preferences", []),
            progress=latest_wm_data.get("progress", ""),
            key_decisions=latest_wm_data.get("key_decisions", []),
            relevant_files=latest_wm_data.get("relevant_files", []),
            next_steps=latest_wm_data.get("next_steps", []),
            critical_context=latest_wm_data.get("critical_context", ""),
            updated_by=latest_wm_data.get("updated_by", "framework_init"),
            updated_at=_parse_iso(latest_wm_data.get("updated_at", "")),
        )

    id_gen = SessionIdGenerator()
    id_gen._msg_counter = max_msg_n
    id_gen._batch_counter = max_batch_n

    state = SessionState()
    state._current_wm = wm
    state.compressed_messages = compressed_messages
    state.batches = batches
    state.user_turn_counter = max_repl_turn
    state.turns_since_last_compression = max(0, max_repl_turn - last_compression_repl_turn)
    state.id_gen = id_gen

    return Session(
        id=session_id,
        messages=live_messages,
        state=state,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


def cmd_list() -> None:
    log_root = REPO_ROOT / "logs"
    if not log_root.exists():
        print("(logs/ directory not found)")
        return
    for d in sorted(log_root.iterdir()):
        ev = d / "events.jsonl"
        if ev.exists():
            mtime = datetime.fromtimestamp(ev.stat().st_mtime).isoformat(timespec="seconds")
            lines = sum(1 for _ in ev.open())
            print(f"  {d.name}   [{mtime}]  ({lines} events)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("session_id", nargs="?", help="session id under logs/")
    p.add_argument("--list", action="store_true", help="list candidate logs and exit")
    args = p.parse_args()

    if args.list or not args.session_id:
        cmd_list()
        if not args.session_id:
            return

    events_path = REPO_ROOT / "logs" / args.session_id / "events.jsonl"
    if not events_path.exists():
        sys.exit(f"error: {events_path} not found")

    session = replay(events_path)
    sessions_dir = REPO_ROOT / "sessions"
    storage = JsonFileStorage(sessions_dir)
    storage.save(session)

    print(f"\n✓ Migrated {session.id}")
    print(f"  Messages live      : {len(session.messages)}")
    print(f"  Messages compressed: {len(session.state.compressed_messages)}")
    print(f"  Batches            : {len(session.state.batches)}")
    print(f"  WM version         : {session.state._current_wm.version if session.state._current_wm else 'none'}")
    print(f"  user_turn_counter  : {session.state.user_turn_counter}")
    print(f"  Saved              : {sessions_dir / (session.id + '.json')}")
    print(f"\nResume:\n  python examples/v2_repl.py --resume {session.id}")


if __name__ == "__main__":
    main()
