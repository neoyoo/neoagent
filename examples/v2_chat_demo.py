"""v2.0 Chat Demo — verifies the core v2 SDK wiring end-to-end.

What this demo exercises:
  1. NeoAgent starts with v2 defaults (InMemoryWorkingMemoryStore, security blocks).
  2. LLM calls update_working_memory → WorkingMemoryUpdatedEvent fires, WM snapshot saved.
  3. Event bus observed: MessageCreated / ToolResultPersisted / WMUpdated / TurnComplete.
  4. Final WM state retrievable via agent.wm_store.

What this demo does NOT trigger (requires longer runs):
  - Compression (needs context_budget exceeded)
  - Memory review (needs 10 turns)
  - Source wrap (needs tool with returns_external_content=True)

Reads LLM config from trip-os .env if present, else neoagent .env, else env vars.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


# ── Load .env from trip-os first (Qwen config), then neoagent (Anthropic) ─────
def _load_env() -> None:
    candidates = [
        Path("/Users/neo/Desktop/project/trip-os/.env"),
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if v and k not in os.environ:
                os.environ[k] = v


_load_env()

from datetime import datetime  # noqa: E402

from neoagent import NeoAgent, NeoAgentConfig  # noqa: E402
from neoagent.events import (  # noqa: E402
    MessageCreatedEvent,
    ToolResultPersistedEvent,
    WorkingMemoryUpdatedEvent,
    TurnCompleteEvent,
    ToolCallEvent,
    BatchCreatedEvent,
)
from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryTool  # noqa: E402
from neoagent.tools.builtin.recall_turn import RecallTurnTool  # noqa: E402
from neoagent.v2.schema import WorkingMemory  # noqa: E402


# ── ANSI colours ──────────────────────────────────────────────────────────────
C = dict(
    R="\033[91m", G="\033[92m", Y="\033[93m", B="\033[94m",
    M="\033[95m", C="\033[96m", D="\033[2m", N="\033[0m",
) if sys.stdout.isatty() else {k: "" for k in "RGYBMCDN"}


def _build_config() -> NeoAgentConfig:
    """Resolve provider/model/api_key from env — prefer Anthropic if key set, else Qwen."""
    anth_key = os.environ.get("ANTHROPIC_API_KEY")
    if anth_key:
        return NeoAgentConfig(
            api_key=anth_key,
            model="claude-sonnet-4-6",
            provider="anthropic",
            system_prompt=(
                "You are a coding assistant. When the user asks you to remember "
                "something about the task, call update_working_memory immediately."
            ),
            max_turns=5,
            context_budget=80000,
        )
    llm_key = os.environ.get("LLM_API_KEY")
    if not llm_key:
        raise RuntimeError(
            "No API key found. Set ANTHROPIC_API_KEY or LLM_API_KEY in .env."
        )
    return NeoAgentConfig(
        api_key=llm_key,
        provider="openai",
        base_url=os.environ.get("LLM_BASE_URL"),
        model=os.environ.get("LLM_MODEL", "qwen3.6-plus"),
        system_prompt=(
            "You are a coding assistant. When the user asks you to remember "
            "something about the task, call update_working_memory immediately. "
            "Use prefix 'c' for constraints_and_preferences (e.g. 'c01: ...')."
        ),
        max_turns=5,
        context_budget=80_000,
    )


class EventTape:
    """Collects events with a compact printed record."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def handler(self, tag: str):
        def _h(event) -> None:
            if isinstance(event, MessageCreatedEvent):
                desc = f"msg_id={event.msg_id} role={event.role} source={event.source_type} turn={event.turn}"
            elif isinstance(event, ToolResultPersistedEvent):
                desc = f"tool_use_id={event.tool_use_id} tool={event.tool_name} size={event.size_bytes}B err={event.is_error}"
            elif isinstance(event, WorkingMemoryUpdatedEvent):
                desc = f"version={event.version} at_turn={event.at_turn} by={event.updated_by}"
            elif isinstance(event, ToolCallEvent):
                desc = f"name={event.name} call_id={event.call_id}"
            elif isinstance(event, TurnCompleteEvent):
                desc = f"turn={event.turn_index} stop={event.stop_reason} tool_calls={event.tool_call_count}"
            elif isinstance(event, BatchCreatedEvent):
                desc = f"batch_id={event.batch_id} turns={event.turns_from}-{event.turns_to}"
            else:
                desc = event.__class__.__name__
            self.records.append((tag, desc))
            print(f"{C['D']}  [event] {tag:22s} {desc}{C['N']}")
        return _h


def _print_header(title: str) -> None:
    print(f"\n{C['C']}{'═' * 64}\n{title}\n{'═' * 64}{C['N']}")


async def main() -> None:
    config = _build_config()
    print(f"{C['B']}Provider:{C['N']} {config.provider}  "
          f"{C['B']}Model:{C['N']} {config.model}  "
          f"{C['B']}Base URL:{C['N']} {config.base_url or 'default'}")

    agent = NeoAgent(config)

    # Register v2 built-in tools (not auto-registered)
    session = agent.new_session()
    agent.register_tool(UpdateWorkingMemoryTool(lambda: session.state))
    agent.register_tool(RecallTurnTool(lambda: session.state))

    # framework_init — application layer bootstraps WorkingMemory.
    # SDK intentionally does not auto-init. The session task anchor (if any)
    # goes into `critical_context`.
    session.state._current_wm = WorkingMemory(
        session_id=session.id,
        version=0,
        at_turn=0,
        constraints_and_preferences=[],
        progress="",
        key_decisions=[],
        relevant_files=[],
        next_steps=[],
        critical_context="",
        updated_by="framework_init",
        updated_at=datetime.now(),
    )

    # Subscribe to the four canonical v2 events + ToolCall + TurnComplete + Batch
    tape = EventTape()
    agent.event_bus.subscribe(MessageCreatedEvent, tape.handler("MessageCreated"))
    agent.event_bus.subscribe(ToolResultPersistedEvent, tape.handler("ToolResultPersisted"))
    agent.event_bus.subscribe(WorkingMemoryUpdatedEvent, tape.handler("WMUpdated"))
    agent.event_bus.subscribe(BatchCreatedEvent, tape.handler("BatchCreated"))
    agent.event_bus.subscribe(ToolCallEvent, tape.handler("ToolCall"))
    agent.event_bus.subscribe(TurnCompleteEvent, tape.handler("TurnComplete"))

    prompts = [
        "请用 update_working_memory 往 constraints_and_preferences 追加一条约束："
        "value='c01: 只能使用 aiohttp + selectolax，禁用 Playwright'，op='append'。",
        "再用 update_working_memory 把 progress 设为："
        "'已确认工具选型：aiohttp + selectolax'。field='progress'，op='set'。",
        "基于当前 working memory 的 constraints 和 progress，给我一个最简的 3 步骤实现大纲。",
    ]

    for i, prompt in enumerate(prompts, 1):
        _print_header(f"TURN {i}")
        print(f"{C['Y']}🧑 user:{C['N']} {prompt}\n")
        reply = await agent.chat(prompt, session=session)
        print(f"\n{C['G']}🤖 assistant:{C['N']} {reply[:500]}{'…' if len(reply) > 500 else ''}")

    # ── Final WorkingMemory ──────────────────────────────────────────────────
    _print_header("Final WorkingMemory")
    wm = await agent.wm_store.get_current(session.id)
    if wm is None:
        print(f"{C['R']}  ⚠ no WM snapshot persisted for session {session.id}{C['N']}")
    else:
        print(f"  session_id      : {wm.session_id}")
        print(f"  version         : {wm.version}")
        print(f"  progress        : {wm.progress!r}")
        print(f"  critical_context: {wm.critical_context!r}")
        print(f"  constraints_and_preferences ({len(wm.constraints_and_preferences)}):")
        for c in wm.constraints_and_preferences:
            print(f"     - {c}")
        print(f"  key_decisions   ({len(wm.key_decisions)}): {wm.key_decisions}")
        print(f"  relevant_files  ({len(wm.relevant_files)}): {wm.relevant_files}")
        print(f"  next_steps      ({len(wm.next_steps)}): {wm.next_steps}")

    # ── Event tape summary ───────────────────────────────────────────────────
    _print_header(f"Event tape ({len(tape.records)} total)")
    counts: dict[str, int] = {}
    for tag, _ in tape.records:
        counts[tag] = counts.get(tag, 0) + 1
    for tag, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {tag:22s} × {n}")


if __name__ == "__main__":
    asyncio.run(main())
