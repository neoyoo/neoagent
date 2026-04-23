"""v2 Terminal REPL — full-feature v2.0 chat with per-turn context logging.

Runs an interactive loop against the v2.0 SDK. Every LLM request is dumped
to logs/<session_id>/turnNNN_reqKK.json (complete system prompt + messages +
tools as sent to the provider). Every event goes to events.jsonl. A human-
readable summary is appended to summary.md after each turn.

Wiring exercised:
  - PromptBuilder: identity + security blocks + injected skill
  - WorkingMemory: framework_init + update_working_memory tool + snapshot path
  - Compression: v2 strategy (OneShotCompressionStrategy via Qwen/Anthropic)
  - Free/Recall: v1 free_tool_result + v2 recall_turn built-ins
  - Tool use: read_file tool (returns_external_content → <source> wrap)
  - Events: MessageCreated / ToolResultPersisted / WMUpdated / BatchCreated / ...

REPL commands:
  /quit      exit
  /wm        print current WorkingMemory as JSON
  /ctx       print the current system prompt
  /tools     list registered tools

Env (all read from neoagent/.env):
  ANTHROPIC_API_KEY   required
  ANTHROPIC_BASE_URL  required (e.g. https://dashscope.aliyuncs.com/apps/anthropic)
  ANTHROPIC_MODEL     required (e.g. qwen3.5-plus)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path


# ── Load neoagent/.env (project-local, never trip-os's) ──────────────────────
def _load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if v and k not in os.environ:
            os.environ[k] = v


_load_env()

from pydantic import BaseModel  # noqa: E402

from neoagent import NeoAgent, NeoAgentConfig  # noqa: E402
from neoagent.core.prompt import PromptSection  # noqa: E402
from neoagent.core.types import ToolResult  # noqa: E402
from neoagent.events import (  # noqa: E402
    BatchCreatedEvent,
    CompressionFailedEvent,
    MessageCreatedEvent,
    ProviderRequestEvent,
    ProviderResponseEvent,
    ToolCallEvent,
    ToolResultPersistedEvent,
    TurnCompleteEvent,
    WorkingMemoryUpdatedEvent,
)
from neoagent.tools.base import BaseTool  # noqa: E402
from neoagent.tools.builtin.recall_turn import RecallTurnTool  # noqa: E402
from neoagent.tools.builtin.update_working_memory import UpdateWorkingMemoryTool  # noqa: E402
from neoagent.v2.schema import WorkingMemory  # noqa: E402
from neoagent.v2.strategies.oneshot_compression import OneShotCompressionStrategy  # noqa: E402
from neoagent.v2.strategies.oneshot_memory_review import OneShotMemoryReviewStrategy  # noqa: E402


# ── read_file tool — real tool_use + returns_external_content triggers <source> wrap ─
class ReadFileInput(BaseModel):
    path: str


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read a small text file relative to the working directory and return its "
        "contents. Path is restricted to the working tree; max 20KB per read."
    )
    input_model = ReadFileInput
    permission = "auto"
    is_concurrent_safe = True
    returns_external_content = True

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()

    async def execute(self, input: ReadFileInput) -> ToolResult:  # type: ignore[override]
        try:
            path = (self.base_dir / input.path).resolve()
            path.relative_to(self.base_dir)
        except ValueError:
            return ToolResult(call_id="", output=f"error: {input.path} escapes base_dir", is_error=True)
        if not path.exists() or not path.is_file():
            return ToolResult(call_id="", output=f"error: {input.path} not found", is_error=True)
        if path.stat().st_size > 20_000:
            return ToolResult(call_id="", output=f"error: file too large ({path.stat().st_size}B)", is_error=True)
        return ToolResult(call_id="", output=path.read_text(errors="replace"))


# ── Serialization helpers ────────────────────────────────────────────────────
def _to_jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    # MappingProxyType (read-only dict view) must be handled before dict check below
    if isinstance(obj, (dict, types.MappingProxyType)):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if is_dataclass(obj):
        # Walk fields manually — asdict() uses deepcopy which breaks on
        # MappingProxyType inside ToolCallEvent.input_data.
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return repr(obj)


def _render_content_flat(content) -> str:
    """Render one message's content field as a flat string with XML-ish block tags.

    Tool calls/results get explicit <tool_use>/<tool_result> wrappers so the view
    mirrors what the LLM sees, and any provider-injected tags (<source>, <working_memory>,
    <compressed_history>, …) inside text blocks remain verbatim.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return repr(content)
    out: list[str] = []
    for block in content:
        # Accept both dict-style (provider payload) and pydantic-style (SDK type) blocks.
        def _get(key, default=None):
            if isinstance(block, dict):
                return block.get(key, default)
            return getattr(block, key, default)
        btype = _get("type") or type(block).__name__
        if btype == "text":
            out.append(str(_get("text", "")))
        elif btype == "tool_use":
            tool_name = _get("name", "?")
            tool_id = _get("id", "?")
            tool_input = _get("input", {})
            try:
                input_str = json.dumps(_to_jsonable(tool_input), ensure_ascii=False, indent=2)
            except Exception:
                input_str = repr(tool_input)
            out.append(f'<tool_use name="{tool_name}" id="{tool_id}">\n{input_str}\n</tool_use>')
        elif btype == "tool_result":
            tool_use_id = _get("tool_use_id", "?")
            c = _get("content", "")
            is_err = _get("is_error", False)
            err_attr = ' is_error="true"' if is_err else ""
            body = c if isinstance(c, str) else _render_content_flat(c)
            out.append(f'<tool_result tool_use_id="{tool_use_id}"{err_attr}>\n{body}\n</tool_result>')
        else:
            out.append(repr(block))
    return "\n".join(out)


def _render_flat_llm_view(turn: int, sub_req: int, event: "ProviderRequestEvent") -> str:
    """Produce the 'what the LLM actually sees' flat text dump for one request."""
    header = f"TURN {turn} / SUB-REQUEST {sub_req}   (llm_turn_idx={event.turn})"
    lines: list[str] = [
        "=" * 78,
        header,
        "=" * 78,
        "",
        "━━━ SYSTEM PROMPT " + "━" * 60,
        event.system,
        "",
        f"━━━ TOOLS ({len(event.tools)}) " + "━" * 58,
    ]
    for tool in event.tools:
        if isinstance(tool, dict):
            name = tool.get("name", "?")
            desc = tool.get("description", "") or ""
        else:
            name = getattr(tool, "name", "?")
            desc = getattr(tool, "description", "") or ""
        lines.append(f"  · {name} — {desc[:120]}")
    lines.append("")
    lines.append(f"━━━ MESSAGES ({len(event.messages)}) " + "━" * 55)
    lines.append("")
    for i, msg in enumerate(event.messages, 1):
        if isinstance(msg, dict):
            role = msg.get("role", "?")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "?")
            content = getattr(msg, "content", "")
        lines.append(f"───── [{i}] role={role} " + "─" * 50)
        lines.append(_render_content_flat(content))
        lines.append("")
    return "\n".join(lines)


def _wm_to_dict(wm: WorkingMemory | None) -> dict:
    if wm is None:
        return {}
    return {
        "session_id": wm.session_id,
        "version": wm.version,
        "at_turn": wm.at_turn,
        "progress": wm.progress,
        "critical_context": wm.critical_context,
        "constraints_and_preferences": list(wm.constraints_and_preferences),
        "key_decisions": list(wm.key_decisions),
        "relevant_files": list(wm.relevant_files),
        "next_steps": list(wm.next_steps),
        "updated_by": wm.updated_by,
        "updated_at": wm.updated_at.isoformat(),
    }


# ── Per-session logger ───────────────────────────────────────────────────────
class SessionLogger:
    def __init__(self, log_root: Path, session_id: str) -> None:
        self.dir = log_root / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self.summary_path = self.dir / "summary.md"
        self.turn = 0
        self.req_in_turn = 0
        self.event_buffer: list[tuple[str, object]] = []
        self.summary_path.write_text(f"# Session {session_id}\n\nStarted {datetime.now().isoformat()}\n")

    def start_turn(self, user_input: str) -> None:
        self.turn += 1
        self.req_in_turn = 0
        self.event_buffer.clear()
        with self.summary_path.open("a") as f:
            f.write(f"\n---\n\n## TURN {self.turn}\n\n**user:**\n\n```\n{user_input}\n```\n")

    def log_request(self, event: ProviderRequestEvent) -> None:
        self.req_in_turn += 1
        base = self.dir / f"turn{self.turn:03d}_req{self.req_in_turn:02d}"
        payload = {
            "repl_turn": self.turn,
            "sub_request": self.req_in_turn,
            "llm_turn_idx": event.turn,
            "system": event.system,
            "messages": _to_jsonable(event.messages),
            "tools": _to_jsonable(event.tools),
        }
        base.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        # Human-readable flat view — XML tags (<working_memory> / <source> /
        # <compressed_history> / ...) appear verbatim so they are easy to eyeball.
        base.with_suffix(".txt").write_text(
            _render_flat_llm_view(self.turn, self.req_in_turn, event)
        )

    def log_event(self, tag: str, event: object) -> None:
        self.event_buffer.append((tag, event))
        rec = {
            "ts": datetime.now().isoformat(),
            "repl_turn": self.turn,
            "tag": tag,
            "data": _to_jsonable(event),
        }
        with self.events_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def end_turn(self, reply: str, wm: WorkingMemory | None) -> None:
        with self.summary_path.open("a") as f:
            f.write(f"\n**events ({len(self.event_buffer)}):**\n")
            for tag, ev in self.event_buffer:
                f.write(f"- `{tag}`: {_describe_event(ev)}\n")
            if wm is not None:
                f.write(f"\n**WorkingMemory:**\n```json\n{json.dumps(_wm_to_dict(wm), ensure_ascii=False, indent=2)}\n```\n")
            f.write(f"\n**assistant:**\n\n{reply}\n")


def _describe_event(ev: object) -> str:
    if isinstance(ev, MessageCreatedEvent):
        return f"msg_id={ev.msg_id} role={ev.role} source={ev.source_type} turn={ev.turn}"
    if isinstance(ev, ToolResultPersistedEvent):
        return f"tool_use_id={ev.tool_use_id} tool={ev.tool_name} size={ev.size_bytes}B err={ev.is_error}"
    if isinstance(ev, WorkingMemoryUpdatedEvent):
        return f"version={ev.version} at_turn={ev.at_turn} by={ev.updated_by}"
    if isinstance(ev, ToolCallEvent):
        return f"name={ev.name} call_id={ev.call_id}"
    if isinstance(ev, TurnCompleteEvent):
        return f"turn_idx={ev.turn_index} stop={ev.stop_reason} tool_calls={ev.tool_call_count}"
    if isinstance(ev, BatchCreatedEvent):
        return f"batch_id={ev.batch_id} turns={ev.turns_from}-{ev.turns_to} members={len(ev.members)}"
    if isinstance(ev, CompressionFailedEvent):
        return f"reason={ev.reason} retry={ev.retry_count}"
    if isinstance(ev, ProviderResponseEvent):
        return f"stop={ev.stop_reason} in={ev.input_tokens} out={ev.output_tokens}"
    return type(ev).__name__


# ── Skill content ────────────────────────────────────────────────────────────
PYTHON_STYLE_SKILL = """\
[SKILL: python_style]
When discussing or writing Python code in this session:
- Always use type annotations on function signatures.
- Prefer async/await over synchronous blocking calls.
- Favour dataclasses or Pydantic models over bare dicts for structured data.
- Write small, composable functions; reserve classes for state machines and protocols.
- Name exception types with the `Error` suffix.
"""


# ── main ─────────────────────────────────────────────────────────────────────
async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    model = os.environ.get("ANTHROPIC_MODEL")
    missing = [
        name for name, val in (
            ("ANTHROPIC_API_KEY", api_key),
            ("ANTHROPIC_BASE_URL", base_url),
            ("ANTHROPIC_MODEL", model),
        ) if not val
    ]
    if missing:
        print(f"error: missing env vars {missing} — fill them in neoagent/.env")
        sys.exit(1)

    compression_strategy = OneShotCompressionStrategy(
        api_key=api_key, model=model, base_url=base_url, max_retries=2,
    )
    review_strategy = OneShotMemoryReviewStrategy(
        api_key=api_key, model=model, base_url=base_url, max_retries=2,
    )

    config = NeoAgentConfig(
        api_key=api_key,
        provider="anthropic",
        base_url=base_url,
        model=model,
        system_prompt=(
            "You are an assistant. Task context and constraints are defined "
            "by the <working_memory> block rendered below — treat it as "
            "authoritative, and follow its `critical_context` / "
            "`constraints_and_preferences` strictly. Tools you can use:\n"
            "- read_file: inspect a file in the working directory\n"
            "- update_working_memory: add/update facts, decisions, next_steps\n"
            "- free_tool_result: collapse a large tool output after extracting what you need\n"
            "- recall_turn: recover compressed turns when you need their detail"
        ),
        max_turns=10,
        context_budget=3_000,  # small so compression triggers within a few turns
        compression_strategy=compression_strategy,
        memory_review_strategy=review_strategy,
    )
    agent = NeoAgent(config)
    session = agent.new_session()

    # framework_init — application layer bootstraps an empty WM.
    # The user does not fill any of these fields; the LLM populates them
    # as the conversation proceeds via update_working_memory, and real
    # applications (e.g. trip-os) may seed fields based on their own onboarding.
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

    # Register tools: v2 builtins + read_file + (optional) v1 free_tool_result
    agent.register_tool(ReadFileTool(base_dir=Path.cwd()))
    agent.register_tool(UpdateWorkingMemoryTool(lambda: session.state))
    agent.register_tool(RecallTurnTool(lambda: session.state))
    try:
        from neoagent.tools.builtin.free_tool_result import FreeToolResultTool
        agent.register_tool(FreeToolResultTool())
    except ImportError:
        pass

    # Render the current WorkingMemory into every system prompt build — this is
    # how the LLM sees `critical_context` / `constraints_and_preferences` /
    # `progress`, so the session's task anchor lives in WM rather than a static string.
    def _render_wm_section() -> str:
        wm = session.state._current_wm
        if wm is None:
            return ""
        lines = [
            "<working_memory>",
            f"critical_context: {wm.critical_context or '(none)'}",
            f"progress: {wm.progress or '(none)'}",
        ]
        for label, items in (
            ("constraints_and_preferences", wm.constraints_and_preferences),
            ("key_decisions", wm.key_decisions),
            ("relevant_files", wm.relevant_files),
            ("next_steps", wm.next_steps),
        ):
            lines.append(f"{label}:")
            if items:
                for it in items:
                    lines.append(f"  - {it}")
            else:
                lines.append("  (none)")
        lines.append("</working_memory>")
        return "\n".join(lines)

    agent._prompt_builder.add_section(PromptSection(
        name="working_memory",
        content=_render_wm_section,
        priority=50,           # above identity (which is priority=0) so it renders after
        is_static=False,       # re-rendered on every build
    ))

    # Inject a skill into PromptBuilder (activated — visible to LLM from turn 1)
    agent._prompt_builder.register_skill(
        "python_style",
        PromptSection(name="python_style", content=PYTHON_STYLE_SKILL, priority=10, is_static=True),
    )
    agent._prompt_builder.activate_skill("python_style")

    # Subscribe logger to events
    log_root = Path.cwd() / "logs"
    logger = SessionLogger(log_root, session.id)
    bus = agent.event_bus
    bus.subscribe(ProviderRequestEvent, logger.log_request)
    bus.subscribe(ProviderResponseEvent, lambda e: logger.log_event("ProviderResponse", e))
    bus.subscribe(MessageCreatedEvent, lambda e: logger.log_event("MessageCreated", e))
    bus.subscribe(ToolCallEvent, lambda e: logger.log_event("ToolCall", e))
    bus.subscribe(ToolResultPersistedEvent, lambda e: logger.log_event("ToolResultPersisted", e))
    bus.subscribe(WorkingMemoryUpdatedEvent, lambda e: logger.log_event("WMUpdated", e))
    bus.subscribe(BatchCreatedEvent, lambda e: logger.log_event("BatchCreated", e))
    bus.subscribe(CompressionFailedEvent, lambda e: logger.log_event("CompressionFailed", e))
    bus.subscribe(TurnCompleteEvent, lambda e: logger.log_event("TurnComplete", e))

    print(f"\nSession  : {session.id}")
    print(f"Provider : anthropic @ {base_url}")
    print(f"Model    : {model}")
    print(f"Budget   : {config.context_budget} tokens")
    print(f"Logs     : {logger.dir}")
    print(f"Tools    : {', '.join(sorted(agent._registry.all_tools().keys()))}")
    print(f"Skill    : python_style (active)")
    print("Commands : /quit  /wm  /ctx  /tools")
    print(f"read_file base_dir = {Path.cwd()}\n")

    while True:
        try:
            user_input = await asyncio.to_thread(input, "🧑 >>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/wm":
            wm = await agent.wm_store.get_current(session.id) or session.state._current_wm
            print(json.dumps(_wm_to_dict(wm), ensure_ascii=False, indent=2))
            continue
        if user_input == "/ctx":
            print(agent._prompt_builder.build())
            continue
        if user_input == "/tools":
            for name, tool in sorted(agent._registry.all_tools().items()):
                print(f"  {name:30s} {tool.description[:80]}")
            continue

        logger.start_turn(user_input)
        try:
            reply = await agent.chat(user_input, session=session)
        except Exception as exc:
            print(f"\n⚠️  chat error: {type(exc).__name__}: {exc}\n")
            continue
        wm = await agent.wm_store.get_current(session.id) or session.state._current_wm
        logger.end_turn(reply, wm)
        print(f"\n🤖 {reply}\n")
        print(f"   ({len(logger.event_buffer)} events → {logger.dir.name}/turn{logger.turn:03d}_*.json)\n")


if __name__ == "__main__":
    asyncio.run(main())
