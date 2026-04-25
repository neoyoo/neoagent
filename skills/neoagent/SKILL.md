---
name: neoagent
description: >
  Use when building agents with the neoagent SDK, understanding how SDK modules
  connect, adding capabilities (memory / custom tools / multi-agent / hooks / HTTP
  channel), or reading/debugging existing neoagent code.
---

# neoagent SDK

> Python async agent framework. Production-ready. Local install only (not on PyPI).

## Install

```bash
pip install -e /path/to/neoagent
```

Replace `/path/to/neoagent` with the absolute path to the neoagent repository on disk.

## Quick Start

```python
import asyncio
import os
from neoagent import NeoAgent, NeoAgentConfig

config = NeoAgentConfig(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-6",
    system_prompt="You are a helpful assistant.",
    max_turns=30,
    # Set context_budget explicitly — default 0 means no limit, which can cause
    # runaway context growth in long sessions.
    context_budget=80_000,
)
agent = NeoAgent(config)

async def main() -> None:
    reply: str = await agent.chat("Hello")
    print(reply)

asyncio.run(main())
```

**Entry points:**
- `agent.chat(message)` → single-turn, returns `str`
- `agent.run(messages, max_turns=N)` → full conversation, returns `ConversationResult`

---

## Module Architecture

All 12 modules and their data-flow relationships:

````
NeoAgentConfig (config.py)
    │
    ▼
NeoAgent (agent.py)  ◄── sole entry point: chat() / run()
    │
    ├── [prompt-system] PromptBuilder (core/prompt.py)
    │       add_section(PromptSection)       permanent prompt blocks
    │       register_skill(name, section)    register skill (inactive)
    │       activate_skill(name)             inject skill → system prompt
    │       deactivate_skill(name)           remove skill from system prompt
    │
    ├── [query-loop] QueryLoop (core/loop.py)   ← execution engine / state machine
    │       │
    │       ├── [tool-system] Provider (providers/)    API calls (Anthropic / OpenAI)
    │       │
    │       ├── ToolExecutor
    │       │       ToolRegistry                  active tools (LLM-visible)
    │       │       DeferredToolRegistry          tool pool (hidden until promoted)
    │       │           ▲ tool_search built-in promotes on demand
    │       │
    │       ├── [hooks] HookManager (hooks.py)    pre/post interception
    │       │       hook(type, handler)           register via method
    │       │       @agent.on(type)               register via decorator
    │       │
    │       └── [context-management] ContextCompressor (core/compress.py)  auto compress
                   should_compress(messages, schemas, context_budget, turns_since_last_compression)
                       → tuple[bool, str]  reason ∈ "turn_count" | "token_threshold" | ""
                       Double trigger: every 10 user turns OR token > 0.7 * budget
    │
    ├── [memory-system] MemoryManager (memory/)    cross-session memory
    │       MemoryExtractor         extract memories after turns
    │       MemoryRetriever         retrieve relevant memories
    │       MemoryStore             persist to disk
    │       (enabled when memory_dir + memory_project_key are set in config)
    │
    ├── [runtime-state / session-recovery] Session + SessionState (session.py)
    │       Session.create()                new session
    │       agent.resume(session_id)        load from storage (preferred API)
    │       Session.resume(id, storage)     low-level classmethod
    │       JsonFileStorage                 persist to session_dir
    │       (enabled when session_dir is set in config)
    │
    ├── [hooks] EventBus (events.py)         publish/subscribe observability bus
    │       agent.event_bus                 public property (use this, not _event_bus)
    │       agent.event_bus.subscribe(EventClass, handler)
    │       agent.event_bus.unsubscribe(EventClass, handler)
    │
    └── [evaluation-observability] observe.py + observe_subscriber.py
            Observer(console=True, log_dir=Path(...))
            ObserverSubscriber(observer).attach(agent.event_bus)

─── Independent extension modules ────────────────────────────────────────

[mcp-skills] MCP (mcp/)                    [multi-agent] Multi-Agent (multi/)
    await agent.add_mcp_server(               Orchestrator(config, max_depth=2)
        name, command, env)                   register_worker(WorkerCard(...))
    → tools auto-injected to                  await orchestrator.run(message)
      DeferredToolRegistry                    await orchestrator.close()

[channel-remote] Channel (channels/)       [evaluation-observability] Eval (eval/)
    FastAPIChannel(agent, host, port)          EvalRunner(agent)
    await channel.serve_forever()             await runner.run(cases) → EvalReport
    POST /v1/run  /v1/run/stream               EvalCase(messages=[...], assertion=fn)
````

**v2 internals — `neoagent/v2/`:**

```
neoagent/v2/
    schema.py           # WorkingMemory, Batch, BatchMember, CompressionDelta, Layer enum
    id_gen.py           # SessionIdGenerator (m1, m2, cm_1, d1)
    nudge.py            # NudgeCounter (memory review trigger)
    compressed_store.py # CompressedMessageStore Protocol + InMemory default
    abc.py              # WorkingMemoryStore Protocol
    stores/             # InMemoryWorkingMemoryStore (default)
    strategies/         # OneShotCompressionStrategy, OneShotMemoryReviewStrategy
    security/           # security helpers (rate limiting, etc.)
    prompts/            # security_blocks.py + ABC contracts
```

---

## wiki Module → neoagent Source

| wiki module | neoagent implementation |
|-------------|------------------------|
| query-loop | `neoagent/core/loop.py` |
| prompt-system | `neoagent/core/prompt.py` (PromptBuilder + PromptSection) |
| tool-system | `neoagent/tools/` — BaseTool, ToolRegistry, DeferredToolRegistry, ToolExecutor |
| context-management | `neoagent/core/compress.py` — ContextCompressor |
| memory-system | `neoagent/memory/` — MemoryExtractor, MemoryRetriever, MemoryStore |
| runtime-state | `neoagent/session.py` — SessionState, Session |
| session-recovery | `neoagent/session.py` — Session.resume() + JsonFileStorage |
| hooks | `neoagent/hooks.py` — HookManager, HookResult; `neoagent/events.py` — EventBus |
| mcp-skills | `neoagent/mcp/` — MCPClient, MCPTool, MCPTransport |
| multi-agent | `neoagent/multi/` — Orchestrator, WorkerCard |
| channel-remote | `neoagent/channels/` — FastAPIChannel, BaseChannel |
| evaluation-observability | `neoagent/eval/` — EvalRunner, EvalCase, EvalReport; `neoagent/observe.py` — Observer; `neoagent/observe_subscriber.py` — ObserverSubscriber |

---

## v2 Extension Points

All four pluggable abstractions follow the same Protocol pattern — implement the interface and wire via `NeoAgentConfig`.

| Protocol | Config field | Default |
|---|---|---|
| `SessionStorage` | `session_dir` (uses `JsonFileStorage`) | `JsonFileStorage` |
| `WorkingMemoryStore` | `working_memory_store` | `InMemoryWorkingMemoryStore` |
| `CompressionStrategy` | `compression_strategy` | `OneShotCompressionStrategy` |
| `MemoryReviewStrategy` | `memory_review_strategy` | `OneShotMemoryReviewStrategy` |
| `CompressedMessageStore` | `compressed_message_store` | `InMemoryCompressedMessageStore` |

### `CompressedMessageStore` Protocol (`neoagent/v2/compressed_store.py`)

```python
class CompressedMessageStore(Protocol):
    async def put_many(self, session_id: str, messages: list[Message]) -> None: ...
    async def get_many(self, session_id: str, msg_ids: list[str]) -> dict[str, Message]: ...
    async def list_ids(self, session_id: str) -> list[str]: ...
    async def delete_session(self, session_id: str) -> None: ...
```

Default: `InMemoryCompressedMessageStore` — wraps `session_state.compressed_messages` so `JsonFileStorage` continues to round-trip bodies via session JSON.

Custom backends (ES / Qdrant / Postgres / S3) implement the Protocol and are wired via `NeoAgentConfig.compressed_message_store`. Compression writes via `put_many`; `recall_turn` reads via `get_many`. They form a coupled pair — both go through the store.

---

## Built-in Tools

These tools are always registered by the SDK and are LLM-callable without additional setup.

| Tool | Path | Purpose |
|---|---|---|
| `tool_search` | `neoagent/tools/builtin/tool_search.py` | Promote tools from DeferredToolRegistry into the active set when needed |
| `recall_turn` | `neoagent/tools/builtin/recall_turn.py` | Recall full original content of `msg_ids` referenced in `<compressed_history>`'s `<recoverable>` block. Queries live + compressed messages. Input: `RecallTurnInput(msg_ids: list[str])` |
| `update_working_memory` | `neoagent/tools/builtin/update_working_memory.py` | LLM-callable WM mutation: `field`, `op`, `value`. Append-only for list fields, set for scalars |
| `free_tool_result` | `neoagent/tools/builtin/free_tool_result.py` | Collapse one or more past tool results to save context (recoverable via `session.state.freed_tool_results`) |
| `skill_load` | `neoagent/tools/builtin/skill_load.py` | Progressive disclosure — load a skill's full content by name (kebab). Supports flat / dir / learned/ layouts. |
| `write_skill` | `neoagent/tools/builtin/write_skill.py` | Persist a discovered extraction pattern as a learned skill under `learned/` |

> All builtin tools have `permission='auto'` — LLM can call without user confirmation. Override at agent level if needed.

---

## Scenario Quick Reference

| # | Scenario | Core modules | Key API / config fields |
|---|----------|-------------|------------------------|
| 1 | Minimal agent | NeoAgent + NeoAgentConfig | `api_key`, `model`, `system_prompt`, `max_turns`, `context_budget` |
| 2 | Custom tool | BaseTool + register_tool() | Subclass `BaseTool`; implement `async execute()`; set `permission`; call `agent.register_tool(tool)` |
| 3 | Cross-session memory | MemoryManager | `memory_dir=Path(...)`, `memory_project_key="..."` in config |
| 4 | Dynamic skill (prompt injection) | PromptBuilder | `agent._prompt_builder.register_skill(name, PromptSection(...))` → `activate_skill(name)` |
| 5 | Event hook + interceptor | HookManager + EventBus | `agent.hook(type, handler)` or `@agent.on(type)`; `agent.event_bus.subscribe(EventClass, handler)` |
| 6 | MCP tools | mcp/ + DeferredToolRegistry | `await agent.add_mcp_server(name, command, env)` → tools auto-registered; LLM uses `tool_search` to promote |
| 7 | Multi-agent orchestration | Orchestrator + WorkerCard | `Orchestrator(config, max_depth=2)`; `register_worker(WorkerCard(...))`; `await orchestrator.run(msg)` |
| 8 | HTTP API server | FastAPIChannel | `FastAPIChannel(agent, host, port)` → `await channel.serve_forever()` |
| 9 | Session recovery (resume) | Session + JsonFileStorage | `session_dir=Path(...)` in config; `session = agent.resume(session_id)` |
| 10 | Observability + eval | Observer + ObserverSubscriber + EvalRunner | `Observer(console=True)`; `ObserverSubscriber(obs).attach(agent.event_bus)`; `EvalRunner(agent).run(cases)` |

---

## Code Skeletons

> **Production-ready standard:** every skeleton below is wired correctly and safe to use as a starting point. Read the Anti-patterns section before adding your own code.

### Scenario 1 — Minimal agent

```python
import asyncio
import os
from neoagent import NeoAgent, NeoAgentConfig

config = NeoAgentConfig(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-6",
    system_prompt="You are a helpful assistant.",
    max_turns=30,
    # Set context_budget explicitly — default 0 means no limit, which can cause
    # runaway context growth in long sessions.
    context_budget=80_000,
)
agent = NeoAgent(config)

async def main() -> None:
    reply: str = await agent.chat("Hello")
    print(reply)

asyncio.run(main())
```

### Scenario 2 — Custom tool

```python
import asyncio
import os
from pydantic import BaseModel
from neoagent import NeoAgent, NeoAgentConfig
from neoagent.tools.base import BaseTool, ToolResult

class SearchInput(BaseModel):
    query: str
    max_results: int = 5

class SearchTool(BaseTool):
    name = "search"
    # description is a prompt for the LLM — be precise and unambiguous
    description = "Search a knowledge base and return up to max_results relevant passages."
    input_model = SearchInput
    # "auto" = LLM can call without asking user. Safe for pure in-memory reads.
    # If your real tool makes network calls or mutates data, use "ask" instead.
    permission = "auto"
    is_concurrent_safe = True

    async def execute(self, input: SearchInput) -> ToolResult:
        results = [f"Result {i} for '{input.query}'" for i in range(input.max_results)]
        return ToolResult(call_id="", output="\n".join(results))

async def main() -> None:
    config = NeoAgentConfig(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
        system_prompt="You are a research assistant with access to a search tool.",
        max_turns=20,
        context_budget=60_000,
    )
    agent = NeoAgent(config)
    agent.register_tool(SearchTool())
    reply = await agent.chat("Search for Python async patterns")
    print(reply)

asyncio.run(main())
```

### Scenario 3 — Cross-session memory

```python
import asyncio
import os
from pathlib import Path
from neoagent import NeoAgent, NeoAgentConfig

# memory_dir + memory_project_key together enable MemoryManager.
# The agent extracts memories after each turn and retrieves relevant
# ones at the start of subsequent sessions — no manual calls needed.
config = NeoAgentConfig(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-6",
    system_prompt="You are a personal assistant who remembers user preferences.",
    max_turns=30,
    context_budget=80_000,
    memory_dir=Path("~/.neoagent/memory").expanduser(),
    # memory_project_key namespaces memories so different projects don't bleed.
    memory_project_key="my-assistant",
)
agent = NeoAgent(config)

async def main() -> None:
    await agent.chat("My preferred language is Python and I hate boilerplate.")

asyncio.run(main())
```

### Scenario 4 — Dynamic skill (prompt injection)

```python
import asyncio
import os
from neoagent import NeoAgent, NeoAgentConfig
from neoagent.core.prompt import PromptSection

CODING_SKILL_CONTENT = """\
You are an expert Python developer. When writing code:
- Use type annotations on all function signatures.
- Prefer dataclasses / Pydantic over plain dicts.
- Write async code unless sync is explicitly required.
- Add docstrings on public functions.
"""

async def main() -> None:
    config = NeoAgentConfig(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
        system_prompt="You are a general assistant.",
        max_turns=20,
        context_budget=60_000,
    )
    agent = NeoAgent(config)

    # Register once — does NOT inject into prompt yet
    # Note: _prompt_builder is private (v0.1.0); no public wrapper exists yet
    agent._prompt_builder.register_skill(
        "coding",
        PromptSection(name="coding", content=CODING_SKILL_CONTENT),
    )

    # Activate only when user switches to a coding task
    agent._prompt_builder.activate_skill("coding")
    reply = await agent.chat("Write a function that fetches JSON from a URL.")
    print(reply)

    # Deactivate when switching back to general mode
    agent._prompt_builder.deactivate_skill("coding")

asyncio.run(main())
```

### Scenario 5 — Event hook + interceptor

```python
import asyncio
import os
from neoagent import NeoAgent, NeoAgentConfig
from neoagent.hooks import HookResult
from neoagent.events import ToolCallEvent, TurnCompleteEvent

async def main() -> None:
    config = NeoAgentConfig(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
        system_prompt="You are an assistant with tool access.",
        max_turns=20,
        context_budget=60_000,
    )
    agent = NeoAgent(config)

    # ── Interceptor hook: runs BEFORE tool execution, can block the call ──
    @agent.on("pre_tool_call")
    async def guard_tool(event) -> HookResult:
        if event.tool_name.startswith("write_"):
            return HookResult.block(reason="Write tools require explicit user approval")
        return HookResult.allow()

    # ── Observation: subscribe to EventBus for passive monitoring ──
    def on_tool_call(event: ToolCallEvent) -> None:
        print(f"[tool] {event.tool_name}")

    def on_turn_complete(event: TurnCompleteEvent) -> None:
        print(f"[turn] usage={event.usage}")

    agent.event_bus.subscribe(ToolCallEvent, on_tool_call)
    agent.event_bus.subscribe(TurnCompleteEvent, on_turn_complete)

    try:
        reply = await agent.chat("Try to write something to disk")
        print(reply)
    finally:
        agent.event_bus.unsubscribe(ToolCallEvent, on_tool_call)
        agent.event_bus.unsubscribe(TurnCompleteEvent, on_turn_complete)

asyncio.run(main())
```

### Scenario 6 — MCP tools

```python
import asyncio
import os
from neoagent import NeoAgent, NeoAgentConfig

async def main() -> None:
    config = NeoAgentConfig(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
        system_prompt="You are an assistant with access to filesystem tools.",
        max_turns=30,
        context_budget=80_000,
    )
    agent = NeoAgent(config)

    # add_mcp_server() connects once and keeps the connection alive.
    # Tools are injected into DeferredToolRegistry (hidden from LLM by default).
    # The built-in tool_search promotes matching tools on demand.
    # Do NOT call add_mcp_server() inside a loop — it spawns a subprocess.
    await agent.add_mcp_server(
        name="filesystem",
        command=["npx", "@anthropic/mcp-server-filesystem", "/tmp"],
        env={"MCP_LOG_LEVEL": "error"},
    )

    reply = await agent.chat("List files in /tmp and summarize what you find.")
    print(reply)

    await agent.remove_mcp_server("filesystem")

asyncio.run(main())
```

### Scenario 7 — Multi-agent orchestration

```python
import asyncio
import os
from neoagent import NeoAgentConfig
from neoagent.multi import Orchestrator, WorkerCard

async def main() -> None:
    config = NeoAgentConfig(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
        system_prompt="You are an orchestrator that delegates tasks to specialists.",
        max_turns=10,
        context_budget=60_000,
    )

    orchestrator = Orchestrator(
        config,
        max_depth=2,
        max_concurrent_workers=5,
    )

    orchestrator.register_worker(WorkerCard(
        name="researcher",
        description="Searches for information and summarizes findings.",
        instruction="You are a research specialist. Find accurate, cited information.",
        tags=("research", "web", "summarize"),
        tools=("bash",),
    ))

    orchestrator.register_worker(WorkerCard(
        name="writer",
        description="Writes polished prose given research notes.",
        instruction="You are a technical writer. Write clearly and concisely.",
        tags=("writing", "editing"),
        tools=(),
    ))

    result: str = await orchestrator.run(
        "Research Python async best practices and write a 3-paragraph summary."
    )
    print(result)

    # Always close — releases worker agent resources
    await orchestrator.close()

asyncio.run(main())
```

### Scenario 8 — HTTP API server

```python
import asyncio
import os
from neoagent import NeoAgent, NeoAgentConfig
from neoagent.channels import FastAPIChannel

async def main() -> None:
    config = NeoAgentConfig(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
        system_prompt="You are a helpful HTTP-accessible assistant.",
        max_turns=30,
        context_budget=80_000,
    )
    agent = NeoAgent(config)

    # Host and port are passed to the constructor, NOT to serve_forever().
    # serve_forever() blocks until interrupted.
    # Endpoints: POST /v1/run (sync), POST /v1/run/stream (SSE)
    channel = FastAPIChannel(
        agent,
        host="127.0.0.1",
        port=8000,
        streaming=True,
    )
    await channel.serve_forever()

asyncio.run(main())
```

### Scenario 9 — Session recovery (resume)

```python
import asyncio
import os
from pathlib import Path
from neoagent import NeoAgent, NeoAgentConfig
from neoagent.session import Session

async def first_session() -> str:
    """Start a new session and return its ID for resumption."""
    config = NeoAgentConfig(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
        system_prompt="You are a coding assistant.",
        max_turns=30,
        context_budget=80_000,
        session_dir=Path("~/.neoagent/sessions").expanduser(),
    )
    agent = NeoAgent(config)
    session: Session = agent.new_session()
    await agent.chat("Start writing a Python web scraper.", session=session)
    return session.id

async def resume_session(session_id: str) -> None:
    """Resume a session from a previous run."""
    config = NeoAgentConfig(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
        system_prompt="You are a coding assistant.",
        max_turns=30,
        context_budget=80_000,
        session_dir=Path("~/.neoagent/sessions").expanduser(),
    )
    agent = NeoAgent(config)
    # agent.resume() uses the configured JsonFileStorage
    session: Session = agent.resume(session_id)
    await agent.chat("Continue — add error handling and retry logic.", session=session)

async def main() -> None:
    sid = await first_session()
    print(f"Session saved: {sid}")
    await resume_session(sid)

asyncio.run(main())
```

### Scenario 10 — Observability + eval

```python
import asyncio
import os
from pathlib import Path
from neoagent import NeoAgent, NeoAgentConfig
from neoagent.observe import Observer
from neoagent.observe_subscriber import ObserverSubscriber
from neoagent.eval.runner import EvalRunner, EvalCase
from neoagent.core.types import Message

async def main() -> None:
    config = NeoAgentConfig(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
        system_prompt="You are a helpful assistant.",
        max_turns=20,
        context_budget=60_000,
    )
    agent = NeoAgent(config)

    observer = Observer(console=True, log_dir=Path("logs/"))
    subscriber = ObserverSubscriber(observer)
    subscriber.attach(agent.event_bus)

    try:
        cases = [
            EvalCase(
                messages=[Message(role="user", content="What is Python?")],
                assertion=lambda r: "python" in r.lower(),
            ),
            EvalCase(
                messages=[Message(role="user", content="Explain asyncio in one sentence.")],
                assertion=lambda r: "async" in r.lower() or "coroutine" in r.lower(),
            ),
        ]
        runner = EvalRunner(agent)
        report = await runner.run(cases)
        print(f"Passed: {report.passed}/{report.total} ({report.pass_rate:.0%})")
    finally:
        subscriber.detach(agent.event_bus)
        observer.close()

asyncio.run(main())
```

---

## CLI Tooling (examples/)

### `examples/v2_repl.py` — interactive terminal REPL

```bash
# New session
python examples/v2_repl.py

# List resumable sessions
python examples/v2_repl.py --list

# Resume an existing session
python examples/v2_repl.py --resume <SESSION_ID>
```

- Logs every LLM request to `logs/<session_id>/turnNNN_reqMM.{txt,json}`
- Persists session state to `sessions/<session_id>.json` via `JsonFileStorage`

### `examples/migrate_log_to_session.py` — replay events log to rebuild a Session JSON

```bash
python examples/migrate_log_to_session.py logs/<id>/events.jsonl
```

Replays a `logs/<id>/events.jsonl` file to rebuild a `Session` JSON for resume. Useful when an older session was logged before `session_dir` was wired.

---

## Anti-patterns

These are the four most common mistakes when generating neoagent code.
**Check every generated code block against these before output.**

### 1 — Loop IO: creating connections inside a loop

```python
# BAD — creates a new MCP subprocess for every item. Resource leak + bottleneck.
for query in queries:
    await agent.add_mcp_server("search", ["python", "search_server.py"])
    result = await agent.chat(query)
    await agent.remove_mcp_server("search")

# GOOD — connect once, reuse for all calls. Run independent queries concurrently.
await agent.add_mcp_server("search", ["python", "search_server.py"])
results = await asyncio.gather(*[agent.chat(q) for q in queries])
await agent.remove_mcp_server("search")
```

### 2 — Permission abuse: `"auto"` on write/delete operations

```python
# BAD — "auto" lets LLM execute without asking the user. Dangerous for mutations.
class DeleteFileTool(BaseTool):
    permission = "auto"   # WRONG: destructive op must require confirmation

# GOOD — write/delete/network-mutating operations always use "ask".
class DeleteFileTool(BaseTool):
    permission = "ask"

class ReadFileTool(BaseTool):
    permission = "auto"   # OK: read-only, no side effects
    is_concurrent_safe = True
```

### 3 — Resource leak: forgetting to detach Observer

```python
# BAD — log file handle never closed; EventBus holds stale handler references.
observer = Observer(log_dir=Path("logs/"))
subscriber = ObserverSubscriber(observer)
subscriber.attach(agent.event_bus)
result = await agent.chat(message)
# leaked file handle + stale subscription

# GOOD — always use try/finally.
observer = Observer(log_dir=Path("logs/"))
subscriber = ObserverSubscriber(observer)
subscriber.attach(agent.event_bus)
try:
    result = await agent.chat(message)
finally:
    subscriber.detach(agent.event_bus)
    observer.close()
```

### 4 — Concurrent safety lie: `is_concurrent_safe=True` on stateful tools

```python
# BAD — self._conn is shared; concurrent calls will race.
class DatabaseTool(BaseTool):
    is_concurrent_safe = True  # WRONG: shared mutable state

    def __init__(self):
        self._conn = create_db_connection()

    async def execute(self, input):
        return ToolResult(call_id="", output=await self._conn.query(input.sql))

# GOOD — inject a shared pool; pool handles concurrency internally.
# Pool is created ONCE outside the tool and passed in (dependency injection).
class DatabaseTool(BaseTool):
    is_concurrent_safe = False  # False: query ordering may matter for callers

    def __init__(self, pool: "AsyncConnectionPool"):
        self._pool = pool  # injected at construction, never re-created

    async def execute(self, input):
        async with self._pool.acquire() as conn:
            result = await conn.query(input.sql)
        return ToolResult(call_id="", output=result)
```

---

## Production-Ready Checklist

Before submitting any neoagent code, verify every item. Unchecked = not production-ready.

### Correctness
- [ ] `BaseTool.permission` is `"ask"` for write, delete, or network-mutating operations. Only pure read-only, side-effect-free tools use `"auto"`.
- [ ] `is_concurrent_safe=True` only when tool has no shared mutable state and no conflicting side effects.
- [ ] `context_budget` is set explicitly (not left at default `0`). Typical range: `60_000`–`120_000`.
- [ ] All `async def` methods use `await` consistently. No sync blocking IO inside `async def`.
- [ ] API keys come from `os.environ`, never hardcoded.

### Performance and resources
- [ ] MCP servers added once per agent lifetime — not inside loops.
- [ ] Independent tasks use `asyncio.gather()` not sequential `await` in a loop.
- [ ] `Observer` + `ObserverSubscriber` cleaned up with `try/finally`.
- [ ] `Orchestrator` closed with `await orchestrator.close()` after use.
- [ ] `FastAPIChannel` uses `serve_forever()` — `channel.run()` does not exist.

### Architecture
- [ ] Observability implemented via EventBus subscriptions, not inline print/log.
- [ ] `BaseTool` subclasses do one thing. Side effects documented in `description`.
- [ ] Config, storage, tools injected at construction — not hardcoded inside classes.

### Code quality
- [ ] All public method signatures have type annotations.
- [ ] `BaseTool.description` is an LLM-facing prompt: precise, unambiguous.
- [ ] Comments explain **why**, not **what**.
- [ ] Private attributes (`_event_bus`, `_prompt_builder`) accessed only when no public API exists; add a comment noting they are private.

### Known private API surface (v0.1.0)

| Private attribute | Use case | Public alternative |
|-------------------|----------|--------------------|
| `agent._prompt_builder` | Register/activate dynamic skills | None yet (planned) |
| `agent._hook_manager` | Direct HookManager access | `agent.hook()` / `@agent.on()` |

> `agent._event_bus` no longer needed — use the public `agent.event_bus` property directly.

---

## Deep Reference

| Need | Where to look |
|------|--------------|
| Full API documentation | `<neoagent-repo>/docs/guide.md` (use your local install path) |
| Source: core loop, prompt, compress | `neoagent/core/` |
| Source: tools, BaseTool, registry | `neoagent/tools/` |
| Source: multi-agent | `neoagent/multi/` |
| Source: MCP client + transport | `neoagent/mcp/` |
| Source: channels (FastAPIChannel) | `neoagent/channels/` |
| Source: events, hooks | `neoagent/events.py`, `neoagent/hooks.py` |
| Source: session, storage | `neoagent/session.py` |
| Source: memory system | `neoagent/memory/` |
| Source: observer, eval | `neoagent/observe.py`, `neoagent/observe_subscriber.py`, `neoagent/eval/` |
| Architecture principles (why) | Use the `ai-knowledge` skill to navigate `wiki/` |
| Pattern cookbook | Use the `ai-knowledge` skill to navigate `cookbook/` |
| Install | `pip install -e /path/to/neoagent` (v0.1.0, not on PyPI) |
