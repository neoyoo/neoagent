# neoagent

> A Python async agent SDK built around pluggable abstractions and a single session as the source of truth.

[English] | [中文](README.zh-CN.md)

## What it is

neoagent is a production-oriented Python SDK for building AI agents. It handles the agent loop, tool dispatch, context compression, cross-session memory, hook interception, MCP integration, multi-agent orchestration, and HTTP channel exposure — all wired together through a single `NeoAgent` class with an async-first design.

The SDK supports Anthropic and OpenAI providers interchangeably. The core agent loop is fully `async/await`. Every public API carries strict type annotations, and the test suite runs 577+ tests (TDD-first development).

This is personal infrastructure — not on PyPI. Install directly from the repository. The current active branch is `v2.0`.

## Design Philosophy

**Pluggable abstractions for every storage decision.** Session persistence, working memory, compressed message storage, compression strategy, and memory review strategy are each backed by a Protocol. The default implementations are ready to use out of the box. Swap any of them for a custom backend (Postgres, Elasticsearch, Qdrant, S3) by implementing one Protocol — no other code changes needed.

**SessionState is the single source of truth.** Everything the agent needs to resume, introspect, or hand off a conversation lives in `SessionState`. No state scattered across singletons or module-level globals. This makes session persistence, crash recovery, and transparent resumption natural rather than bolted on.

**Event-driven by default.** Every meaningful action — tool calls, turn completions, compression events, working memory mutations — emits a typed Event via the `EventBus`. Observers attach without touching core code. Hook handlers intercept and can block tool calls before execution. The two systems (hooks and events) are composable and independent.

**Double-trigger compression keeps context budgets honest.** The `ContextCompressor` fires on two independent signals: every 10 user turns, or when token usage crosses 70% of `context_budget`. When compression runs, original message bodies are written to `CompressedMessageStore` indexed by `msg_id`. The built-in `recall_turn` tool lets the agent retrieve full originals on demand — no information is permanently discarded.

**Deferred tool registry prevents context bloat.** Tools can live in a `DeferredToolRegistry` (invisible to the LLM) and be promoted into the active registry on demand via the built-in `tool_search` tool. Large MCP tool sets attach this way by default — the LLM only sees what it needs when it needs it.

## Quick Start

Install from the local repository checkout:

```bash
pip install -e /path/to/neoagent
```

Minimal agent:

```python
import asyncio
import os
from neoagent import NeoAgent, NeoAgentConfig

config = NeoAgentConfig(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-6",
    system_prompt="You are a helpful assistant.",
    max_turns=30,
    context_budget=80_000,
)
agent = NeoAgent(config)

async def main() -> None:
    reply: str = await agent.chat("Hello")
    print(reply)

asyncio.run(main())
```

Two entry points: `agent.chat(message)` returns a `str` for single-turn use. `agent.run(messages, max_turns=N)` returns a `ConversationResult` for full conversation control.

## Architecture

The SDK is organized into 12 modules. Abbreviated layout:

```
NeoAgent (agent.py)
  ├── QueryLoop (core/loop.py)         — execution engine, state machine
  │     ├── Provider (providers/)      — Anthropic / OpenAI adapters
  │     ├── ToolExecutor               — ToolRegistry + DeferredToolRegistry
  │     ├── HookManager (hooks.py)     — pre/post interception
  │     └── ContextCompressor          — auto-compress on turn count or token threshold
  ├── MemoryManager (memory/)          — cross-session memory extract + retrieve
  ├── Session + SessionState (session.py) — persistent sessions, resumption
  ├── EventBus (events.py)             — publish/subscribe observability bus
  └── Observer (observe.py)            — structured logging + eval

Independent extension modules:
  mcp/          — MCPClient, StdioTransport, MCPTool
  multi/        — Orchestrator, WorkerCard
  channels/     — FastAPIChannel (POST /v1/run, /v1/run/stream)
  eval/         — EvalRunner, EvalCase, EvalReport

v2 internals (neoagent/v2/):
  schema.py           — WorkingMemory, Batch, CompressionDelta, Layer
  compressed_store.py — CompressedMessageStore Protocol + InMemory default
  abc.py              — WorkingMemoryStore Protocol
  strategies/         — OneShotCompressionStrategy, OneShotMemoryReviewStrategy
```

See `skills/neoagent/SKILL.md` for the full module diagram, all 10 scenario skeletons, and the production-ready checklist.

## Pluggable Extensions

All five extension points follow the same pattern: implement the Protocol, pass the instance via `NeoAgentConfig`.

| Protocol | Config field | Default impl |
|---|---|---|
| `SessionStorage` | `session_dir` | `JsonFileStorage` |
| `WorkingMemoryStore` | `working_memory_store` | `InMemoryWorkingMemoryStore` |
| `CompressedMessageStore` | `compressed_message_store` | `InMemoryCompressedMessageStore` |
| `CompressionStrategy` | `compression_strategy` | `OneShotCompressionStrategy` |
| `MemoryReviewStrategy` | `memory_review_strategy` | `OneShotMemoryReviewStrategy` |

`CompressedMessageStore` is the v2.0 addition. It decouples where compressed message bodies live from the session record itself. The default wraps `session_state.compressed_messages` so `JsonFileStorage` continues to round-trip everything in one file. Point it at Postgres or Elasticsearch to store bodies independently at scale.

## Examples

**`examples/v2_repl.py`** — interactive terminal REPL. Exercises the full v2.0 feature set: `PromptBuilder`, working memory snapshots, compression, free/recall, custom tools, and per-turn request logging. Supports `--list` and `--resume <SESSION_ID>`.

**`examples/migrate_log_to_session.py`** — replays a `logs/<id>/events.jsonl` file to rebuild a `Session` JSON. Useful when older sessions were captured before `session_dir` was configured.

## Companion Skill

`skills/neoagent/SKILL.md` is the canonical detailed guide for building with this SDK. It contains the full module diagram, all scenario skeletons (minimal agent through eval harness), the anti-patterns reference, and the production-ready checklist. Symlink or copy it to `~/.claude/skills/` to enable it in Claude Code.

```bash
ln -s /path/to/neoagent/skills/neoagent ~/.claude/skills/neoagent
```

## Acknowledgments

This SDK stands on a lot of prior art:

- **[Claude Code](https://claude.com/claude-code)** (Anthropic) — the architectural reference. Hooks, skills + MCP integration, deferred tool registry, working-memory snapshots, and the overall agent-loop discipline all originate from studying Claude Code's design.
- **Andrej Karpathy** — the framing of LLMs as a new compute surface ("Software 3.0," LLM-as-OS), and the [vibe-coding thread](https://x.com/karpathy/status/2015883857489522876) that crystallized into [Forrest Chang's andrej-karpathy-skills CLAUDE.md](https://github.com/forrestchang/andrej-karpathy-skills). The four principles — *Think before coding · Simplicity first · Surgical changes · Goal-driven execution* — were adopted directly into our development workflow; their fingerprints are visible across v2.
- **[MemGPT / Letta](https://github.com/letta-ai/letta)** — validated the layered "working memory + archival memory" shape that `WorkingMemory` and `compressed_history` inherit.
- **The wider AI agent community** (AgentScope, DeerFlow, and others) — catalogued comparatively in the sister [ai-knowledge](https://github.com/neoyoo/ai-knowledge) vault. We read them to decide what to borrow and where to differ.

Nothing here is novel. The opinionated part is the assembly.

## License

MIT. See `LICENSE`.

## Status

Personal infrastructure project. APIs may evolve. `v2.0` is the current active branch. Not on PyPI — install with `pip install -e /path/to/neoagent`.
