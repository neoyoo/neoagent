# neoagent — Cross-Review Context Brief

**Purpose**: Complete context for a Codex CLI cross-review. The reviewer has no prior context — everything needed to understand the project is here.

**Date**: 2026-04-11  
**Status**: v2 complete, 265 tests passing, awaiting real API validation before v3 planning

---

## 1. Project Overview

**What**: neoagent is a Python agent framework — a self-contained library for building LLM-powered agents that execute tools in a loop. Think "Claude Code SDK but self-controlled."

**Why**: The author (Neo) wanted a framework he fully owns and controls, informed by deep source-code analysis of 5 existing frameworks (Claude Code, OpenHarness, DeerFlow, Hermes Agent, mempalace). Rather than contributing to or depending on those frameworks, this is a clean-room implementation that selectively absorbs the best design decisions from each.

**Who uses it**: Internal use only. No public API stability guarantees yet.

**Philosophy**:
- Claude Code skeleton (loop structure, tool permission model, compression strategy)
- OpenHarness Pydantic input model pattern (one model for validation AND schema generation)
- DeerFlow compression trigger (70% threshold) and skill lazy-loading concept
- Hermes iterative summary compression (incremental `_previous_summary` updates)
- mempalace verbatim storage approach (raw markdown files, not a database)

**Language/Runtime**: Python 3.11+, fully async (asyncio), Pydantic v2, tiktoken, anthropic SDK, openai SDK

---

## 2. Architecture

### Layer Diagram

```
┌─────────────────────────────────────────────────────┐
│                   NeoAgent (agent.py)               │
│  Entry point — wires config → provider → registry  │
│  → prompt_builder → loop; exposes chat() / run()   │
└────────────────┬────────────────────────────────────┘
                 │
    ┌────────────▼────────────┐
    │     core/ (runtime)     │
    │  loop.py   — QueryLoop  │
    │  prompt.py — PromptBuilder, PromptSection  │
    │  compress.py — ContextCompressor           │
    │  types.py  — Message, Turn, ToolCall, etc. │
    └────┬──────────┬─────────┘
         │          │
┌────────▼──┐  ┌────▼──────────────────────────────┐
│providers/ │  │          tools/                    │
│base.py    │  │  base.py       — BaseTool ABC      │
│anthropic  │  │  registry.py   — ToolRegistry      │
│openai     │  │  permission.py — PermissionChecker │
└────────────┘  │  pathguard.py  — path validation  │
                │  builtin/                         │
                │    read, write, edit, bash,       │
                │    grep, glob                     │
                └───────────────────────────────────┘
         │
┌────────▼───────────────────────────────────┐
│              memory/ (v2)                  │
│  store.py     — MemoryStore (files)        │
│  retriever.py — MemoryRetriever (lexical)  │
│  extractor.py — MemoryExtractor (LLM)      │
│  manager.py   — MemoryManager (orchestrator) │
└────────────────────────────────────────────┘
         │
┌────────▼───────────────────────┐
│       observe.py (v2)          │
│  Observer — dev logging hooks  │
└────────────────────────────────┘
```

### File Map

```
neoagent/
├── agent.py              Entry point. NeoAgent class. Wires all components.
│                         Exposes: chat(str)->str, run(messages)->ConversationResult,
│                         enable_memory(), enable_logging(), register_tool()
│
├── config.py             NeoAgentConfig dataclass. Fields: api_key, model, provider,
│                         base_url, max_turns(30), context_budget(0=auto),
│                         max_result_size(50000), memory_dir, memory_project_key
│
├── observe.py            Observer class. Hooks for provider request/response,
│                         tool call/result, compression check/done, memory events,
│                         skill activate/deactivate. Writes to console + timestamped
│                         log file. Dual output: truncated to console, full to file.
│
├── core/
│   ├── types.py          All wire types. TextBlock, ToolUseBlock, ToolResultBlock
│   │                     (all Pydantic BaseModel). Message, Turn, ConversationResult
│   │                     (Pydantic). ToolCall, ToolResult (dataclass).
│   │                     ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock
│   │
│   ├── loop.py           QueryLoop. The agent execution engine.
│   │                     run(messages) -> ConversationResult
│   │                     Handles: token budget check, compression, prompt build,
│   │                     provider call, max_tokens retry, tool dispatch, memory
│   │                     extraction trigger, observer hooks. max_turns hard limit.
│   │
│   ├── prompt.py         PromptBuilder + PromptSection.
│   │                     Sections have name, content(str|Callable), priority, is_static.
│   │                     build() sorts: static first (by priority), then dynamic.
│   │                     Skill lazy loading: register_skill(), activate_skill(),
│   │                     deactivate_skill(), is_skill_active()
│   │
│   └── compress.py       ContextCompressor. Iterative LLM summarization (v2).
│                         should_compress() checks (msg_tokens + tool_tokens) > budget*0.7
│                         compress(): LLM path (anchor + summary + recent 6) with circuit
│                         breaker (3 failures → fallback truncation). _sanitize_tool_pairs()
│                         removes orphaned tool_use/tool_result blocks. Role alternation fix.
│
├── tools/
│   ├── base.py           BaseTool ABC. Declares: name, description, input_model,
│   │                     permission("ask"), is_concurrent_safe(False).
│   │                     get_schema() produces Anthropic-format tool schema from input_model.
│   │
│   ├── registry.py       ToolRegistry. register(), get_schemas() (skips "deny" tools),
│   │                     execute(calls) with concurrency partitioning.
│   │                     Result truncation at max_result_size (default 50000 chars).
│   │                     Validation via input_model.model_validate() before execute.
│   │
│   ├── permission.py     PermissionChecker. auto→True, deny→False, ask→callback or
│   │                     auto_approve flag. Default: auto_approve=True (with warning).
│   │                     Supports async ask_callback: (name, description, input_dict)->bool
│   │
│   ├── pathguard.py      validate_path(file_path, allowed_directories) -> Path.
│   │                     Resolves symlinks via Path.resolve(). Raises ValueError if
│   │                     path not within any allowed directory.
│   │
│   └── builtin/
│       ├── read.py       ReadTool. permission="auto", concurrent_safe=True.
│       │                 Input: file_path, offset(0), limit(2000).
│       │                 Returns line-numbered output. Uses pathguard.
│       │
│       ├── write.py      WriteTool. permission="ask", not concurrent_safe.
│       │                 Creates parent directories. Uses pathguard.
│       │
│       ├── edit.py       EditTool. permission="ask", not concurrent_safe.
│       │                 Exact-match replace: fails if 0 or 2+ matches.
│       │
│       ├── bash.py       BashTool. permission="ask", not concurrent_safe.
│       │                 Default 120s timeout. Regex blocklist (rm -rf, chmod 777,
│       │                 mkfs, dd to /dev, curl|sh, etc.). asyncio subprocess.
│       │
│       ├── grep.py       GrepTool. permission="auto", concurrent_safe=True.
│       │                 Shells out to system grep -r -n -E. Uses pathguard.
│       │
│       └── glob.py       GlobTool. permission="auto", concurrent_safe=True.
│                         base.glob(pattern). Uses pathguard on base path only.
│
├── providers/
│   ├── base.py           Provider ABC. create(system, messages, tools, **kwargs)->Response.
│   │                     get_context_window()->int. Response dataclass: content, stop_reason,
│   │                     input_tokens, output_tokens. Properties: tool_use_blocks, text_content.
│   │
│   ├── anthropic.py      AnthropicProvider. Default model: claude-sonnet-4-20250514.
│   │                     Context window: hardcoded 200_000. Serializes Message objects
│   │                     to Anthropic API format. Parses text+tool_use blocks from response.
│   │                     Supports base_url for custom endpoints.
│   │
│   └── openai.py         OpenAIProvider. Default model: gpt-4o. Context window from
│                         lookup table (gpt-4o/mini=128k, o1/o3/o4-mini=200k, fallback=128k).
│                         Converts Anthropic-format schemas to OpenAI function calling format.
│                         Maps stop reasons: stop→end_turn, tool_calls→tool_use, length→max_tokens.
│
└── memory/
    ├── store.py          MemoryStore. File-based. Layout: memory_dir/MEMORY.md (index)
    │                     + {topic}.md files. Index format: markdown link list
    │                     "- [description](filename)". Methods: read/write_index,
    │                     read/write/delete_topic, list_topics().
    │
    ├── retriever.py      MemoryRetriever. Lexical keyword scoring over topic descriptions.
    │                     retrieve(query=None): if query=None returns index only.
    │                     With query: scores by word overlap, returns top-5 matched files
    │                     (max 2000 chars each). Falls back to all topics if no matches.
    │
    ├── extractor.py      MemoryExtractor. LLM-based. Triggers if tool_calls >= 5 OR
    │                     token_delta >= 4000. Sends last 8000 chars of conversation.
    │                     LLM returns JSON array: [{filename, description, content}].
    │                     Filename sanitized: must match ^[a-zA-Z0-9_\-]+\.md$, no path traversal.
    │                     Rebuilds MEMORY.md index from directory scan after storing.
    │
    └── manager.py        MemoryManager. Orchestrates store+retriever+extractor.
                          record_tool_calls(n), maybe_extract(messages, tokens).
                          Token baseline on first call; resets after each extraction.
                          build_prompt_section(query=None) -> str for PromptBuilder injection.
```

---

## 3. Version Status

### v1 (complete — commits 83aa259 through 92df1a0)
- QueryLoop with max_turns, max_tokens retry, tool dispatch
- ToolSystem: BaseTool, ToolRegistry, PermissionChecker, 6 builtin tools
- PromptSystem: PromptBuilder, PromptSection, static/dynamic boundary
- ContextCompressor: LLM summarization (basic, no iterative summary)
- AnthropicProvider only
- Integration test suite

### v2 (complete — commits 3b6589a through 8014f52)
Additions over v1:
- **Memory System**: MemoryStore + MemoryRetriever + MemoryExtractor + MemoryManager
  wired into QueryLoop and NeoAgent.enable_memory()
- **Iterative compression** (Hermes B2): `_previous_summary` carries forward context
  across multiple compressions; structured GOAL/PROGRESS/DECISIONS/FILES/NEXT STEPS/KEY CONTEXT template
- **Skill lazy loading** in PromptBuilder: register_skill() / activate_skill() / deactivate_skill()
- **Observer** framework: `Observer` class with hooks for all major events; dual output (console + file)
- **OpenAI provider** with full tool calling support and Anthropic→OpenAI format conversion
- **base_url** support for OpenAI-compatible proxies
- **Critical bug fixes** (detailed in Section 6):
  - PermissionChecker wired into ToolRegistry
  - BashTool command blocklist
  - File tools path restriction (pathguard + allowed_directories)
  - Provider max_tokens and kwargs handling
  - Role alternation fix after compression
  - Memory filename sanitization
  - Token baseline reset
  - Observer context manager + encapsulation fixes

**Current test count**: 265 tests, all passing (2.68s)  
**No real API tests yet** — all tests use mocks or local file operations.

### v3 (planned, not started)
- Multi-Agent: subagent dispatching, AgentTool
- Hooks / event system for pre/post tool execution
- MCP protocol support
- Channel / interface layer (CLI, web)

---

## 4. Design Decisions (with rationale)

### 4.1 Query Loop — Simple While Loop (KB "Plan A")

**Chosen**: A flat `for turn_idx in range(max_turns)` loop. No state machine, no graph.

**Alternatives considered** (from KB analysis):
- State machine (LangGraph style) — explicit state transitions, good for complex workflows
- DAG execution (crew-style) — pre-defined agent graph
- Event-driven (Hermes) — 9 callback types, very observable but complex

**Why chosen**: For programming assistant / CLI use cases, a simple loop is optimal. The additional complexity of state machines only pays off when you need branching workflows defined at configuration time. The loop's simplicity means the entire execution path is readable in ~80 lines.

**Trade-offs**:
- Cannot express parallel agent workflows (v3 problem)
- No visual "execution graph" for debugging
- max_turns is the only termination mechanism besides end_turn

### 4.2 max_tokens Handling — Retry with Higher Limit

**Chosen**: First attempt at 8192 tokens. On `max_tokens` stop reason, retry once at 16384. If still `max_tokens`, treat response as end_turn (no corrupt tool calls).

**Why**: Claude Code pattern. Avoids failing hard on responses that are legitimately long. The fallback to end_turn is safer than trying to parse a truncated tool_use block.

**Trade-offs**: Two API calls in the worst case. The fallback loses the truncated output entirely — the agent just stops.

**Code**: `loop.py:54-64`, `_DEFAULT_MAX_TOKENS = 8192`, `_MAX_RETRY_TOKENS = 16384`

### 4.3 Tool System — Pydantic input_model (OpenHarness pattern)

**Chosen**: Each tool declares `input_model: type[BaseModel]`. This single class serves both purposes: (1) validate incoming JSON from the LLM, (2) generate the JSON schema passed to the API via `model_json_schema()`.

**Alternatives**: Separate schema dict + manual validation (Claude Code's JS approach), or decorator-based schema generation (some Python frameworks).

**Why**: Pydantic v2 provides rigorous validation and schema generation from the same source. Eliminates the drift risk between "what the LLM thinks the tool accepts" and "what the code actually validates." OpenHarness showed this pattern works cleanly in Python.

**Trade-offs**: Tools must use Pydantic models; cannot use plain dicts or dataclasses for inputs.

### 4.4 Concurrency — fail-closed (Claude Code pattern)

**Chosen**: `is_concurrent_safe: bool = False` default. Only tools that explicitly opt in run concurrently via `asyncio.gather`. All others run serially.

**Why**: Safety by default. A tool author who does not think about concurrency safety gets serial execution. Opt-in for concurrent execution requires the author to reason about it.

**Trade-offs**: Most tool calls are serial even when they could be parallel. BashTool and file-write tools are correctly serial. ReadTool and GrepTool opt in as concurrent_safe.

**Code**: `registry.py:48-57`, `_partition_by_concurrency()`

### 4.5 Compression Strategy — Iterative LLM Summarization (Hermes B2)

**Chosen**: When token estimate exceeds 70% of context budget, compress the "middle" of the conversation (keep anchor=first message and recent=last 6 messages) via LLM call. The summary is incremental: `_previous_summary` is passed to each new compression call so context accumulates rather than resets. Circuit breaker: 3 consecutive failures → fallback to truncation.

**Alternatives**:
- Simple truncation (drop oldest) — fast but loses all context
- Embedding-based retrieval compression — probabilistic, bad for production
- No compression — eventually fails on long tasks

**Why**: For code/task agents, what happened earlier matters (files created, decisions made). Pure truncation loses this. LLM summarization preserves the semantics. The iterative approach (Hermes B2) avoids each compression starting from scratch, so very long tasks accumulate summaries properly.

**Token estimation**: Uses `tiktoken` with `cl100k_base` (GPT-4 tokenizer). This is an approximation for Claude models (typically ±10-20%). Accurate enough for threshold decisions; labeled clearly in code comments.

**Trade-offs**: Compression itself makes one API call, adding latency and cost. Tool schema tokens are included in the estimate (learned from Hermes: 50+ tools = 20-30K extra tokens).

**Code**: `compress.py`, `_KEEP_RECENT = 6`, circuit breaker `max_failures=3`

### 4.6 Memory System — File-based Markdown Store

**Chosen**: `~/.neoagent/memory/{project_key}/` directory with `MEMORY.md` (index) and per-topic `.md` files. Project key defaults to SHA-256 of `cwd()` (first 8 chars).

**Alternatives**:
- SQLite — structured but overkill for small memory sets; harder to inspect/edit manually
- Vector database (FAISS, Chroma) — semantic search but heavyweight dependency
- Single file — simple but doesn't scale to many topics

**Why**: Markdown files are inspectable, editable, and version-controllable by the user. Informed by mempalace's "verbatim raw markdown" design philosophy. No external database dependencies.

**Retrieval**: Lexical keyword scoring, not semantic. Scores topic descriptions by word overlap with query. Falls back to all topics if no keyword matches. Limits: top 5 files, 2000 chars each.

**Extraction trigger**: `tool_calls >= 5 OR token_delta >= 4000`. Only runs at conversation end (`end_turn`). The threshold avoids extracting on trivial single-question sessions.

**Trade-offs**:
- Retrieval is lexical, not semantic — will miss conceptually related topics with different vocabulary
- Extraction is synchronous (not background) — adds latency at conversation end
- No memory deduplication or conflict resolution
- Index rebuild scans directory on every extraction (acceptable at small scale)

### 4.7 Provider Abstraction

**Chosen**: `Provider` ABC with `create(system, messages, tools, **kwargs) -> Response` and `get_context_window() -> int`. Implementations: `AnthropicProvider` and `OpenAIProvider`.

**Why**: The `system` parameter is passed separately (not as a system message) to match the Anthropic API natively. OpenAIProvider handles the conversion internally (prepends as `{"role": "system", ...}`).

**Trade-offs**:
- Anthropic hardcodes context window at 200_000 (correct for claude-3/claude-4 models, but could be wrong for future models)
- OpenAI uses a lookup table + default 128_000 (same issue, model names change)
- Tool schemas are passed in Anthropic format; OpenAI provider converts them

---

## 5. Security Model

### 5.1 Permission System (PermissionChecker)

Three levels, declared per tool via `permission` field:
- `"auto"` — always allowed, no user confirmation (ReadTool, GrepTool, GlobTool)
- `"ask"` — requires user confirmation. Default behavior:
  - If `ask_callback` is set: calls it async, returns user's decision
  - If no `ask_callback` and `auto_approve=True` (default): auto-approves with a warning log
  - If no `ask_callback` and `auto_approve=False`: denies
- `"deny"` — always blocked; tool is also hidden from the model (not in `get_schemas()`)

**Important**: The default `auto_approve=True` means in headless/non-interactive mode, `"ask"` tools (including BashTool, WriteTool, EditTool) are auto-approved. This is by design for programmatic use but must be understood by integrators.

**File**: `tools/permission.py`

### 5.2 BashTool Command Blocklist

Default blocked regex patterns (applied via `re.search` against the command string):
```python
r"\brm\s+(-[a-zA-Z]*f|-[a-zA-Z]*r|--force|--recursive)\b"  # rm -rf, rm -f, rm -r
r"\bchmod\s+777\b"
r"\bmkfs\b"
r"\bdd\s+.*of=/dev/"
r">\s*/dev/sd"
r"\b(curl|wget)\b.*\|\s*(ba)?sh"  # curl|sh, wget|bash
```

**Limitations**:
- Blocklist is regex, not a sandbox. A determined attacker can bypass (e.g., `r''m -rf` with quote tricks, `eval "rm -rf"`, variable expansion).
- No working directory restriction for bash — it can `cd` anywhere and operate outside `allowed_directories`.
- No resource limits (CPU, memory, network, disk writes).
- Timeout is set by the LLM input (default 120s, configurable).

**File**: `tools/builtin/bash.py`

### 5.3 File Tool Path Restriction (pathguard)

`ReadTool`, `WriteTool`, `EditTool`, `GrepTool`, `GlobTool` all call `validate_path(file_path, allowed_directories)` before execution.

`validate_path` in `pathguard.py`:
1. `Path(file_path).resolve()` — resolves symlinks and normalizes
2. Checks `resolved.is_relative_to(allowed.resolve())` for each allowed directory
3. Raises `ValueError` if none match

Default `allowed_directories` for all file tools: `[Path.cwd()]` — the working directory at tool instantiation time.

**Symlink behavior**: `resolve()` follows symlinks, so a symlink pointing outside `allowed_directories` would be resolved to its real path and blocked. This is correct.

**Limitation**: GlobTool validates the base `path` parameter but the resulting matches are not individually validated — they come from `base.glob(pattern)` which is contained within the validated base directory, so this is safe.

**File**: `tools/pathguard.py`

### 5.4 Memory Filename Sanitization

In `extractor.py`, LLM-generated filenames are sanitized before storage:
```python
if "/" in filename or ".." in filename or "\\" in filename:
    continue
if not re.match(r'^[a-zA-Z0-9_\-]+\.md$', filename):
    continue
```

This prevents path traversal attacks where a malicious or hallucinating LLM returns `"filename": "../../../etc/passwd"`.

**File**: `memory/extractor.py:109-113`

---

## 6. Issues Found and Fixed (Full History)

### CRITICAL (fixed before v2 shipped)

**C1: PermissionChecker not wired into ToolRegistry**  
Commit `d8045ba`. `ToolRegistry.execute()` was calling tools without permission checks. `PermissionChecker` existed but was instantiated separately and never consulted. Fix: `ToolRegistry.__init__` now accepts `permission_checker` param; `_run_one()` calls `checker.check(tool, validated_input)` before `tool.execute()`.

**C2: BashTool had no command blocklist**  
Commit `d8045ba`. `BashTool.execute()` ran any command without filtering. Fix: `_DEFAULT_BLOCKED_PATTERNS` regex list applied via `re.search` before subprocess creation.

**C3: File tools had no path restriction**  
Commit `4c38417`. `ReadTool`, `WriteTool`, `EditTool`, `GrepTool`, `GlobTool` accepted arbitrary paths with no validation. Fix: `validate_path()` in `pathguard.py`, called at top of each tool's `execute()`. `allowed_directories` param added to tool constructors.

**C4: Provider `create()` ignored `max_tokens` kwarg; retry logic was inverted**  
Commit `1f0238b`. `AnthropicProvider.create()` did not pass `max_tokens` to the SDK call. The retry path was calling `_retry_with_lower_max()` (non-existent method) instead of `_retry_with_higher_max()`. Fix: `max_tokens = kwargs.get("max_tokens", self.max_tokens)`, loop uses `_DEFAULT_MAX_TOKENS = 8192` on first call and `_MAX_RETRY_TOKENS = 16384` on retry. Renamed method.

### IMPORTANT (fixed in same batch)

**I1: Role alternation not repaired after compression**  
Commit `4c6d6e8`. After compression, `_sanitize_tool_pairs()` could produce consecutive same-role messages (two "user" messages after removing tool pairs). This causes Anthropic API errors. Fix: Added role alternation enforcement in `_sanitize_tool_pairs()` — inserts a placeholder `"(context removed during compression)"` message when consecutive same-role messages appear.

**I2: Memory filename sanitization missing**  
Commit `4c6d6e8`. `MemoryExtractor` wrote LLM-returned filenames directly to disk. Fix: regex validation `^[a-zA-Z0-9_\-]+\.md$` + explicit checks for `/`, `..`, `\`.

**I3: Token baseline never reset after memory extraction**  
Commit `4c6d6e8`. `MemoryManager._initial_token_estimate` was set on first call and never updated, causing `token_delta` to grow monotonically across the session and triggering extraction on every subsequent conversation turn. Fix: reset `_initial_token_estimate = current_tokens` after successful extraction.

**I4: Observer used as context manager but `__enter__`/`__exit__` missing**  
Commit `bfdd3c8`. Code in tests used `with Observer(...) as obs:`. Fix: Added `__enter__` (returns self) and `__exit__` (calls close()).

**I5: tiktoken/cl100k_base undocumented approximation**  
Commit `bfdd3c8`. The approximation was silent. Fix: Added explicit comment in `compress.py` explaining that cl100k_base is GPT-4's tokenizer and gives ±10-20% for Claude models.

**I6: Project key used `str(Path.cwd())` without hashing in `agent.py`**  
Commit `bfdd3c8`. The raw cwd string was used as directory name. Fix: SHA-256 of `str(Path.cwd()).encode()`, first 8 chars — short but collision-resistant enough for local use.

**I7: QueryLoop accessed `_memory_manager` as public attribute from `NeoAgent.enable_memory()`**  
Commit `bfdd3c8`. `self._loop._memory_manager = memory_manager` bypasses encapsulation. Fix: `QueryLoop.__init__` already accepts `memory_manager` param; `enable_memory()` reconstructs the loop or assigns via private attribute. Currently still uses direct attribute assignment — partial fix, noted as remaining issue.

**I8: Observer `on_memory_extract_trigger/skip` methods exist but are never called**  
Commit `bfdd3c8`. The Observer has `on_memory_extract_trigger()` and `on_memory_extract_skip()` methods but `MemoryManager.maybe_extract()` does not call them. The Observer hooks for memory are partially wired.

### MINOR

- `PermissionChecker.auto_approve` default changed to `True` (with warning log) rather than silently blocking — better for programmatic use
- BashTool: `proc.returncode != 0` correctly returns `is_error=True` with the output
- E2E tests pass `allowed_directories=[tmp_path]` to all file tools — necessary after pathguard enforcement (commit `8014f52`)
- GrepTool shells out to system `grep` — no Python fallback if `grep` is not available on PATH

---

## 7. Known Limitations and Deferred to v3

**Architecture gaps:**

1. **chat() is stateless** — `NeoAgent.chat(message)` creates a fresh single-message history every call. There is no built-in session management; multi-turn conversations require callers to manage `Message` lists manually via `run()`.

2. **No multi-agent / subagent dispatching** — No `AgentTool`, no way to spawn sub-agents from within a tool call. Planned for v3.

3. **No hooks / event system** — No pre/post tool execution hooks. Observer is read-only (logging); it cannot intercept or modify tool calls.

4. **No MCP protocol** — All tools must be Python classes. No external MCP server support.

**Memory system gaps:**

5. **Memory extraction is synchronous at conversation end** — `maybe_extract()` is `await`-ed inline in the loop's `end_turn` path. It makes an LLM API call before returning to the caller. On long conversations, this adds noticeable latency. No background task / fire-and-forget.

6. **Memory retrieval is lexical only** — Keyword overlap scoring. Will miss semantic matches. No embedding-based retrieval. Acceptable for v2; v3 may add vector search.

7. **No memory deduplication** — Repeated conversations about the same topic accumulate duplicate memory files. `_rebuild_index()` uses first non-empty line of each file as description, which may not be the originally provided description.

8. **Observer memory hooks not wired** — `on_memory_extract_trigger()` and `on_memory_extract_skip()` methods exist on Observer but are never called from MemoryManager.

**Skill lazy loading gaps:**

9. **Skill activation is manual API only** — `PromptBuilder.activate_skill(name)` and `deactivate_skill(name)` must be called programmatically. There is no automatic activation (e.g., based on conversation topic detection). The DeerFlow-inspired auto-activation design was deliberately not implemented (non-deterministic).

**Provider gaps:**

10. **Context window sizes are hardcoded** — `AnthropicProvider` returns `200_000` always. `OpenAIProvider` has a lookup table. Neither queries the API for the actual context window. New model releases may be incorrect.

11. **tiktoken approximation** — `cl100k_base` gives ±10-20% for Claude models. For most use cases this is fine, but very tight context budgets could misfire.

12. **OpenAI provider: `max_tokens` default is 4096** — Different from Anthropic's 8192. No warning is shown.

**Test gaps:**

13. **No real API tests** — All 265 tests are mocked or local-file-based. The framework has not been validated against actual Anthropic or OpenAI APIs.

14. **No error recovery tests** — No tests for API rate limits, network timeouts, or partial responses.

15. **No concurrent safety tests** — The concurrent execution path is code-reviewed but not stress-tested.

---

## 8. What We Want Reviewed

### Q1: Remaining Security Gaps

- Is the BashTool blocklist sufficient? What bypass techniques are we missing?
- The `auto_approve=True` default in `PermissionChecker` — is this acceptable behavior for a library default, or should it be `False` with explicit opt-in?
- `validate_path` uses `Path.resolve()` which follows symlinks. Are there edge cases we missed (e.g., bind mounts, FUSE filesystems)?
- GrepTool shells out to system `grep` — is this a security concern (command injection via `pattern` parameter)?

### Q2: Architecture Soundness for v3

- The `NeoAgent` class currently does a lot of wiring in `enable_memory()` and `enable_logging()` by directly setting attributes on `self._loop` (`_memory_manager`, `_observer`). This is an encapsulation violation. Should `QueryLoop` be rebuilt when features are enabled, or should it accept all deps upfront?
- For multi-agent (v3), `QueryLoop` needs to dispatch to sub-agents via an `AgentTool`. How should the sub-agent share the parent's `ToolRegistry` and `PermissionChecker`? Or should each sub-agent get its own isolated copies?
- Memory persistence is per-project (SHA-256 of cwd). Is this the right isolation boundary for multi-agent scenarios where sub-agents might share or have separate memories?

### Q3: API Design Issues

- `NeoAgent.chat()` is stateless (single message → single response). Should this be `chat(messages: list[Message]) -> str` to support multi-turn, or should there be explicit `Session` objects?
- `PromptSection` uses `priority: int` with `is_static: bool` for ordering (static sorts before dynamic within priority). Is this the right abstraction? Could it cause confusion when a high-priority dynamic section appears after all static sections regardless of its priority value?
- `BaseTool.permission` is a `str` literal field, not enforced by ABC. A subclass that sets `permission = "invalid"` would silently pass to `PermissionChecker.check()`, which only checks for exact `"auto"` / `"deny"` — everything else is treated as `"ask"`. Should this be a typed enum?
- `ToolResult.call_id` is set to `""` at construction and overwritten by `ToolRegistry._run_one()`. This is a two-step initialization smell. Should `ToolResult` be constructed without `call_id` and have it injected?

### Q4: Test Coverage Gaps

- No tests for `compress.py`'s role alternation repair (the placeholder insertion after `_sanitize_tool_pairs`). This is a correctness-critical path for Anthropic API compatibility.
- No tests for concurrent tool execution in `ToolRegistry.execute()` — the `asyncio.gather` path.
- No tests for `MemoryExtractor` with malformed JSON from the LLM (the code handles it via try/except but edge cases like partial JSON aren't tested).
- `OpenAIProvider` message serialization has complex logic for mixed text+tool_use assistant messages. Are all combinations covered?
- No test for `BashTool` timeout behavior (would require `time.sleep` in a test command — slow but important).

### Q5: Anti-patterns and Code Smells

- `ContextCompressor._previous_summary` is public (no leading `_` ... wait, it has `_`). But `loop.py:45` reads `self._compressor._previous_summary` directly — crossing a module boundary to access private state.
- `BaseTool` declares `name`, `description`, `input_model` as class-level annotations without `ClassVar` — they look like instance attributes but are intended as class attributes. This is a Python ABC pattern limitation but can confuse type checkers.
- `PermissionChecker.__init__` accepts `auto_approve: bool = True`. This default is safe for programmatic use but is a footgun if someone deploys this with BashTool registered without an `ask_callback`. The warning log is the only protection.
- `MemoryExtractor._rebuild_index()` is called after every `extract()` and scans the entire memory directory. Fine at small scale, but `O(n)` in number of memory files.
- `NeoAgent.enable_memory()` and `enable_logging()` both accept `Path | None` and have `"Path | None"` as string annotation (forward reference). This is because of `from __future__ import annotations` at top + conditional import of `Path`. It works but is unusual style.

---

## 9. How to Run

### Setup

```bash
# Python 3.11+ required
git clone <repo>
cd neoagent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Dependencies** (from `pyproject.toml`):
- `anthropic>=0.28.0`
- `openai>=1.0`
- `pydantic>=2.0`
- `tiktoken>=0.7.0`
- Dev: `pytest>=8.0`, `pytest-asyncio>=0.23`

### Unit Tests (265 tests, no API key needed)

```bash
python3 -m pytest tests/ -v
```

Test layout:
- `tests/core/` — types, loop, prompt, compress (mock provider)
- `tests/tools/` — base, registry, permission, pathguard, all 6 builtins
- `tests/providers/` — Anthropic and OpenAI providers (mocked SDK)
- `tests/memory/` — store, retriever, extractor, manager (mock provider + tmp files)
- `tests/test_integration.py` — full loop scenarios (mock provider)
- `tests/test_observe.py` — Observer with file output (no API)
- `tests/test_agent.py` — NeoAgent wiring (mock provider)
- `tests/test_scaffold.py` — basic instantiation test

### Run a specific test file

```bash
python3 -m pytest tests/core/test_loop.py -v
python3 -m pytest tests/tools/builtin/test_bash.py -v
python3 -m pytest tests/memory/ -v
```

### Integration Tests (mock provider, no API key)

```bash
python3 -m pytest tests/test_integration.py -v
```

These test: single-turn conversation, tool call flow with ReadTool, max_turns enforcement, compression threshold detection.

### Observer Logging

Enable in code:
```python
from neoagent import NeoAgent
from neoagent.config import NeoAgentConfig

config = NeoAgentConfig(api_key="...", provider="anthropic")
agent = NeoAgent(config)
observer = agent.enable_logging(log_dir=Path("logs/"), console=True)

result = await agent.run(messages)

observer.close()
# Timestamped log file created at: logs/YYYY-MM-DD-HH-MM-SS.log
```

The log file gets full untruncated content; console gets truncated version (500 chars default for text, 300 chars for tool args).

### Real API Smoke Test (requires API key)

**Not in the test suite yet.** Manual test pattern:
```python
import asyncio
from neoagent import NeoAgent
from neoagent.config import NeoAgentConfig
from neoagent.tools.builtin.bash import BashTool

async def main():
    config = NeoAgentConfig(api_key="sk-ant-...", provider="anthropic")
    agent = NeoAgent(config)
    agent.register_tool(BashTool())
    result = await agent.chat("What is 2+2?")
    print(result)

asyncio.run(main())
```

---

## 10. Repository Structure Reference

```
neoagent/
├── pyproject.toml          Build config, dependencies, pytest config
├── neoagent/               Main package
│   ├── __init__.py         Exports: NeoAgent, NeoAgentConfig, Message, ConversationResult
│   ├── agent.py
│   ├── config.py
│   ├── observe.py
│   ├── core/               Loop, prompt, compress, types
│   ├── tools/              BaseTool, registry, permission, pathguard, builtin/
│   ├── providers/          base, anthropic, openai
│   └── memory/             store, retriever, extractor, manager
└── tests/
    ├── conftest.py         pytest-asyncio mode = "auto"
    ├── core/
    ├── tools/
    ├── providers/
    ├── memory/
    ├── test_agent.py
    ├── test_integration.py
    ├── test_observe.py
    └── test_scaffold.py
```

---

## 11. Commit History Summary

| Commit | Description |
|--------|-------------|
| `83aa259` | BaseTool ABC, get_schema() |
| `817aa40` | ToolRegistry, concurrent execution |
| `554bbe6` | PermissionChecker (unconnected at this point) |
| `83aa259` | Provider ABC, AnthropicProvider |
| `ec5266a` | PromptBuilder, PromptSection, static/dynamic |
| `e9a7d14` | ContextCompressor, LLM summarize, circuit-breaker |
| `e5fccfc` | QueryLoop, tool dispatch, compression trigger |
| `345aa0a` | NeoAgent, NeoAgentConfig |
| `45a318f` | ReadTool |
| `8f1f3f4` | BashTool, GrepTool, GlobTool |
| `f771ab3` | WriteTool, EditTool |
| `92df1a0` | Integration tests |
| `3b6589a` | OpenAI provider |
| `f6a6427` | base_url support |
| `d27b154` | MemoryStore |
| `ca21811` | MemoryRetriever |
| `54054f2` | MemoryExtractor |
| `9888ad6` | MemoryManager wired into loop |
| `58732af` | Iterative compression upgrade (Hermes B2) |
| `e72b6e1` | Skill lazy loading in PromptBuilder |
| `ebd136f` | V2 smoke tests |
| `8dcdb1a` | V2 observable integration test |
| `7691eae` | Observer |
| `b14f75f` | Memory + observer fixes |
| **`1f0238b`** | **CRITICAL: Provider max_tokens fix, retry logic** |
| **`d8045ba`** | **CRITICAL: PermissionChecker wired, BashTool blocklist** |
| **`4c38417`** | **CRITICAL: pathguard + allowed_directories for file tools** |
| `4c6d6e8` | Role alternation fix, filename sanitization, token baseline |
| `bfdd3c8` | Observer context manager, tiktoken doc, sha256, encapsulation |
| `8014f52` | E2E tests: pass allowed_directories to tools |
