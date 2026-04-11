"""
v2 Observable Integration Test — neoagent

Exercises and OBSERVES all three v2 features with detailed logging:
  A. Memory System  — full lifecycle: extraction, persistence, new-session recall
  B. Context Compression — trigger detection, summary accumulation, reset
  C. Skill Lazy Loading — prompt diff before/after activate/deactivate

Usage:
    python examples/v2_integration_test.py

Auto-detects provider from .env (ANTHROPIC_API_KEY takes priority over OPENAI_API_KEY).

This is a DIAGNOSTIC tool — it prints every observable internal state.
Pass/fail is secondary; seeing what's happening is the point.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-load .env from project root
# ---------------------------------------------------------------------------

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _k, _v = _k.strip(), _v.strip()
            if _v and _k not in os.environ:
                os.environ[_k] = _v

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty() or os.environ.get("FORCE_COLOR")

GREEN  = "\033[92m" if _USE_COLOUR else ""
RED    = "\033[91m" if _USE_COLOUR else ""
CYAN   = "\033[96m" if _USE_COLOUR else ""
YELLOW = "\033[93m" if _USE_COLOUR else ""
MAGENTA = "\033[95m" if _USE_COLOUR else ""
BOLD   = "\033[1m"  if _USE_COLOUR else ""
DIM    = "\033[2m"  if _USE_COLOUR else ""
RESET  = "\033[0m"  if _USE_COLOUR else ""


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _header(title: str) -> None:
    bar = "═" * 68
    print(f"\n{CYAN}{bar}{RESET}")
    print(f"{BOLD}  {title}{RESET}  {DIM}[{_ts()}]{RESET}")
    print(f"{CYAN}{bar}{RESET}")


def _subheader(title: str) -> None:
    bar = "─" * 60
    print(f"\n{YELLOW}{bar}{RESET}")
    print(f"{BOLD}  {title}{RESET}  {DIM}[{_ts()}]{RESET}")
    print(f"{YELLOW}{bar}{RESET}")


def _obs(label: str, value: object) -> None:
    """Print an observable key/value pair."""
    val_str = str(value)
    # Truncate very long values but show how long they were
    if len(val_str) > 400:
        val_str = val_str[:400] + f"{DIM}... [{len(str(value))} chars total]{RESET}"
    print(f"  {MAGENTA}[OBS]{RESET} {BOLD}{label}{RESET}: {val_str}")


def _info(msg: str) -> None:
    print(f"  {DIM}[{_ts()}]{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def _err(msg: str) -> None:
    print(f"  {RED}[ERR]{RESET} {msg}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET} {msg}")


# ---------------------------------------------------------------------------
# Provider / config detection (same pattern as e2e_test.py)
# ---------------------------------------------------------------------------

def _detect_config() -> dict:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key    = os.environ.get("OPENAI_API_KEY")

    if anthropic_key:
        return {
            "provider": "anthropic",
            "api_key": anthropic_key,
            "model": os.environ.get("ANTHROPIC_MODEL") or "claude-haiku-4-20250514",
            "base_url": os.environ.get("ANTHROPIC_BASE_URL") or None,
            "auto_approve_tools": True,
        }
    if openai_key:
        return {
            "provider": "openai",
            "api_key": openai_key,
            "model": os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
            "base_url": os.environ.get("OPENAI_BASE_URL") or None,
            "auto_approve_tools": True,
        }

    print(
        f"{RED}ERROR:{RESET} No API key found in environment or .env file.\n"
        "Set ANTHROPIC_API_KEY or OPENAI_API_KEY and re-run.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Test A: Memory System — full lifecycle
# ---------------------------------------------------------------------------

async def test_a_memory_lifecycle(cfg: dict) -> None:
    """
    Full memory lifecycle:
      1. Create agent with enable_memory() pointing to a tmp dir
      2. Observe empty memory prompt section
      3. Run a conversation with 5+ tool calls (BashTool × 5 commands)
      4. Observe MemoryManager internal counters after conversation
      5. Inspect files written to memory_dir
      6. Create a NEW agent with the SAME memory_dir
      7. Observe memory prompt section is now non-empty (cross-session recall)
    """
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig
    from neoagent.tools.builtin.bash import BashTool

    _header("TEST A: Memory System — Full Lifecycle")

    with tempfile.TemporaryDirectory(prefix="neoagent_mem_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        # ── Step 1: Create agent with memory ──────────────────────────────
        _subheader("A1. Create agent + enable_memory()")
        config = NeoAgentConfig(**cfg)
        agent = NeoAgent(config)
        agent.enable_memory(memory_dir=tmp_path)
        agent.register_tool(BashTool())

        _obs("Memory dir", tmp_path)
        _obs("_loop._memory_manager type", type(agent._loop._memory_manager).__name__)

        # ── Step 2: Observe empty memory prompt section ────────────────────
        _subheader("A2. Memory prompt section — BEFORE any conversation")
        mm = agent._loop._memory_manager
        section_before = mm.build_prompt_section()
        _obs("build_prompt_section() (empty)", repr(section_before) if section_before else "(empty string)")
        _obs("_tool_calls_count (initial)", mm._tool_calls_count)
        _obs("_initial_token_estimate (initial)", mm._initial_token_estimate)

        # ── Step 3: Run conversation producing 5+ tool calls ──────────────
        _subheader("A3. Run conversation — 5 bash commands (triggers extraction)")
        _info("Sending prompt requesting 5 echo commands sequentially...")

        prompt = (
            "I'm Neo, a senior AI engineer working on the neoagent project. "
            "My preferences: I always use Python 3.11+, prefer pydantic v2, and "
            "follow TDD workflow. The project repo is at /tmp/neoagent. "
            "Now please run these 5 commands to verify my environment: "
            "(1) echo 'python_version=3.11' "
            "(2) echo 'framework=neoagent' "
            "(3) echo 'test_runner=pytest' "
            "(4) echo 'style=black' "
            "(5) echo 'typing=strict'"
        )

        try:
            response = await agent.chat(prompt)
            _obs("Agent response (excerpt)", response[:300] if response else "(empty)")
        except Exception as exc:
            _err(f"chat() raised: {exc}")
            import traceback
            traceback.print_exc()
            return

        # ── Step 4: Observe MemoryManager counters after conversation ──────
        _subheader("A4. MemoryManager state — AFTER conversation")
        _obs("_tool_calls_count", mm._tool_calls_count)
        _obs("_initial_token_estimate (baseline set on first call)", mm._initial_token_estimate)

        # Extraction threshold reference
        _obs("Extraction threshold (tool calls)", 5)
        _obs("Extraction threshold (token delta)", 4000)

        if mm._tool_calls_count == 0 and mm._initial_token_estimate > 0:
            _ok("tool_calls_count reset to 0 — extraction was triggered at end_turn")
        elif mm._tool_calls_count >= 5:
            _warn("tool_calls_count >= 5 — extraction may NOT have triggered (check token delta)")
        else:
            _info("tool_calls_count < 5 — agent may not have used the bash tool 5 times")

        # ── Step 5: Inspect memory_dir files ──────────────────────────────
        _subheader("A5. Inspect files in memory_dir")
        all_files = sorted(tmp_path.glob("*"))
        _obs("Files in memory_dir", [f.name for f in all_files])

        memory_md = tmp_path / "MEMORY.md"
        if memory_md.exists():
            _ok("MEMORY.md exists")
            _obs("MEMORY.md contents", memory_md.read_text(encoding="utf-8"))
        else:
            _warn(
                "MEMORY.md does NOT exist — extraction did not fire. "
                "This can happen if: agent made < 5 tool calls, or token delta < 4000, "
                "or the LLM returned an empty JSON array []."
            )

        topic_files = [f for f in all_files if f.suffix == ".md" and f.name != "MEMORY.md"]
        for tf in topic_files:
            _obs(f"Topic file [{tf.name}]", tf.read_text(encoding="utf-8"))

        # ── Step 6 + 7: New agent with same memory_dir ────────────────────
        _subheader("A6. New agent — same memory_dir (cross-session recall)")
        config2 = NeoAgentConfig(**cfg)
        agent2 = NeoAgent(config2)
        agent2.enable_memory(memory_dir=tmp_path)

        mm2 = agent2._loop._memory_manager
        section_new = mm2.build_prompt_section()
        _obs("build_prompt_section() in new session", repr(section_new) if section_new else "(empty — no memories persisted)")

        if section_new:
            _ok("Cross-session memory recall WORKS — new agent sees previous memories")
        else:
            _warn("No memories visible in new session (extraction may not have fired or LLM returned [])")

        # Also observe via prompt_builder
        full_prompt = agent2._prompt_builder.build()
        _obs("Full system prompt of new agent (first 500 chars)", full_prompt[:500])


# ---------------------------------------------------------------------------
# Test B: Context Compression — trigger + iterative summary observation
# ---------------------------------------------------------------------------

async def test_b_context_compression(cfg: dict) -> None:
    """
    Context compression observation:
      1. Create agent with a tiny context_budget (1500 tokens)
      2. Observe compressor state before any conversation
      3. Multi-turn conversation — check should_compress + _previous_summary
      4. Observe _previous_summary after compression fires
      5. Test reset_session_state()
    """
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig
    from neoagent.tools.builtin.bash import BashTool

    _header("TEST B: Context Compression — Trigger + Iterative Summary")

    TINY_BUDGET = 500  # very small — will trigger at ~350 tokens (70% threshold)

    # ── Step 1: Create agent with tiny context_budget ─────────────────────
    _subheader("B1. Create agent — context_budget=1500")
    cfg_copy = dict(cfg)
    cfg_copy["context_budget"] = TINY_BUDGET
    config = NeoAgentConfig(**cfg_copy)
    agent = NeoAgent(config)
    agent.register_tool(BashTool())

    comp = agent._loop._compressor
    _obs("context_budget", agent._loop.context_budget)
    _obs("Compression trigger threshold (70%)", int(TINY_BUDGET * 0.7))
    _obs("_previous_summary BEFORE any conversation", repr(comp._previous_summary))
    _obs("_consecutive_failures", comp._consecutive_failures)

    # ── Step 2: Turn 1 — short exchange ──────────────────────────────────
    _subheader("B2. Turn 1 — short echo command")
    from neoagent.core.types import Message

    msgs_after_t1: list[Message] = []

    async def _chat_and_capture(prompt_text: str) -> str:
        """Run chat and capture messages that were in the loop via a turn callback."""
        nonlocal msgs_after_t1
        # We re-use the loop directly so we can inspect state mid-flight
        messages = [Message(role="user", content=prompt_text)]
        result = await agent._loop.run(messages)
        if result.turns:
            last = result.turns[-1].response
            from neoagent.core.types import TextBlock
            if isinstance(last.content, list):
                return "\n".join(b.text for b in last.content if isinstance(b, TextBlock))
            return last.content if isinstance(last.content, str) else ""
        return ""

    try:
        resp1 = await _chat_and_capture("Run `echo 'hello world'` and report the output.")
        _obs("Turn 1 response", resp1[:200])
    except Exception as exc:
        _err(f"Turn 1 raised: {exc}")
        import traceback
        traceback.print_exc()

    _subheader("B3. Compressor state after Turn 1")
    _obs("_previous_summary after T1", repr(comp._previous_summary))
    _obs("_consecutive_failures after T1", comp._consecutive_failures)

    # Manually test should_compress with a fabricated token count
    dummy_msgs = [Message(role="user", content="x" * 800), Message(role="assistant", content="y" * 800)]
    dummy_tokens = comp.estimate_tokens(dummy_msgs)
    dummy_should = comp.should_compress(dummy_msgs, [], TINY_BUDGET)
    _obs("estimate_tokens for ~1600-char messages", dummy_tokens)
    _obs("should_compress with those dummy messages", dummy_should)

    # ── Step 3: Turn 2 — longer exchange to push over threshold ──────────
    _subheader("B4. Turn 2 — longer prompt to push context over budget")
    long_prompt = (
        "Please run these commands one by one: "
        "`echo 'alpha beta gamma delta epsilon'`, "
        "`echo 'one two three four five six seven eight nine ten'`, "
        "`echo 'the quick brown fox jumps over the lazy dog'`. "
        "After each, tell me what you got. Then summarise all three outputs."
    )
    try:
        resp2 = await _chat_and_capture(long_prompt)
        _obs("Turn 2 response", resp2[:300])
    except Exception as exc:
        _err(f"Turn 2 raised: {exc}")
        import traceback
        traceback.print_exc()

    _subheader("B5. Compressor state after Turn 2")
    _obs("_previous_summary after T2", repr(comp._previous_summary))
    _obs("_consecutive_failures after T2", comp._consecutive_failures)

    if comp._previous_summary:
        _ok("Compression WAS triggered — _previous_summary is non-None")
        _obs("Summary content", comp._previous_summary)
    else:
        _warn(
            "_previous_summary is still None — compression has not fired yet. "
            "The conversation may not have exceeded the 70% threshold of 1500 tokens. "
            "Try lowering context_budget further (e.g. 800) or adding more turns."
        )

    # ── Step 4: Demonstrate iterative summary (Turn 3) ───────────────────
    _subheader("B6. Turn 3 — one more exchange to test iterative summary accumulation")
    try:
        resp3 = await _chat_and_capture("Run `echo 'final step'` and confirm you got it.")
        _obs("Turn 3 response", resp3[:200])
    except Exception as exc:
        _err(f"Turn 3 raised: {exc}")
        import traceback
        traceback.print_exc()

    _obs("_previous_summary after T3 (should include T2 content if updated)", repr(comp._previous_summary))

    # ── Step 5: reset_session_state() ────────────────────────────────────
    _subheader("B7. reset_session_state() — clears iterative summary")
    _obs("_previous_summary BEFORE reset", repr(comp._previous_summary))
    comp.reset_session_state()
    _obs("_previous_summary AFTER reset", repr(comp._previous_summary))
    _obs("_consecutive_failures AFTER reset", comp._consecutive_failures)

    if comp._previous_summary is None and comp._consecutive_failures == 0:
        _ok("reset_session_state() correctly cleared all state")
    else:
        _err("reset_session_state() did not fully clear state — investigate")


# ---------------------------------------------------------------------------
# Test C: Skill Lazy Loading — observable prompt changes
# ---------------------------------------------------------------------------

async def test_c_skill_lazy_loading(cfg: dict) -> None:
    """
    Skill lazy loading observation:
      1. Create agent
      2. Register skill "code_expert" (not yet active)
      3. Print system prompt BEFORE activation — skill should be absent
      4. Activate skill
      5. Print system prompt AFTER activation — skill content must appear
      6. Chat with activated skill — agent should behave accordingly
      7. Deactivate skill
      8. Print system prompt AFTER deactivation — skill content must be gone
      9. Confirm skill is still registered (can be re-activated)
    """
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig
    from neoagent.core.prompt import PromptSection

    _header("TEST C: Skill Lazy Loading — Observable Prompt Changes")

    config = NeoAgentConfig(**cfg)
    agent = NeoAgent(config)

    SKILL_NAME = "code_expert"
    SKILL_CONTENT = (
        "You are an expert Python developer with 15 years of experience. "
        "When asked to write code, always include type hints and docstrings. "
        "Prefer list comprehensions over for-loops where appropriate."
    )

    # ── Step 1: Register skill (NOT activated) ────────────────────────────
    _subheader("C1. Register skill — NOT yet activated")
    skill_section = PromptSection(
        name=SKILL_NAME,
        content=SKILL_CONTENT,
        priority=10,
        is_static=True,
    )
    agent._prompt_builder.register_skill(SKILL_NAME, skill_section)

    _obs("Skill registered", SKILL_NAME)
    _obs(f"is_skill_active('{SKILL_NAME}') after register", agent._prompt_builder.is_skill_active(SKILL_NAME))
    _obs("_skills dict keys", list(agent._prompt_builder._skills.keys()))

    # ── Step 2: Observe system prompt BEFORE activation ───────────────────
    _subheader("C2. System prompt — BEFORE activation")
    prompt_before = agent._prompt_builder.build()
    _obs("Full system prompt (before)", prompt_before)
    skill_absent = SKILL_CONTENT[:40] not in prompt_before
    if skill_absent:
        _ok("Skill content is ABSENT from system prompt (correct — not yet activated)")
    else:
        _err("Skill content is PRESENT before activation — lazy loading bug!")

    # ── Step 3: Activate skill ────────────────────────────────────────────
    _subheader("C3. Activate skill")
    agent._prompt_builder.activate_skill(SKILL_NAME)
    _obs(f"is_skill_active('{SKILL_NAME}') after activate_skill()", agent._prompt_builder.is_skill_active(SKILL_NAME))

    # ── Step 4: Observe system prompt AFTER activation ────────────────────
    _subheader("C4. System prompt — AFTER activation")
    prompt_after = agent._prompt_builder.build()
    _obs("Full system prompt (after)", prompt_after)
    skill_present = SKILL_CONTENT[:40] in prompt_after
    if skill_present:
        _ok("Skill content IS present in system prompt (lazy loading works)")
    else:
        _err("Skill content is NOT present after activation — bug!")

    # ── Step 5: Chat with activated skill ─────────────────────────────────
    _subheader("C5. Chat with activated skill — agent should act as code_expert")
    _info("Asking agent to write a simple function (should include type hints + docstring)")
    try:
        resp = await agent.chat("Write a Python function that takes a list of integers and returns their sum.")
        _obs("Agent response (with skill active)", resp[:400])
        has_type_hints = "->" in resp or ": int" in resp or ": list" in resp or "List" in resp
        has_docstring = '"""' in resp or "'''" in resp
        _obs("Response has type hints", has_type_hints)
        _obs("Response has docstring", has_docstring)
        if has_type_hints or has_docstring:
            _ok("Agent appears to be following skill instructions")
        else:
            _warn("Response lacks type hints/docstring — skill may not be influencing behaviour strongly")
    except Exception as exc:
        _err(f"chat() raised: {exc}")
        import traceback
        traceback.print_exc()

    # ── Step 6: Deactivate skill ──────────────────────────────────────────
    _subheader("C6. Deactivate skill")
    agent._prompt_builder.deactivate_skill(SKILL_NAME)
    _obs(f"is_skill_active('{SKILL_NAME}') after deactivate_skill()", agent._prompt_builder.is_skill_active(SKILL_NAME))

    # ── Step 7: Observe system prompt AFTER deactivation ─────────────────
    _subheader("C7. System prompt — AFTER deactivation")
    prompt_deactivated = agent._prompt_builder.build()
    _obs("Full system prompt (after deactivation)", prompt_deactivated)
    skill_gone = SKILL_CONTENT[:40] not in prompt_deactivated
    if skill_gone:
        _ok("Skill content is ABSENT again (deactivation works)")
    else:
        _err("Skill content still present after deactivation — bug!")

    # ── Step 8: Confirm skill still registered (can re-activate) ─────────
    _subheader("C8. Confirm skill still registered after deactivation")
    _obs("_skills dict keys (after deactivate)", list(agent._prompt_builder._skills.keys()))
    still_registered = SKILL_NAME in agent._prompt_builder._skills
    if still_registered:
        _ok("Skill is still in _skills registry — can be re-activated without re-registering")
    else:
        _err("Skill was removed from _skills — deactivate() should NOT remove from registry")

    # ── Step 9: Re-activate and confirm it works again ────────────────────
    _subheader("C9. Re-activate skill — confirm idempotent registration")
    agent._prompt_builder.activate_skill(SKILL_NAME)
    prompt_reactivated = agent._prompt_builder.build()
    skill_back = SKILL_CONTENT[:40] in prompt_reactivated
    _obs(f"is_skill_active('{SKILL_NAME}') after re-activation", agent._prompt_builder.is_skill_active(SKILL_NAME))
    if skill_back:
        _ok("Re-activation succeeded — skill content present again")
    else:
        _err("Re-activation failed — skill not in prompt after second activate_skill()")

    # ── Summary diff ──────────────────────────────────────────────────────
    _subheader("C10. Prompt length diff summary")
    _obs("Prompt length — before activation", len(prompt_before))
    _obs("Prompt length — after activation", len(prompt_after))
    _obs("Prompt length — after deactivation", len(prompt_deactivated))
    _obs("Prompt length — after re-activation", len(prompt_reactivated))
    _obs("Skill content length", len(SKILL_CONTENT))
    delta = len(prompt_after) - len(prompt_before)
    _obs("Delta (after − before)", delta)
    if abs(delta - len(SKILL_CONTENT)) < 50:
        _ok(f"Delta ({delta}) matches skill content length ({len(SKILL_CONTENT)}) — accounting for section header")
    else:
        _info(f"Delta is {delta} vs skill content {len(SKILL_CONTENT)} — difference due to section header/formatting")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    cfg = _detect_config()

    bar = "═" * 68
    print(f"\n{CYAN}{bar}{RESET}")
    print(f"{BOLD}  neoagent v2 Observable Integration Test{RESET}")
    print(f"{CYAN}{bar}{RESET}")
    print(f"  Provider  : {BOLD}{cfg['provider']}{RESET}")
    print(f"  Model     : {BOLD}{cfg['model']}{RESET}")
    if cfg.get("base_url"):
        print(f"  Base URL  : {cfg['base_url']}")
    print(f"  Timestamp : {_ts()}")
    print(f"{CYAN}{bar}{RESET}")

    print(f"\n{DIM}This test prints internal state at each key point.")
    print(f"Goal: observe WHAT is happening, not just whether the agent responds correctly.{RESET}")

    # Run all three tests sequentially (each uses its own agents/state)
    await test_a_memory_lifecycle(cfg)
    await test_b_context_compression(cfg)
    await test_c_skill_lazy_loading(cfg)

    _header("DONE")
    print(f"  All three test sections completed at {_ts()}")
    print(f"  Review {BOLD}[OBS]{RESET} lines above for observable internal state.")
    print(f"  Review {GREEN}[OK]{RESET} / {YELLOW}[WARN]{RESET} / {RED}[ERR]{RESET} lines for findings.\n")


if __name__ == "__main__":
    asyncio.run(main())
