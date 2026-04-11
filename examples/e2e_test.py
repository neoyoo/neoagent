"""
End-to-end tests for neoagent framework using real API calls.

Usage:
    ANTHROPIC_API_KEY=sk-... python examples/e2e_test.py
    OPENAI_API_KEY=sk-...    python examples/e2e_test.py

Auto-detects provider based on which env var is set.
If both are set, ANTHROPIC_API_KEY takes priority.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Auto-load .env file from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if value and key not in os.environ:
                os.environ[key] = value

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty() or os.environ.get("FORCE_COLOR")

GREEN = "\033[92m" if _USE_COLOUR else ""
RED   = "\033[91m" if _USE_COLOUR else ""
CYAN  = "\033[96m" if _USE_COLOUR else ""
BOLD  = "\033[1m"  if _USE_COLOUR else ""
RESET = "\033[0m"  if _USE_COLOUR else ""


def _pass(msg: str = "PASS") -> str:
    return f"{GREEN}{BOLD}{msg}{RESET}"


def _fail(msg: str = "FAIL") -> str:
    return f"{RED}{BOLD}{msg}{RESET}"


def _header(title: str) -> None:
    bar = "─" * 60
    print(f"\n{CYAN}{bar}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{CYAN}{bar}{RESET}")


# ---------------------------------------------------------------------------
# Provider / config detection
# ---------------------------------------------------------------------------

def _detect_config() -> dict:
    """Return config dict with provider, api_key, model, base_url."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key    = os.environ.get("OPENAI_API_KEY")

    if anthropic_key:
        return {
            "provider": "anthropic",
            "api_key": anthropic_key,
            "model": os.environ.get("ANTHROPIC_MODEL") or "claude-haiku-4-20250514",
            "base_url": os.environ.get("ANTHROPIC_BASE_URL") or None,
        }
    if openai_key:
        return {
            "provider": "openai",
            "api_key": openai_key,
            "model": os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
            "base_url": os.environ.get("OPENAI_BASE_URL") or None,
        }

    print(
        f"{RED}ERROR:{RESET} No API key found in environment or .env file.\n"
        "Set ANTHROPIC_API_KEY or OPENAI_API_KEY and re-run.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Test runner state
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool]] = []


def _record(name: str, passed: bool) -> None:
    _results.append((name, passed))
    label = _pass() if passed else _fail()
    print(f"\nResult: {label}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_simple_chat(cfg: dict) -> None:
    """Test 1: Simple chat — ask for 2+2, expect '4' in response."""
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig

    _header("Test 1: Simple Chat")

    config = NeoAgentConfig(**cfg)
    agent  = NeoAgent(config)

    try:
        response = await agent.chat("What is 2+2? Reply with just the number.")
        print(f"Response: {response!r}")
        passed = "4" in response
        _record("Simple Chat", passed)
    except Exception as exc:
        print(f"{RED}Exception:{RESET} {exc}")
        _record("Simple Chat", False)


async def test_tool_read(cfg: dict) -> None:
    """Test 2: Tool use (ReadTool) — agent reads a temp file and reports contents."""
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig
    from neoagent.tools.builtin.read import ReadTool

    _header("Test 2: Tool Use — Read")

    secret = "NEOAGENT_SENTINEL_VALUE_42"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(f"Hello from neoagent e2e test!\n{secret}\n")
        tmp_path = f.name

    try:
        config = NeoAgentConfig(**cfg)
        agent  = NeoAgent(config)
        agent.register_tool(ReadTool(allowed_directories=[Path(tmp_path).parent]))

        prompt = f"Please read the file at {tmp_path} and tell me what it contains."
        response = await agent.chat(prompt)
        print(f"Response: {response!r}")
        passed = secret in response
        _record("Tool Use (Read)", passed)
    except Exception as exc:
        print(f"{RED}Exception:{RESET} {exc}")
        _record("Tool Use (Read)", False)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def test_tool_bash(cfg: dict) -> None:
    """Test 3: Tool use (BashTool) — agent runs 'echo hello' and reports output."""
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig
    from neoagent.tools.builtin.bash import BashTool

    _header("Test 3: Tool Use — Bash")

    config = NeoAgentConfig(**cfg)
    agent  = NeoAgent(config)
    agent.register_tool(BashTool())

    try:
        response = await agent.chat(
            "Run the shell command `echo hello` and tell me what it printed."
        )
        print(f"Response: {response!r}")
        passed = "hello" in response.lower()
        _record("Tool Use (Bash)", passed)
    except Exception as exc:
        print(f"{RED}Exception:{RESET} {exc}")
        _record("Tool Use (Bash)", False)


async def test_multi_turn_tools(cfg: dict) -> None:
    """Test 4: Multi-turn tool use — GlobTool + ReadTool over a temp dir with .py files."""
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig
    from neoagent.tools.builtin.glob import GlobTool
    from neoagent.tools.builtin.read import ReadTool

    _header("Test 4: Multi-turn Tool Use — Glob + Read")

    # Build a temp directory with 3 .py files
    tmp_dir = tempfile.mkdtemp(prefix="neoagent_e2e_")
    files_created: list[str] = []
    try:
        for i in range(1, 4):
            p = Path(tmp_dir) / f"module_{i}.py"
            p.write_text(f"# module {i}\nVALUE_{i} = {i * 10}\n")
            files_created.append(str(p))

        config = NeoAgentConfig(**cfg)
        agent  = NeoAgent(config)
        agent.register_tool(GlobTool(allowed_directories=[Path(tmp_dir)]))
        agent.register_tool(ReadTool(allowed_directories=[Path(tmp_dir)]))

        prompt = (
            f"List all .py files in {tmp_dir} using the glob tool "
            f"(pattern '*.py'), then read the first one and tell me its contents."
        )
        response = await agent.chat(prompt)
        print(f"Response: {response!r}")

        # Expect the agent to mention at least one .py filename and some content
        mentioned_py = any(
            Path(f).name in response for f in files_created
        )
        mentioned_content = "VALUE_" in response or "module" in response.lower()
        passed = mentioned_py and mentioned_content
        _record("Multi-turn Tool Use (Glob+Read)", passed)
    except Exception as exc:
        print(f"{RED}Exception:{RESET} {exc}")
        _record("Multi-turn Tool Use (Glob+Read)", False)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def test_memory_enable(cfg: dict) -> None:
    """Test 5: enable_memory() — agent with memory enabled completes a chat."""
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig
    import tempfile
    from pathlib import Path

    _header("Test 5: Memory System — enable_memory()")

    config = NeoAgentConfig(**cfg)
    agent = NeoAgent(config)

    with tempfile.TemporaryDirectory() as tmp:
        agent.enable_memory(memory_dir=Path(tmp))
        try:
            response = await agent.chat("What is 3+3? Reply with just the number.")
            print(f"Response: {response!r}")
            passed = "6" in response
            _record("Memory — enable_memory()", passed)
        except Exception as exc:
            print(f"{RED}Exception:{RESET} {exc}")
            _record("Memory — enable_memory()", False)


async def test_skill_lazy_load(cfg: dict) -> None:
    """Test 6: skill lazy loading — activate/deactivate a skill section."""
    from neoagent.agent import NeoAgent
    from neoagent.config import NeoAgentConfig
    from neoagent.core.prompt import PromptSection

    _header("Test 6: Skill Lazy Loading")

    config = NeoAgentConfig(**cfg)
    agent = NeoAgent(config)
    agent._prompt_builder.register_skill(
        "math_expert",
        PromptSection(name="math_expert", content="You excel at math.", priority=1, is_static=True),
    )
    agent._prompt_builder.activate_skill("math_expert")

    try:
        response = await agent.chat("What is 7 * 8? Reply with just the number.")
        print(f"Response: {response!r}")
        passed = "56" in response
        _record("Skill Lazy Loading", passed)
    except Exception as exc:
        print(f"{RED}Exception:{RESET} {exc}")
        _record("Skill Lazy Loading", False)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary() -> None:
    _header("Summary")
    total  = len(_results)
    passed = sum(1 for _, ok in _results if ok)
    failed = total - passed

    for name, ok in _results:
        label = _pass("PASS") if ok else _fail("FAIL")
        print(f"  {label}  {name}")

    print()
    if failed == 0:
        print(f"{GREEN}{BOLD}All {total} tests passed.{RESET}")
    else:
        print(
            f"{RED}{BOLD}{failed}/{total} test(s) failed.{RESET}  "
            f"{GREEN}{passed} passed.{RESET}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    cfg = _detect_config()
    print(f"\n{BOLD}neoagent e2e test suite{RESET}")
    print(f"Provider : {cfg['provider']}")
    print(f"Model    : {cfg['model']}")
    if cfg.get("base_url"):
        print(f"Base URL : {cfg['base_url']}")

    await test_simple_chat(cfg)
    await test_tool_read(cfg)
    await test_tool_bash(cfg)
    await test_multi_turn_tools(cfg)
    await test_memory_enable(cfg)
    await test_skill_lazy_load(cfg)

    _print_summary()

    # Exit with non-zero code if any test failed
    failed = sum(1 for _, ok in _results if not ok)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
