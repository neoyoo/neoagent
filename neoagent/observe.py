from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from neoagent.core.types import Message, TextBlock, ToolUseBlock

logger = logging.getLogger(__name__)

# ANSI colors (only for console)
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_MAGENTA = "\033[95m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [{len(text)} chars total]"


class Observer:
    """Framework-level observability for neoagent.

    Hooks into QueryLoop, ToolRegistry, ContextCompressor, and MemoryManager
    to log all interactions. Outputs to console and/or file.

    Usage:
        observer = Observer(log_dir=Path("logs/"), console=True)
        # ... attach to agent ...
        observer.close()
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        console: bool = True,
    ) -> None:
        self._console = console
        self._file: IO[str] | None = None
        self._log_path: Path | None = None
        self.enabled: bool = True

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
            self._log_path = log_dir / filename
            self._file = open(self._log_path, "w", encoding="utf-8")

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    def _write(self, line: str, color: str = "") -> None:
        """Write a line to console and/or file."""
        if not self.enabled:
            return
        if self._console:
            print(f"{color}{line}{_RESET}" if color else line)
        if self._file:
            # Strip ANSI for file output
            self._file.write(line + "\n")
            self._file.flush()

    def _section(self, title: str) -> None:
        bar = "─" * 60
        self._write(f"\n{bar}", _DIM)
        self._write(f"  {title}  [{_ts()}]", _BOLD)
        self._write(bar, _DIM)

    # ── Provider events ──────────────────────────────────────────

    def on_provider_request(
        self, system: str, messages: list, tools: list, turn: int
    ) -> None:
        self._section(f"PROVIDER REQUEST (turn {turn})")
        self._write(f"  System prompt: {_truncate(system, 200)}", _CYAN)
        self._write(f"  Messages: {len(messages)}", _CYAN)
        # Log last message content
        if messages:
            last = messages[-1]
            if isinstance(last.content, str):
                self._write(f"  Last message ({last.role}): {_truncate(last.content, 300)}")
            elif isinstance(last.content, list):
                block_types = [type(b).__name__ for b in last.content]
                self._write(f"  Last message ({last.role}): {block_types}")
        self._write(f"  Tools: {[t.get('name', '?') for t in tools] if tools else '(none)'}")

    def on_provider_response(
        self, content: list, stop_reason: str, input_tokens: int, output_tokens: int, turn: int
    ) -> None:
        self._section(f"PROVIDER RESPONSE (turn {turn})")
        self._write(f"  Stop reason: {stop_reason}", _GREEN)
        self._write(f"  Tokens: {input_tokens} in / {output_tokens} out")
        for block in content:
            from neoagent.core.types import TextBlock, ToolUseBlock
            if isinstance(block, TextBlock):
                self._write(f"  Text: {_truncate(block.text, 300)}")
            elif isinstance(block, ToolUseBlock):
                args = json.dumps(block.input)[:200]
                self._write(f"  Tool call: {block.name}({args})", _YELLOW)

    # ── Tool events ──────────────────────────────────────────────

    def on_tool_call(self, name: str, input_data: dict) -> None:
        args = json.dumps(input_data)[:300]
        self._write(f"  TOOL CALL: {name}({args})", _YELLOW)

    def on_tool_result(self, name: str, output: str, is_error: bool) -> None:
        status = "ERROR" if is_error else "OK"
        self._write(
            f"  TOOL RESULT [{status}]: {name} -> {_truncate(output, 300)}",
            _GREEN if not is_error else "\033[91m",
        )

    # ── Compression events ────────────────────────────────────────

    def on_compress_check(self, token_count: int, tool_tokens: int, budget: int, should: bool) -> None:
        self._write(
            f"  COMPRESS CHECK: {token_count}+{tool_tokens} tokens vs budget {budget} "
            f"(threshold {int(budget * 0.7)}) -> {'TRIGGER' if should else 'skip'}",
            _MAGENTA if should else _DIM,
        )

    def on_compress_done(self, summary: str, previous_summary: str | None) -> None:
        self._section("COMPRESSION COMPLETE")
        self._write(f"  Summary: {_truncate(summary, 400)}", _MAGENTA)
        if previous_summary:
            self._write(f"  Previous summary was: {_truncate(previous_summary, 200)}", _DIM)

    def on_compress_fallback(self, reason: str) -> None:
        self._write(f"  COMPRESS FALLBACK: {reason}", "\033[91m")

    # ── Memory events ─────────────────────────────────────────────

    def on_memory_extract_trigger(self, tool_calls: int, token_delta: int) -> None:
        self._write(
            f"  MEMORY EXTRACT: triggered (tool_calls={tool_calls}, token_delta={token_delta})",
            _MAGENTA,
        )

    def on_memory_extract_done(self, items_count: int, filenames: list[str]) -> None:
        self._write(f"  MEMORY STORED: {items_count} item(s) -> {filenames}", _GREEN)

    def on_memory_extract_skip(self, tool_calls: int, token_delta: int) -> None:
        self._write(
            f"  MEMORY SKIP: below threshold (tool_calls={tool_calls}, token_delta={token_delta})",
            _DIM,
        )

    # ── Skill events ─────────────────────────────────────────────

    def on_skill_activate(self, name: str) -> None:
        self._write(f"  SKILL ACTIVATE: {name}", _GREEN)

    def on_skill_deactivate(self, name: str) -> None:
        self._write(f"  SKILL DEACTIVATE: {name}", _DIM)

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
