from __future__ import annotations
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

from neoagent.core.types import TextBlock, ToolResultBlock, ToolUseBlock

if TYPE_CHECKING:
    from neoagent.core.types import Message

logger = logging.getLogger(__name__)

# ANSI colors
_CYAN    = "\033[96m"
_GREEN   = "\033[92m"
_YELLOW  = "\033[93m"
_BLUE    = "\033[94m"
_MAGENTA = "\033[95m"
_RED     = "\033[91m"
_DIM     = "\033[2m"
_BOLD    = "\033[1m"
_RESET   = "\033[0m"

_ROLE_COLOR = {
    "SYSTEM":    _CYAN,
    "USER":      _GREEN,
    "ASSISTANT": _YELLOW,
    "TOOL":      _BLUE,
}


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [{len(text)} chars total]"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _indent(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


class Observer:
    """Conversation-style observability for neoagent.

    Outputs each message with a clear role label:
      [SYSTEM]    — system prompt (shown once)
      [USER]      — message submitted to the LLM
      [ASSISTANT] — LLM text response or tool call decision
      [TOOL]      — tool execution result

    Usage:
        observer = Observer(log_dir=Path("logs/"), console=True)
        subscriber = ObserverSubscriber(observer)
        subscriber.attach(agent.event_bus)
        ...
        subscriber.detach(agent.event_bus)
        observer.close()
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        console: bool = True,
        name: str | None = None,
    ) -> None:
        self._console = console
        self._name = name
        self._file: IO[str] | None = None
        self._log_path: Path | None = None
        self.enabled: bool = True
        self._system_shown = False

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"-{name}" if name else ""
            filename = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}{suffix}.log"
            self._log_path = log_dir / filename
            self._file = open(self._log_path, "w", encoding="utf-8")

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    # ── Internal write helpers ───────────────────────────────────────────────

    def _write(self, line: str, color: str = "", full: str | None = None) -> None:
        if not self.enabled:
            return
        console_line = f"{color}{line}{_RESET}" if color else line
        file_line = full if full is not None else line
        if self._console:
            print(console_line)
        if self._file:
            self._file.write(_strip_ansi(file_line) + "\n")
            self._file.flush()

    def _role_block(self, role: str, body: str, suffix: str = "",
                    full_body: str | None = None) -> None:
        label = f"[{role}]" if not suffix else f"[{role} → {suffix}]"
        prefix = f"{_DIM}[{self._name}]{_RESET} " if self._name else ""
        file_prefix = f"[{self._name}] " if self._name else ""
        color = _ROLE_COLOR.get(role, "")
        header_console = f"{prefix}{color}{_BOLD}{label}{_RESET}  {_DIM}{_ts()}{_RESET}"
        header_file = f"{file_prefix}{label}  {_ts()}"
        indented = _indent(body)
        full_indented = _indent(full_body or body)
        self._write(header_console, full=header_file)
        self._write(indented, full=full_indented)
        self._write("")

    def _divider(self) -> None:
        self._write(f"{_DIM}{'─' * 60}{_RESET}")

    # ── Provider events ──────────────────────────────────────────────────────

    def on_provider_request(
        self, system: str, messages: list, tools: list, turn: int
    ) -> None:
        self._divider()

        # System prompt — once per session
        if not self._system_shown and system:
            self._system_shown = True
            self._role_block("SYSTEM", _truncate(system, 300), full_body=system)

        if not messages:
            return

        last: Message = messages[-1]
        content = last.content

        if isinstance(content, str):
            # Plain text user message
            self._role_block("USER", content)
        elif isinstance(content, list):
            # Check what blocks are in this message
            tool_result_blocks = [b for b in content if isinstance(b, ToolResultBlock)]
            text_blocks = [b for b in content if isinstance(b, TextBlock)]

            if tool_result_blocks:
                # Tool results injected back — skip, already shown by on_tool_result
                pass
            elif text_blocks:
                body = "\n".join(b.text for b in text_blocks)
                self._role_block("USER", body)

    def on_provider_response(
        self, content: list, stop_reason: str, input_tokens: int, output_tokens: int, turn: int
    ) -> None:
        text_parts: list[str] = []
        tool_calls: list[str] = []

        for block in content:
            if isinstance(block, TextBlock) and block.text.strip():
                text_parts.append(block.text.strip())
            elif isinstance(block, ToolUseBlock):
                try:
                    args = json.dumps(dict(block.input), ensure_ascii=False, indent=2)
                except Exception:
                    args = str(block.input)
                tool_calls.append(f"▶ {block.name}\n{_indent(args, 4)}")

        lines: list[str] = []
        if text_parts:
            lines.extend(text_parts)
        if tool_calls:
            lines.extend(tool_calls)
        if not lines:
            lines.append("(no content)")

        token_note = f"{_DIM}[tokens: {input_tokens} in / {output_tokens} out]{_RESET}"
        body_console = "\n".join(lines) + f"\n{token_note}"
        body_file    = "\n".join(lines) + f"\n[tokens: {input_tokens} in / {output_tokens} out]"

        self._role_block("ASSISTANT", body_console, suffix=stop_reason, full_body=body_file)

    # ── Tool events ──────────────────────────────────────────────────────────

    def on_tool_call(self, name: str, input_data: dict) -> None:
        try:
            args = json.dumps(dict(input_data), ensure_ascii=False, indent=2)
        except Exception:
            args = str(input_data)
        self._role_block("TOOL", f"{name}\n{_indent(args, 4)}", suffix="call")

    def on_tool_result(self, name: str, output: str, is_error: bool) -> None:
        status = "✗ ERROR" if is_error else "✓"
        self._role_block(
            "TOOL",
            f"{name} {status}\n{_truncate(output, 400)}",
            full_body=f"{name} {status}\n{output}",
        )

    # ── Compression events ────────────────────────────────────────────────────

    def on_compress_check(self, token_count: int, tool_tokens: int, budget: int, should: bool) -> None:
        if should:
            line = f"[COMPRESS TRIGGER] {token_count}+{tool_tokens} tokens"
            self._write(f"{_MAGENTA}{line}{_RESET}", full=line)

    def on_compress_done(self, summary: str, previous_summary: str | None) -> None:
        self._write(f"{_MAGENTA}[COMPRESS DONE]{_RESET}")
        self._write(
            _indent(_truncate(summary, 300)), _DIM,
            full=_indent(summary),
        )
        self._write("")

    def on_compress_fallback(self, reason: str) -> None:
        self._write(f"{_RED}[COMPRESS FALLBACK] {reason}{_RESET}")

    # ── Memory events ─────────────────────────────────────────────────────────

    def on_memory_extract_done(self, items_count: int, filenames: list[str]) -> None:
        self._write(f"{_DIM}[MEMORY] stored {items_count} item(s): {filenames}{_RESET}")

    def on_memory_extract_skip(self, tool_calls: int, token_delta: int) -> None:
        pass  # not worth printing on every turn

    # ── Tool result lifecycle ─────────────────────────────────────────────────

    def on_tool_result_freed(
        self, tool_use_id: str, tool_name: str, size: int, preview: str, reason: str
    ) -> None:
        prefix = f"{_DIM}[{self._name}]{_RESET} " if self._name else ""
        file_prefix = f"[{self._name}] " if self._name else ""
        console = (
            f"{prefix}{_MAGENTA}[FREED]{_RESET} {_DIM}{_ts()}{_RESET} "
            f"{tool_name} id={tool_use_id} size={size}B reason={reason}\n"
            f"  {_DIM}preview={preview!r}{_RESET}"
        )
        file_line = (
            f"{file_prefix}[FREED] {_ts()} "
            f"{tool_name} id={tool_use_id} size={size}B reason={reason}\n"
            f"  preview={preview!r}"
        )
        self._write(console, full=file_line)

    # ── Skill events ──────────────────────────────────────────────────────────

    def on_skill_activate(self, name: str) -> None:
        self._write(f"{_DIM}[SKILL +] {name}{_RESET}")

    def on_skill_deactivate(self, name: str) -> None:
        self._write(f"{_DIM}[SKILL -] {name}{_RESET}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __enter__(self) -> "Observer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
