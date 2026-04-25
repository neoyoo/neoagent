from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from neoagent.v2.abc import (
        CompressionStrategy,
        MemoryProvider,
        MemoryReviewStrategy,
        WorkingMemoryStore,
    )
    from neoagent.v2.compressed_store import CompressedMessageStore

@dataclass
class NeoAgentConfig:
    api_key: str
    model: str = "claude-sonnet-4-20250514"
    provider: Literal["anthropic", "openai"] = "anthropic"
    base_url: str | None = None
    max_turns: int = 30
    context_budget: int = 0
    max_result_size: int = 50000
    auto_approve_tools: bool = False
    memory_dir: Path | None = None
    memory_project_key: str | None = None
    session_dir: Path | None = None
    system_prompt: str | None = None

    # ── v2 optional injections ────────────────────────────────────────────────
    # All Optional; default None/True — does not break existing construction paths.
    wm_store: "WorkingMemoryStore | None" = field(default=None)
    compression_strategy: "CompressionStrategy | None" = field(default=None)
    memory_review_strategy: "MemoryReviewStrategy | None" = field(default=None)
    memory_provider: "MemoryProvider | None" = field(default=None)
    compressed_message_store: "CompressedMessageStore | None" = field(default=None)
    enable_source_wrap: bool = True
    enable_security_prompt_blocks: bool = True

    def __repr__(self) -> str:
        fields = []
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if f == "api_key" and val:
                val = val[:4] + "***" if len(val) > 4 else "***"
            fields.append(f"{f}={val!r}")
        return f"{self.__class__.__name__}({', '.join(fields)})"
