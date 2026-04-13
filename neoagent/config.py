from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

    def __repr__(self) -> str:
        fields = []
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if f == "api_key" and val:
                val = val[:4] + "***" if len(val) > 4 else "***"
            fields.append(f"{f}={val!r}")
        return f"{self.__class__.__name__}({', '.join(fields)})"
