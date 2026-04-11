from __future__ import annotations
from dataclasses import dataclass, field
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
    memory_dir: Path | None = None
    memory_project_key: str | None = None
