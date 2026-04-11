from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass
class NeoAgentConfig:
    api_key: str
    model: str = "claude-sonnet-4-20250514"
    provider: Literal["anthropic", "openai"] = "anthropic"
    max_turns: int = 30
    context_budget: int = 0
    max_result_size: int = 50000
