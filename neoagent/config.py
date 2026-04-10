from __future__ import annotations
from dataclasses import dataclass

@dataclass
class NeoAgentConfig:
    api_key: str
    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 30
    context_budget: int = 0
    max_result_size: int = 50000
