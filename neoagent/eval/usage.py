from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neoagent.events import ProviderResponseEvent


@dataclass
class ModelUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0


class UsageTracker:
    def __init__(self) -> None:
        self._per_model: dict[str, ModelUsage] = {}
        self._current_model: str = "unknown"

    def _handle_response(self, event: "ProviderResponseEvent") -> None:
        model = self._current_model
        if model not in self._per_model:
            self._per_model[model] = ModelUsage(model=model)
        usage = self._per_model[model]
        usage.input_tokens += event.input_tokens
        usage.output_tokens += event.output_tokens
        usage.request_count += 1

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self._per_model.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self._per_model.values())

    @property
    def per_model_usage(self) -> dict[str, ModelUsage]:
        return dict(self._per_model)

    def reset(self) -> None:
        self._per_model.clear()
