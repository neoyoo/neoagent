# neoagent/v2/nudge.py
"""Turn-based memory-review nudge counter.

Spec refs: § 11.4
"""
from __future__ import annotations


class NudgeCounter:
    """Tracks turn count; triggers memory review every N turns.

    Usage::

        nc = NudgeCounter(threshold=10)
        nc.tick()                       # call once per completed turn
        nc.should_trigger_review()      # True when count is a multiple of threshold
        nc.reset()                      # back to zero (new session)
    """

    def __init__(self, threshold: int = 10) -> None:
        self.threshold = threshold
        self._count: int = 0

    def tick(self) -> None:
        """Increment the turn counter by one."""
        self._count += 1

    def should_trigger_review(self) -> bool:
        """Return True when count is a non-zero multiple of *threshold*."""
        return self._count > 0 and self._count % self.threshold == 0

    def reset(self) -> None:
        """Reset counter to zero (use when starting a new session)."""
        self._count = 0
