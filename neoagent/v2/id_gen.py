# neoagent/v2/id_gen.py
"""Session-local monotonic ID generator for stable, LLM-readable identifiers.

Spec refs: § 8.2 (lines 1308-1334)
"""
from __future__ import annotations


class SessionIdGenerator:
    """Session-local monotonic counters for stable, LLM-readable ids.

    Each counter is independent.  Counters start at 0 and increment
    before returning, so the first id of each type is always index 1.

    Examples::

        gen = SessionIdGenerator()
        gen.next_msg_id()      # "m1"
        gen.next_msg_id()      # "m2"
        gen.next_batch_id()    # "cm_1"
        gen.next_decision_id() # "d1"
    """

    def __init__(self) -> None:
        self._msg_counter: int = 0
        self._batch_counter: int = 0
        self._decision_counter: int = 0

    def next_msg_id(self) -> str:
        """Return next message id: ``"m1"``, ``"m2"``, …"""
        self._msg_counter += 1
        return f"m{self._msg_counter}"

    def next_batch_id(self) -> str:
        """Return next compression-batch id: ``"cm_1"``, ``"cm_2"``, …"""
        self._batch_counter += 1
        return f"cm_{self._batch_counter}"

    def next_decision_id(self) -> str:
        """Return next decision id: ``"d1"``, ``"d2"``, …"""
        self._decision_counter += 1
        return f"d{self._decision_counter}"
