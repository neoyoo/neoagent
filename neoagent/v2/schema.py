# neoagent/v2/schema.py
"""Canonical dataclasses for neoagent v2.

Spec refs:
  § 2.1-2.7  (lines 149-436)
  § 2.3a     (lines 270-296)
  § 10.2     (lines 1597-1630)
  § 15.1     Layer enum (lines 2487+)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal


@dataclass
class WorkingMemory:
    """spec § 2.3, lines 236-269 + § 2.3a lines 270-296"""
    session_id: str
    version: int
    at_turn: int
    goal: str                                   # immutable after framework_init
    constraints_and_preferences: list[str]      # list[str], prefix c01:
    progress: str
    key_decisions: list[str]                    # prefix d01:
    relevant_files: list[str]                   # prefix f01:
    next_steps: list[str]                       # prefix n01:
    critical_context: str
    updated_by: Literal["llm_tool", "framework_init", "framework_compression"]
    updated_at: datetime


@dataclass
class BatchMember:
    """spec § 10.2, lines 1619-1625"""
    id: str                                     # "m5" / "t7", must be original msg_id
    role: Literal["user", "assistant", "tool"]
    preview: str                                # compressor-generated, 20-40 chars


@dataclass
class CompressionDelta:
    """spec § 10.2, lines 1604-1630"""
    batch_summary: str                          # 7-section structured text
    batch_members: list[BatchMember]
    working_memory_delta: list[dict]            # [{field, op, value, item_id?}]


@dataclass
class CompressionContext:
    """spec § 10.2, lines 1597-1603"""
    session_id: str
    messages: list
    previous_batches: list
    previous_wm: dict
    trigger: Literal["token_threshold", "message_count", "window_ratio"]


@dataclass
class Batch:
    """spec § 2.5, lines 325-353"""
    session_id: str
    batch_id: str                               # "cm_1" (session-local)
    turns_from: int
    turns_to: int
    time_from: datetime
    time_to: datetime
    summary: str
    members: list[BatchMember]
    trigger: str
    created_at: datetime


@dataclass
class MemoryEntry:
    """spec § 2.6, lines 354-393"""
    user_id: str
    memory_id: str
    type: str                                   # "preference" / "fact" / "goal" / ...
    category: str | None
    content: str
    confidence: float = 0.5
    evidence: dict | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_reinforced_at: datetime | None = None
    usage_count: int = 0


class Layer(str, Enum):
    """spec § 15.1 + decision G, flat 7 layers"""
    IDENTITY = "identity"
    PERSISTENT_MEMORY = "persistent_memory"
    CAPABILITIES = "capabilities"
    SECURITY = "security"
    WORKING_MEMORY = "working_memory"
    COMPRESSED_HISTORY = "compressed_history"
    MEMORY_CONTEXT = "memory_context"
