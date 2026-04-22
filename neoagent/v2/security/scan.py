# neoagent/v2/security/scan.py
"""Injection detection for external content.
Returns ScanResult with should_block flag and matched patterns.

spec § 18.5 (11 injection regex patterns, lines 2984-3005)
spec § 18.6 (9 zero-width unicode codepoints, lines 3006-3026)
"""

import re
from dataclasses import dataclass, field

# 11 injection regex patterns (spec § 18.5)
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?i)ignore\s+(all\s+|previous\s+|above\s+)?(instructions?|prompts?|rules?)"),
        "ignore_instructions",
    ),
    (
        re.compile(r"(?i)(forget|disregard)\s+(everything|all|prior|previous)"),
        "forget_prior",
    ),
    (
        re.compile(r"(?i)you\s+are\s+now\s+(a|an)?\s*\w+"),
        "role_hijack_now",
    ),
    (
        re.compile(r"(?i)new\s+(instructions?|rules?|system\s+prompt)[:：]"),
        "new_instructions",
    ),
    (
        re.compile(r"(?i)(system|admin|root)\s*[:：]"),
        "fake_system_role",
    ),
    (
        re.compile(r"(?i)act\s+as\s+(if\s+)?you\s+(are|have\s+no)"),
        "act_as_jailbreak",
    ),
    (
        re.compile(r"(?i)reveal\s+(your|the)\s+(system\s+prompt|instructions?|rules?)"),
        "reveal_prompt",
    ),
    (
        re.compile(r"(?i)(execute|run|eval)\s+(the\s+)?(following|this)\s+(code|command|script)"),
        "execute_attempt",
    ),
    (
        re.compile(r"(?i)developer\s+mode"),
        "developer_mode",
    ),
    (
        re.compile(r"(?i)DAN\s+mode|do\s+anything\s+now"),
        "dan_jailbreak",
    ),
    (
        re.compile(r"(?i)<\s*/?\s*(system|assistant|user)\s*>"),
        "fake_role_tag",
    ),
]

# 9 zero-width / invisible unicode codepoints (spec § 18.6)
_HIDDEN_UNICODE: dict[str, str] = {
    "​": "ZERO_WIDTH_SPACE",
    "‌": "ZERO_WIDTH_NON_JOINER",
    "‍": "ZERO_WIDTH_JOINER",
    "⁠": "WORD_JOINER",
    "﻿": "BOM/ZERO_WIDTH_NO_BREAK_SPACE",
    "­": "SOFT_HYPHEN",
    "‮": "RIGHT_TO_LEFT_OVERRIDE",
    "‭": "LEFT_TO_RIGHT_OVERRIDE",
    "᠎": "MONGOLIAN_VOWEL_SEPARATOR",
}


@dataclass
class ScanResult:
    should_block: bool
    matched_patterns: list[str] = field(default_factory=list)
    hidden_unicode: list[str] = field(default_factory=list)


def scan_source_content(text: str) -> ScanResult:
    """Scan external content for prompt injection patterns.

    Returns ScanResult with:
    - should_block: True if any injection pattern matches
    - matched_patterns: list of matched pattern names
    - hidden_unicode: list of detected zero-width/invisible codepoint names
    """
    matched: list[str] = []
    for pattern, name in _INJECTION_PATTERNS:
        if pattern.search(text):
            matched.append(name)

    hidden: list[str] = []
    for cp, name in _HIDDEN_UNICODE.items():
        if cp in text:
            hidden.append(name)

    # Conservative default: only injection pattern matches trigger block.
    # Hidden unicode is flagged but does not block by itself (can be configured
    # by callers to also block when hidden_unicode is non-empty).
    should_block = bool(matched)
    return ScanResult(
        should_block=should_block,
        matched_patterns=matched,
        hidden_unicode=hidden,
    )
