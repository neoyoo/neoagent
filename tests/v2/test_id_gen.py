# tests/v2/test_id_gen.py
"""TDD tests for SessionIdGenerator.

Spec refs: § 8.2 (lines 1308-1334)
"""
from __future__ import annotations

import pytest

from neoagent.v2.id_gen import SessionIdGenerator


def test_initial_state_no_ids():
    gen = SessionIdGenerator()
    # Fresh instance — internal counters should all be 0
    assert gen._msg_counter == 0
    assert gen._batch_counter == 0
    assert gen._decision_counter == 0


def test_next_msg_id_sequence():
    gen = SessionIdGenerator()
    assert gen.next_msg_id() == "m1"
    assert gen.next_msg_id() == "m2"
    assert gen.next_msg_id() == "m3"


def test_next_batch_id_sequence():
    gen = SessionIdGenerator()
    assert gen.next_batch_id() == "cm_1"
    assert gen.next_batch_id() == "cm_2"


def test_next_decision_id_sequence():
    gen = SessionIdGenerator()
    assert gen.next_decision_id() == "d1"
    assert gen.next_decision_id() == "d2"


def test_counters_are_independent():
    gen = SessionIdGenerator()
    # Advance msg counter 3x
    gen.next_msg_id()
    gen.next_msg_id()
    gen.next_msg_id()
    # Advance batch counter 1x
    gen.next_batch_id()
    # Decision counter still at 0
    assert gen.next_decision_id() == "d1"
    # Batch counter is at 1, next should be cm_2
    assert gen.next_batch_id() == "cm_2"
    # Msg counter is at 3, next should be m4
    assert gen.next_msg_id() == "m4"
