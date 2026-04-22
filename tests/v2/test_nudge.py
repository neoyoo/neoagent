# tests/v2/test_nudge.py
"""TDD tests for NudgeCounter.

Spec refs: § 11.4
"""
from __future__ import annotations

import pytest

from neoagent.v2.nudge import NudgeCounter


def test_initial_no_trigger():
    nc = NudgeCounter(threshold=10)
    assert nc.should_trigger_review() is False


def test_tick_1_to_9_no_trigger():
    nc = NudgeCounter(threshold=10)
    for _ in range(9):
        nc.tick()
    assert nc.should_trigger_review() is False


def test_tick_10_triggers():
    nc = NudgeCounter(threshold=10)
    for _ in range(10):
        nc.tick()
    assert nc.should_trigger_review() is True


def test_tick_11_no_trigger():
    nc = NudgeCounter(threshold=10)
    for _ in range(11):
        nc.tick()
    assert nc.should_trigger_review() is False


def test_tick_20_triggers():
    nc = NudgeCounter(threshold=10)
    for _ in range(20):
        nc.tick()
    assert nc.should_trigger_review() is True


def test_reset_clears_trigger():
    nc = NudgeCounter(threshold=10)
    for _ in range(10):
        nc.tick()
    assert nc.should_trigger_review() is True
    nc.reset()
    assert nc.should_trigger_review() is False
    assert nc._count == 0


def test_custom_threshold():
    nc = NudgeCounter(threshold=5)
    for _ in range(4):
        nc.tick()
    assert nc.should_trigger_review() is False
    nc.tick()
    assert nc.should_trigger_review() is True
    nc.tick()
    assert nc.should_trigger_review() is False
    for _ in range(4):
        nc.tick()
    assert nc.should_trigger_review() is True
