from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from neoagent.eval.runner import EvalCase, EvalCaseResult, EvalReport, EvalRunner
from neoagent.eval.metrics import SessionMetrics
from neoagent.core.types import ConversationResult, Message, Turn, TextBlock
from neoagent.events import EventBus


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_agent(result: ConversationResult | None = None, *, raise_exc: Exception | None = None) -> MagicMock:
    """Create a mock NeoAgent with a controllable run() coroutine and real EventBus."""
    agent = MagicMock()
    agent.event_bus = EventBus()

    if raise_exc is not None:
        agent.run = AsyncMock(side_effect=raise_exc)
    else:
        if result is None:
            result = _make_result("ok")
        agent.run = AsyncMock(return_value=result)

    return agent


def _make_result(text: str = "ok") -> ConversationResult:
    turn = Turn(
        response=Message(role="assistant", content=[TextBlock(text=text)]),
        stop_reason="end_turn",
    )
    return ConversationResult(turns=[turn], reason="completed")


def _make_message(text: str = "hello") -> Message:
    return Message(role="user", content=text)


def _true_assertion(result: ConversationResult) -> bool:
    return True


def _false_assertion(result: ConversationResult) -> bool:
    return False


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_eval_runner_all_pass():
    """All cases pass → report has 100% pass rate."""
    agent = _make_agent()
    runner = EvalRunner(agent)

    cases = [
        EvalCase(name="case1", messages=[_make_message()], assertion=_true_assertion),
        EvalCase(name="case2", messages=[_make_message()], assertion=_true_assertion),
    ]

    report = await runner.run(cases)

    assert report.total == 2
    assert report.passed == 2
    assert report.failed == 0
    assert report.pass_rate == 1.0
    assert all(r.passed for r in report.cases)
    assert all(r.error is None for r in report.cases)


@pytest.mark.asyncio
async def test_eval_runner_all_fail():
    """All cases fail (assertion returns False) → pass_rate == 0.0."""
    agent = _make_agent()
    runner = EvalRunner(agent)

    cases = [
        EvalCase(name="f1", messages=[_make_message()], assertion=_false_assertion),
        EvalCase(name="f2", messages=[_make_message()], assertion=_false_assertion),
    ]

    report = await runner.run(cases)

    assert report.total == 2
    assert report.passed == 0
    assert report.failed == 2
    assert report.pass_rate == 0.0


@pytest.mark.asyncio
async def test_eval_runner_handles_agent_exception():
    """If agent.run() raises, the case is failed and error recorded."""
    exc = RuntimeError("provider down")
    agent = _make_agent(raise_exc=exc)
    runner = EvalRunner(agent)

    cases = [
        EvalCase(name="bad", messages=[_make_message()], assertion=_true_assertion),
    ]

    report = await runner.run(cases)

    assert report.total == 1
    assert report.passed == 0
    assert report.failed == 1
    case_result = report.cases[0]
    assert not case_result.passed
    assert case_result.result is None
    assert "agent raised" in case_result.error
    assert "RuntimeError" in case_result.error


@pytest.mark.asyncio
async def test_eval_runner_handles_assertion_exception():
    """If assertion raises, the case is failed with 'assertion raised' in error."""
    agent = _make_agent()
    runner = EvalRunner(agent)

    def bad_assertion(result: ConversationResult) -> bool:
        raise ValueError("bad assertion logic")

    cases = [
        EvalCase(name="assert_err", messages=[_make_message()], assertion=bad_assertion),
    ]

    report = await runner.run(cases)

    assert report.passed == 0
    case_result = report.cases[0]
    assert not case_result.passed
    assert "assertion raised" in case_result.error
    assert "ValueError" in case_result.error
    # result should still be populated (agent succeeded)
    assert case_result.result is not None


@pytest.mark.asyncio
async def test_eval_runner_collects_metrics_per_case():
    """Each case result carries a SessionMetrics instance (not None)."""
    agent = _make_agent()
    runner = EvalRunner(agent)

    cases = [
        EvalCase(name="m1", messages=[_make_message()], assertion=_true_assertion),
        EvalCase(name="m2", messages=[_make_message()], assertion=_true_assertion),
    ]

    report = await runner.run(cases)

    for case_result in report.cases:
        assert isinstance(case_result.metrics, SessionMetrics)


@pytest.mark.asyncio
async def test_eval_runner_unsubscribes_after_case():
    """After each case, MetricsCollector handlers are unsubscribed from EventBus."""
    agent = _make_agent()
    runner = EvalRunner(agent)

    cases = [
        EvalCase(name="unsub", messages=[_make_message()], assertion=_true_assertion),
    ]

    await runner.run(cases)

    # After the run, the EventBus should have no remaining handlers for the
    # four metric event types.
    from neoagent.events import (
        ProviderRequestEvent,
        ProviderResponseEvent,
        ToolCallEvent,
        TurnCompleteEvent,
    )
    bus = agent.event_bus
    for event_type in [ProviderRequestEvent, ProviderResponseEvent, ToolCallEvent, TurnCompleteEvent]:
        assert bus._handlers.get(event_type, []) == [], (
            f"Handlers for {event_type.__name__} were not cleaned up"
        )


@pytest.mark.asyncio
async def test_eval_runner_pass_rate():
    """pass_rate is correctly calculated as passed / total."""
    agent = _make_agent()
    runner = EvalRunner(agent)

    cases = [
        EvalCase(name="p1", messages=[_make_message()], assertion=_true_assertion),
        EvalCase(name="p2", messages=[_make_message()], assertion=_true_assertion),
        EvalCase(name="f1", messages=[_make_message()], assertion=_false_assertion),
        EvalCase(name="f2", messages=[_make_message()], assertion=_false_assertion),
    ]

    report = await runner.run(cases)

    assert report.total == 4
    assert report.passed == 2
    assert report.failed == 2
    assert report.pass_rate == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_eval_runner_mixed_results():
    """Agent exception on one case doesn't prevent other cases from running."""
    exc = RuntimeError("flaky")
    # First call raises, second returns success
    agent = MagicMock()
    agent.event_bus = EventBus()
    agent.run = AsyncMock(side_effect=[exc, _make_result("done")])

    runner = EvalRunner(agent)

    cases = [
        EvalCase(name="fails", messages=[_make_message()], assertion=_true_assertion),
        EvalCase(name="passes", messages=[_make_message()], assertion=_true_assertion),
    ]

    report = await runner.run(cases)

    assert report.total == 2
    assert report.passed == 1
    assert report.failed == 1
    assert report.cases[0].name == "fails"
    assert not report.cases[0].passed
    assert report.cases[1].name == "passes"
    assert report.cases[1].passed


@pytest.mark.asyncio
async def test_eval_runner_empty_cases():
    """Running with empty case list returns a report with zeros and 0.0 pass_rate."""
    agent = _make_agent()
    runner = EvalRunner(agent)

    report = await runner.run([])

    assert report.total == 0
    assert report.passed == 0
    assert report.failed == 0
    assert report.pass_rate == 0.0
    assert report.cases == []


def test_eval_case_dataclass_fields():
    """EvalCase has the expected fields and defaults."""
    msg = _make_message("hi")

    case = EvalCase(name="my_case", messages=[msg], assertion=_true_assertion)
    assert case.name == "my_case"
    assert case.messages == [msg]
    assert case.assertion is _true_assertion
    assert case.max_turns is None
    assert case.metadata == {}

    case2 = EvalCase(
        name="c2",
        messages=[msg],
        assertion=_false_assertion,
        max_turns=5,
        metadata={"tag": "smoke"},
    )
    assert case2.max_turns == 5
    assert case2.metadata == {"tag": "smoke"}


def test_eval_report_pass_rate_zero_total():
    """EvalReport.pass_rate returns 0.0 when total == 0 (no division by zero)."""
    report = EvalReport(total=0, passed=0, failed=0, cases=[])
    assert report.pass_rate == 0.0
