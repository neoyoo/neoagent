from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from neoagent.eval.metrics import MetricsCollector, SessionMetrics
from neoagent.events import (
    ProviderRequestEvent,
    ProviderResponseEvent,
    ToolCallEvent,
    TurnCompleteEvent,
)

if TYPE_CHECKING:
    from neoagent.agent import NeoAgent
    from neoagent.core.types import Message, ConversationResult


@dataclass
class EvalCase:
    """A single evaluation case: input messages + assertion function."""

    name: str
    messages: list["Message"]
    assertion: Callable[["ConversationResult"], bool | Awaitable[bool]]
    max_turns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalCaseResult:
    """Result of running a single EvalCase."""

    name: str
    passed: bool
    result: "ConversationResult | None"
    error: str | None
    metrics: SessionMetrics


@dataclass
class EvalReport:
    """Aggregated report for a batch of EvalCase runs."""

    total: int
    passed: int
    failed: int
    cases: list[EvalCaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that passed. Returns 0.0 if total == 0."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total


class EvalRunner:
    """Runs a batch of EvalCase objects against a NeoAgent instance.

    Each case gets its own fresh MetricsCollector so metrics are isolated.
    Agent exceptions and assertion exceptions are both caught and recorded in
    EvalCaseResult.error — they never abort the remaining cases.

    Usage::

        runner = EvalRunner(agent)
        report = await runner.run(cases)
        print(f"Pass rate: {report.pass_rate:.0%}")
    """

    def __init__(self, agent: "NeoAgent") -> None:
        self._agent = agent

    async def run(self, cases: list[EvalCase]) -> EvalReport:
        """Run all cases sequentially and return an EvalReport."""
        results: list[EvalCaseResult] = []

        for case in cases:
            result = await self._run_case(case)
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        return EvalReport(
            total=len(results),
            passed=passed,
            failed=failed,
            cases=results,
        )

    async def _run_case(self, case: EvalCase) -> EvalCaseResult:
        """Run a single EvalCase with isolated metrics collection."""
        collector = MetricsCollector()
        bus = self._agent.event_bus

        # Subscribe all four handlers
        bus.subscribe(ProviderRequestEvent, collector._on_provider_request)
        bus.subscribe(ProviderResponseEvent, collector._on_provider_response)
        bus.subscribe(ToolCallEvent, collector._on_tool_call)
        bus.subscribe(TurnCompleteEvent, collector._on_turn_complete)

        conv_result: "ConversationResult | None" = None
        passed = False
        error: str | None = None

        try:
            # Run the agent
            try:
                conv_result = await self._agent.run(
                    messages=list(case.messages),
                    max_turns=case.max_turns,
                )
            except Exception as exc:
                error = f"agent raised: {type(exc).__name__}: {exc}"
                return EvalCaseResult(
                    name=case.name,
                    passed=False,
                    result=None,
                    error=error,
                    metrics=collector.get_session_metrics(),
                )

            # Run the assertion
            try:
                import inspect
                assertion_result = case.assertion(conv_result)
                if inspect.isawaitable(assertion_result):
                    assertion_result = await assertion_result
                passed = bool(assertion_result)
            except Exception as exc:
                error = f"assertion raised: {type(exc).__name__}: {exc}"
                passed = False

        finally:
            # Always unsubscribe — symmetric cleanup
            bus.unsubscribe(ProviderRequestEvent, collector._on_provider_request)
            bus.unsubscribe(ProviderResponseEvent, collector._on_provider_response)
            bus.unsubscribe(ToolCallEvent, collector._on_tool_call)
            bus.unsubscribe(TurnCompleteEvent, collector._on_turn_complete)

        return EvalCaseResult(
            name=case.name,
            passed=passed,
            result=conv_result,
            error=error,
            metrics=collector.get_session_metrics(),
        )
