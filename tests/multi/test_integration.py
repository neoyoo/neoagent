"""Integration tests: end-to-end multi-agent flow.

All LLM calls are mocked.  Real framework components
(TaskTracker, WorkerPool, EventBus, Orchestrator, etc.) are used.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neoagent.config import NeoAgentConfig
from neoagent.core.types import ConversationResult, Message, TextBlock, Turn
from neoagent.multi.orchestrator import Orchestrator
from neoagent.multi.task import Task, TaskResult, TokenUsage
from neoagent.multi.tools.cancel_task import CancelTaskInput, CancelTaskTool
from neoagent.multi.tools.delegate_task import DelegateTaskInput, DelegateTaskTool
from neoagent.multi.tools import _emit_complete_event, _emit_dispatch_event
from neoagent.multi.tools.spawn_worker import SpawnWorkerInput, SpawnWorkerTool
from neoagent.multi.worker import WorkerCard, _create_worker_agent


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_config() -> NeoAgentConfig:
    return NeoAgentConfig(api_key="sk-test")


def _mock_provider() -> MagicMock:
    p = MagicMock()
    p.get_context_window.return_value = 200_000
    return p


def _make_completed_conversation(text: str = "done") -> ConversationResult:
    return ConversationResult(
        turns=[
            Turn(
                response=Message(
                    role="assistant",
                    content=[TextBlock(text=text)],
                ),
                tool_calls=[],
                tool_results=[],
                stop_reason="end_turn",
            )
        ],
        reason="completed",
    )


def _make_worker_card(name: str = "analyst", instruction: str = "You analyze code.") -> WorkerCard:
    return WorkerCard(
        name=name,
        description=f"{name} worker",
        instruction=instruction,
        tags=(),
        model=None,
        tools=(),
    )


# ---------------------------------------------------------------------------
# Full flow: delegate_task
# ---------------------------------------------------------------------------


class TestDelegateTaskFlow:
    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_delegate_task_full_flow(self, mock_prov: MagicMock) -> None:
        """Orchestrator → delegate_task → mock worker runs → TaskResult returned."""
        mock_prov.return_value = _mock_provider()

        orch = Orchestrator(_make_config())
        card = _make_worker_card("analyst")
        orch.register_worker(card)

        tool = DelegateTaskTool(orchestrator=orch, depth=0)

        with patch("neoagent.multi.tools.delegate_task._create_worker_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(
                return_value=_make_completed_conversation("Analysis: no issues found.")
            )
            mock_create.return_value = mock_agent

            inp = DelegateTaskInput(worker_name="analyst", instruction="Analyze auth.py")
            result = await tool.execute(inp)

        assert result.is_error is False
        assert "completed" in result.output
        assert "Analysis: no issues found." in result.output

    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_delegate_task_unknown_worker(self, mock_prov: MagicMock) -> None:
        """delegate_task with unknown worker_name returns error with list_workers hint."""
        mock_prov.return_value = _mock_provider()
        orch = Orchestrator(_make_config())

        tool = DelegateTaskTool(orchestrator=orch, depth=0)
        inp = DelegateTaskInput(worker_name="ghost-worker", instruction="Do something")
        result = await tool.execute(inp)

        assert result.is_error is True
        assert "ghost-worker" in result.output
        assert "list_workers" in result.output

    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_delegate_task_worker_failure_returns_error_result(
        self, mock_prov: MagicMock
    ) -> None:
        """When worker.run() raises, the ToolResult reports failure."""
        mock_prov.return_value = _mock_provider()

        orch = Orchestrator(_make_config())
        orch.register_worker(_make_worker_card("failing-worker"))

        tool = DelegateTaskTool(orchestrator=orch, depth=0)

        with patch("neoagent.multi.tools.delegate_task._create_worker_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=RuntimeError("worker crashed"))
            mock_create.return_value = mock_agent

            inp = DelegateTaskInput(
                worker_name="failing-worker", instruction="Do work that crashes"
            )
            result = await tool.execute(inp)

        # _run_worker catches the exception → status=failed → _format_task_result is_error=True
        assert result.is_error is True
        assert "failed" in result.output

    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_delegate_task_emits_dispatch_and_complete_events(
        self, mock_prov: MagicMock
    ) -> None:
        """TaskDispatchEvent and TaskCompleteEvent are emitted around execution."""
        mock_prov.return_value = _mock_provider()
        from neoagent.events import TaskCompleteEvent, TaskDispatchEvent

        orch = Orchestrator(_make_config())
        orch.register_worker(_make_worker_card("evt-worker"))

        dispatch_events: list[TaskDispatchEvent] = []
        complete_events: list[TaskCompleteEvent] = []
        orch._event_bus.subscribe(TaskDispatchEvent, dispatch_events.append)
        orch._event_bus.subscribe(TaskCompleteEvent, complete_events.append)

        tool = DelegateTaskTool(orchestrator=orch, depth=0)
        with patch("neoagent.multi.tools.delegate_task._create_worker_agent") as mc:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(
                return_value=_make_completed_conversation("done")
            )
            mc.return_value = mock_agent

            inp = DelegateTaskInput(worker_name="evt-worker", instruction="Do work")
            await tool.execute(inp)

        assert len(dispatch_events) == 1
        assert dispatch_events[0].worker_name == "evt-worker"
        assert len(complete_events) == 1
        assert complete_events[0].worker_name == "evt-worker"
        assert complete_events[0].status in ("completed", "failed", "cancelled")


# ---------------------------------------------------------------------------
# Full flow: spawn_worker
# ---------------------------------------------------------------------------

VALID_WORKER_MD = """\
---
name: temp-analyst
model: claude-sonnet-4-6
tags: [analysis]
tools: []
---
You are a temporary analyst.
"""


class TestSpawnWorkerFlow:
    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_spawn_worker_full_flow(self, mock_prov: MagicMock) -> None:
        """spawn_worker: parse md → create worker → runs task → returns result."""
        mock_prov.return_value = _mock_provider()

        orch = Orchestrator(_make_config())
        tool = SpawnWorkerTool(orchestrator=orch, depth=0)

        with patch("neoagent.multi.tools.spawn_worker._create_worker_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(
                return_value=_make_completed_conversation("Temporary analysis complete.")
            )
            mock_create.return_value = mock_agent

            inp = SpawnWorkerInput(
                md_definition=VALID_WORKER_MD,
                instruction="Run a quick analysis",
            )
            result = await tool.execute(inp)

        assert result.is_error is False
        assert "completed" in result.output
        assert "Temporary analysis complete." in result.output

    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_spawn_worker_invalid_md_missing_name(self, mock_prov: MagicMock) -> None:
        """spawn_worker with md missing 'name' field returns error."""
        mock_prov.return_value = _mock_provider()
        orch = Orchestrator(_make_config())
        tool = SpawnWorkerTool(orchestrator=orch, depth=0)

        inp = SpawnWorkerInput(
            md_definition="---\nmodel: sonnet\n---\nNo name field.",
            instruction="Do something",
        )
        result = await tool.execute(inp)

        assert result.is_error is True
        assert "name" in result.output.lower()

    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_spawn_worker_emits_dispatch_and_complete_events(
        self, mock_prov: MagicMock
    ) -> None:
        """TaskDispatchEvent and TaskCompleteEvent are emitted when spawning."""
        mock_prov.return_value = _mock_provider()
        from neoagent.events import TaskCompleteEvent, TaskDispatchEvent

        orch = Orchestrator(_make_config())
        dispatch_events: list[TaskDispatchEvent] = []
        complete_events: list[TaskCompleteEvent] = []
        orch._event_bus.subscribe(TaskDispatchEvent, dispatch_events.append)
        orch._event_bus.subscribe(TaskCompleteEvent, complete_events.append)

        tool = SpawnWorkerTool(orchestrator=orch, depth=0)

        with patch("neoagent.multi.tools.spawn_worker._create_worker_agent") as mc:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(
                return_value=_make_completed_conversation("spawn done")
            )
            mc.return_value = mock_agent

            inp = SpawnWorkerInput(
                md_definition=VALID_WORKER_MD,
                instruction="Analyze something",
            )
            await tool.execute(inp)

        assert len(dispatch_events) == 1
        assert dispatch_events[0].worker_name == "temp-analyst"
        assert len(complete_events) == 1
        assert complete_events[0].worker_name == "temp-analyst"


# ---------------------------------------------------------------------------
# Parallel workers: one fails, one succeeds
# ---------------------------------------------------------------------------


class TestParallelWorkers:
    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_parallel_two_workers_one_fails_one_succeeds(
        self, mock_prov: MagicMock
    ) -> None:
        """Two parallel workers: failure is isolated, other continues."""
        mock_prov.return_value = _mock_provider()

        orch = Orchestrator(_make_config())
        orch.register_worker(_make_worker_card("worker-a"))
        orch.register_worker(_make_worker_card("worker-b"))

        async def run_task(worker_name: str, should_fail: bool):
            tool = DelegateTaskTool(orchestrator=orch, depth=0)
            with patch(
                "neoagent.multi.tools.delegate_task._create_worker_agent"
            ) as mc:
                if should_fail:
                    mc.return_value = MagicMock(
                        run=AsyncMock(side_effect=RuntimeError("worker-a crashed"))
                    )
                else:
                    mc.return_value = MagicMock(
                        run=AsyncMock(
                            return_value=_make_completed_conversation("worker-b done")
                        )
                    )
                inp = DelegateTaskInput(worker_name=worker_name, instruction="Do work")
                return await tool.execute(inp)

        results = await asyncio.gather(
            run_task("worker-a", should_fail=True),
            run_task("worker-b", should_fail=False),
        )

        # worker-a failed → is_error=True
        assert results[0].is_error is True
        assert "failed" in results[0].output

        # worker-b succeeded
        assert results[1].is_error is False
        assert "worker-b done" in results[1].output


# ---------------------------------------------------------------------------
# Depth control
# ---------------------------------------------------------------------------


class TestDepthControl:
    @patch("neoagent.agent._create_provider")
    def test_worker_at_max_depth_has_no_spawn_worker(self, mock_prov: MagicMock) -> None:
        """Worker created at depth == max_depth should not receive spawn_worker tool."""
        mock_prov.return_value = _mock_provider()
        from neoagent.events import EventBus

        card = _make_worker_card("deep-worker")

        orch = MagicMock()
        orch.max_depth = 2
        orch.config = NeoAgentConfig(api_key="sk-test")
        orch._tool_pool = {}
        orch._event_bus = EventBus()

        agent = _create_worker_agent(card, orch, depth=2)  # at max_depth

        tool_names = {s["name"] for s in agent._registry.get_schemas()}
        assert "spawn_worker" not in tool_names

    @patch("neoagent.agent._create_provider")
    def test_worker_below_max_depth_has_spawn_worker(self, mock_prov: MagicMock) -> None:
        """Worker created below max_depth should have spawn_worker registered."""
        mock_prov.return_value = _mock_provider()
        from neoagent.events import EventBus

        card = _make_worker_card("mid-worker")

        orch = MagicMock()
        orch.max_depth = 2
        orch.config = NeoAgentConfig(api_key="sk-test")
        orch._tool_pool = {}
        orch._event_bus = EventBus()

        agent = _create_worker_agent(card, orch, depth=1)  # below max_depth

        tool_names = {s["name"] for s in agent._registry.get_schemas()}
        assert "spawn_worker" in tool_names


# ---------------------------------------------------------------------------
# Cancel task
# ---------------------------------------------------------------------------


class TestCancelTask:
    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_cancel_running_task(self, mock_prov: MagicMock) -> None:
        """Cancel a running task via CancelTaskTool."""
        mock_prov.return_value = _mock_provider()
        orch = Orchestrator(_make_config())

        # Create a long-running asyncio task
        async def slow_task():
            await asyncio.sleep(999)

        asyncio_task = asyncio.create_task(slow_task())

        # Register the Task and associate the asyncio.Task
        task = Task(task_id="slow-1", instruction="slow work")
        orch._task_tracker.track(task)
        orch._task_tracker.set_asyncio_task("slow-1", asyncio_task)

        cancel_tool = CancelTaskTool(orchestrator=orch)
        inp = CancelTaskInput(task_id="slow-1")
        result = await cancel_tool.execute(inp)

        assert result.is_error is False
        assert "slow-1" in result.output
        # Give event loop a tick to process the cancellation
        await asyncio.sleep(0)
        assert asyncio_task.cancelled() or asyncio_task.cancelling() > 0 or asyncio_task.done()

    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_cancel_nonexistent_task(self, mock_prov: MagicMock) -> None:
        """Cancel a non-existent task returns error."""
        mock_prov.return_value = _mock_provider()
        orch = Orchestrator(_make_config())

        cancel_tool = CancelTaskTool(orchestrator=orch)
        inp = CancelTaskInput(task_id="not-real")
        result = await cancel_tool.execute(inp)

        assert result.is_error is True
        assert "not-real" in result.output

    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_cancel_already_completed_task(self, mock_prov: MagicMock) -> None:
        """Cancelling an already-completed task returns error."""
        mock_prov.return_value = _mock_provider()
        orch = Orchestrator(_make_config())

        task = Task(task_id="done-1", instruction="finished work")
        orch._task_tracker.track(task)
        fake_result = TaskResult(
            task_id="done-1",
            status="completed",
            output="done",
            error=None,
            usage=TokenUsage(0, 0),
            work_summary=None,
            turns_completed=1,
        )
        orch._task_tracker.complete("done-1", fake_result)

        cancel_tool = CancelTaskTool(orchestrator=orch)
        inp = CancelTaskInput(task_id="done-1")
        result = await cancel_tool.execute(inp)

        assert result.is_error is True


# ---------------------------------------------------------------------------
# Event bubbling
# ---------------------------------------------------------------------------


class TestEventBubbling:
    @patch("neoagent.agent._create_provider")
    def test_worker_events_bubble_to_orchestrator(self, mock_prov: MagicMock) -> None:
        """Events emitted by a worker's EventBus are wrapped as WorkerEvent
        and re-emitted on the orchestrator's EventBus."""
        from neoagent.events import EventBus, ToolCallEvent, WorkerEvent
        from neoagent.multi.events import _setup_event_bubble

        mock_prov.return_value = _mock_provider()

        # A real NeoAgent as the worker
        from neoagent.agent import NeoAgent

        worker_agent = NeoAgent(_make_config())

        # Orchestrator mock with a real EventBus
        orch = MagicMock()
        orch._event_bus = EventBus()

        received: list[WorkerEvent] = []
        orch._event_bus.subscribe(WorkerEvent, received.append)

        _setup_event_bubble(
            worker_agent, orch, "test-worker", task_id="t99", depth=1
        )

        # Emit ToolCallEvent from the worker's bus
        evt = ToolCallEvent(name="read", input_data={}, call_id="c1")
        worker_agent._event_bus.emit(evt)

        assert len(received) == 1
        we = received[0]
        assert we.worker_name == "test-worker"
        assert we.task_id == "t99"
        assert we.depth == 1
        assert isinstance(we.inner, ToolCallEvent)
        assert we.inner.name == "read"

    @patch("neoagent.agent._create_provider")
    def test_event_bubble_uses_task_id_from_create_worker_agent(
        self, mock_prov: MagicMock
    ) -> None:
        """_create_worker_agent passes task_id to _setup_event_bubble."""
        from neoagent.events import EventBus

        mock_prov.return_value = _mock_provider()

        card = _make_worker_card("labelled-worker")
        orch = MagicMock()
        orch.max_depth = 0
        orch.config = NeoAgentConfig(api_key="sk-test")
        orch._tool_pool = {}
        orch._event_bus = EventBus()

        with patch("neoagent.multi.events._setup_event_bubble") as mock_bubble:
            _create_worker_agent(card, orch, depth=0, task_id="explicit-tid")

        mock_bubble.assert_called_once()
        call_args = mock_bubble.call_args[0]
        # (agent, orchestrator, worker_name, task_id, depth)
        assert call_args[3] == "explicit-tid"


# ---------------------------------------------------------------------------
# Emit helpers: _emit_dispatch_event / _emit_complete_event
# ---------------------------------------------------------------------------


class TestEmitHelpers:
    def test_emit_dispatch_event_emits_correct_event(self) -> None:
        from neoagent.events import EventBus, TaskDispatchEvent

        bus = EventBus()
        received: list[TaskDispatchEvent] = []
        bus.subscribe(TaskDispatchEvent, received.append)

        orch = MagicMock()
        orch._event_bus = bus

        _emit_dispatch_event(orch, "tid-1", "worker-x", "do something", depth=1)

        assert len(received) == 1
        e = received[0]
        assert e.task_id == "tid-1"
        assert e.worker_name == "worker-x"
        assert e.instruction == "do something"
        assert e.depth == 1

    def test_emit_complete_event_emits_correct_event(self) -> None:
        from neoagent.events import EventBus, TaskCompleteEvent

        bus = EventBus()
        received: list[TaskCompleteEvent] = []
        bus.subscribe(TaskCompleteEvent, received.append)

        orch = MagicMock()
        orch._event_bus = bus

        result = TaskResult(
            task_id="tid-2",
            status="completed",
            output="done",
            error=None,
            usage=TokenUsage(10, 20),
            work_summary=None,
            turns_completed=3,
        )
        _emit_complete_event(orch, result, "worker-y", depth=0)

        assert len(received) == 1
        e = received[0]
        assert e.task_id == "tid-2"
        assert e.worker_name == "worker-y"
        assert e.status == "completed"
        assert e.turns_completed == 3

    def test_emit_dispatch_event_never_raises_on_bus_error(self) -> None:
        """_emit_dispatch_event silently swallows exceptions from the bus."""
        orch = MagicMock()
        orch._event_bus.emit.side_effect = RuntimeError("bus exploded")

        # Should not raise
        _emit_dispatch_event(orch, "tid", "w", "instr", depth=0)

    def test_emit_complete_event_never_raises_on_bus_error(self) -> None:
        """_emit_complete_event silently swallows exceptions from the bus."""
        orch = MagicMock()
        orch._event_bus.emit.side_effect = RuntimeError("bus exploded")

        result = TaskResult(
            task_id="t",
            status="failed",
            output="",
            error="oops",
            usage=None,
            work_summary=None,
            turns_completed=0,
        )
        # Should not raise
        _emit_complete_event(orch, result, "w", depth=0)


# ---------------------------------------------------------------------------
# Semaphore limits concurrency
# ---------------------------------------------------------------------------


class TestSemaphoreLimit:
    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_semaphore_limits_concurrent_workers(self, mock_prov: MagicMock) -> None:
        """max_concurrent_workers=2 allows only 2 concurrent workers at a time."""
        mock_prov.return_value = _mock_provider()
        orch = Orchestrator(_make_config(), max_concurrent_workers=2)

        # Semaphore starts at capacity=2
        assert orch._semaphore._value == 2

        # Acquire twice — exhausted
        await orch._semaphore.acquire()
        await orch._semaphore.acquire()
        assert orch._semaphore._value == 0

        # Third acquire blocks
        acquired = False

        async def try_acquire():
            nonlocal acquired
            await orch._semaphore.acquire()
            acquired = True

        task = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.01)
        assert acquired is False  # still blocked

        # Release one — third can proceed
        orch._semaphore.release()
        await asyncio.sleep(0.01)
        assert acquired is True

        # Cleanup
        orch._semaphore.release()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_orchestrator_default_semaphore_value(self, mock_prov: MagicMock) -> None:
        """Default max_concurrent_workers=5 means semaphore starts at 5."""
        mock_prov.return_value = _mock_provider()
        orch = Orchestrator(_make_config())

        assert orch._semaphore._value == 5


# ---------------------------------------------------------------------------
# Real worker factory (no _create_worker_agent mock)
# ---------------------------------------------------------------------------


def _mock_full_provider() -> MagicMock:
    """Provider mock that supports real agent.run() — create() returns a proper Response."""
    from neoagent.core.types import TextBlock as _TextBlock
    from neoagent.providers.base import Response as _Response

    p = MagicMock()
    p.get_context_window.return_value = 200_000
    p.create = AsyncMock(
        return_value=_Response(
            content=[_TextBlock(text="done")],
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=5,
        )
    )
    return p


class TestRealWorkerFactory:
    """Tests that go through real _create_worker_agent without mocking it.

    Only the LLM provider is mocked. This catches config/attribute bugs
    that tests using mocked _create_worker_agent cannot detect.
    """

    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_delegate_task_real_worker_factory(self, mock_prov: MagicMock) -> None:
        """delegate_task with real _create_worker_agent must not raise AttributeError."""
        mock_prov.return_value = _mock_full_provider()

        config = NeoAgentConfig(api_key="test", model="claude-haiku-4-5")
        orch = Orchestrator(config)

        card = WorkerCard(
            name="helper",
            description="A helper",
            instruction="You are a helper.",
            tags=(),
            model=None,
            tools=(),
        )
        orch.register_worker(card)

        tool = orch._brain._registry.get_tool("delegate_task")
        assert tool is not None

        result = await tool.execute(DelegateTaskInput(worker_name="helper", instruction="say hi"))

        assert not result.is_error, f"Expected success, got error: {result.output}"

    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_cancel_does_not_leak_cancelled_error(self, mock_prov: MagicMock) -> None:
        """Cancelling a task must not raise CancelledError from tool.execute()."""
        mock_prov.return_value = _mock_provider()

        config = NeoAgentConfig(api_key="test", model="claude-haiku-4-5")
        orch = Orchestrator(config)

        card = WorkerCard(
            name="helper",
            description="A helper",
            instruction="You are a helper.",
            tags=(),
            model=None,
            tools=(),
        )
        orch.register_worker(card)

        from neoagent.multi.tools.delegate_task import DelegateTaskInput

        # Start the task and immediately cancel via tracker
        tool = orch._brain._registry.get_tool("delegate_task")

        exec_task = asyncio.create_task(
            tool.execute(DelegateTaskInput(worker_name="helper", instruction="work"))
        )
        await asyncio.sleep(0)  # let it start
        # Cancel all active tasks
        for tid in orch._task_tracker.list_active():
            orch._task_tracker.cancel(tid)

        # execute() must NOT raise CancelledError — it should return a ToolResult
        try:
            result = await asyncio.wait_for(exec_task, timeout=2.0)
            # If we get here, it returned normally — check it's not an uncaught error
            # (it might be a cancelled result or a completed result)
        except asyncio.CancelledError:
            pytest.fail("tool.execute() must not raise CancelledError — it must return a ToolResult")
        except asyncio.TimeoutError:
            exec_task.cancel()
            # Timeout is ok for this test — the key is no CancelledError

    @pytest.mark.asyncio
    @patch("neoagent.agent._create_provider")
    async def test_spawn_worker_real_worker_factory(self, mock_prov: MagicMock) -> None:
        """spawn_worker with real _create_worker_agent must not raise AttributeError."""
        mock_prov.return_value = _mock_full_provider()

        config = NeoAgentConfig(api_key="test", model="claude-haiku-4-5")
        orch = Orchestrator(config)

        tool = orch._brain._registry.get_tool("spawn_worker")
        assert tool is not None

        md = "---\nname: temp-worker\nmodel: claude-haiku-4-5\ntags: []\ntools: []\n---\nYou are a temp helper."
        result = await tool.execute(SpawnWorkerInput(md_definition=md, instruction="say hello"))

        assert not result.is_error, f"Expected success, got error: {result.output}"


def test_delegate_task_is_concurrent_safe() -> None:
    """DelegateTaskTool must be marked concurrent safe for parallel execution."""
    from unittest.mock import MagicMock, patch
    from neoagent.config import NeoAgentConfig
    from neoagent.multi.orchestrator import Orchestrator
    from neoagent.multi.tools.delegate_task import DelegateTaskTool

    config = NeoAgentConfig(api_key="test", model="claude-haiku-4-5")
    with patch("neoagent.agent._create_provider") as m:
        m.return_value = MagicMock()
        orch = Orchestrator(config)

    tool = DelegateTaskTool(orchestrator=orch)
    assert tool.is_concurrent_safe is True
