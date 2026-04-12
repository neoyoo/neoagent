from __future__ import annotations

import asyncio
import dataclasses
import uuid

import pytest

from neoagent.core.types import Message, TextBlock, ToolUseBlock
from neoagent.multi.task import Task, TaskResult, TaskTracker, TokenUsage, _extract_work_summary


class TestTokenUsage:
    def test_fields(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_total_property(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total == 150

    def test_zero_total(self) -> None:
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        assert usage.total == 0

    def test_frozen(self) -> None:
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            usage.input_tokens = 999  # type: ignore[misc]


class TestTask:
    def test_basic_fields(self) -> None:
        task = Task(task_id="t1", instruction="do something")
        assert task.task_id == "t1"
        assert task.instruction == "do something"

    def test_default_constraints(self) -> None:
        task = Task(task_id="t1", instruction="work")
        assert task.max_turns == 20
        assert task.timeout == 1800
        assert task.max_output_tokens == 2000

    def test_custom_constraints(self) -> None:
        task = Task(task_id="t1", instruction="work", max_turns=5, timeout=300, max_output_tokens=500)
        assert task.max_turns == 5
        assert task.timeout == 300
        assert task.max_output_tokens == 500

    def test_frozen(self) -> None:
        task = Task(task_id="t1", instruction="work")
        with pytest.raises(dataclasses.FrozenInstanceError):
            task.instruction = "changed"  # type: ignore[misc]

    def test_context_is_tuple_default(self) -> None:
        task = Task(task_id="t1", instruction="work")
        assert task.context == ()
        assert isinstance(task.context, tuple)

    def test_context_with_values(self) -> None:
        task = Task(task_id="t1", instruction="work", context=("ctx1", "ctx2"))
        assert task.context == ("ctx1", "ctx2")
        assert isinstance(task.context, tuple)

    def test_metadata_default_empty(self) -> None:
        task = Task(task_id="t1", instruction="work")
        assert task.metadata == {}

    def test_metadata_custom(self) -> None:
        task = Task(task_id="t1", instruction="work", metadata={"key": "value"})
        assert task.metadata["key"] == "value"

    def test_create_factory_generates_uuid(self) -> None:
        task = Task.create(instruction="do work")
        assert task.instruction == "do work"
        # task_id should be a valid UUID
        parsed = uuid.UUID(task.task_id)
        assert str(parsed) == task.task_id

    def test_create_factory_unique_ids(self) -> None:
        t1 = Task.create(instruction="task 1")
        t2 = Task.create(instruction="task 2")
        assert t1.task_id != t2.task_id

    def test_create_factory_passes_through_fields(self) -> None:
        task = Task.create(
            instruction="work",
            context=("a", "b"),
            max_turns=10,
            timeout=600,
            max_output_tokens=1000,
            metadata={"priority": "high"},
        )
        assert task.context == ("a", "b")
        assert task.max_turns == 10
        assert task.timeout == 600
        assert task.max_output_tokens == 1000
        assert task.metadata["priority"] == "high"

    def test_create_factory_returns_task(self) -> None:
        task = Task.create(instruction="x")
        assert isinstance(task, Task)


class TestTaskResult:
    def _usage(self) -> TokenUsage:
        return TokenUsage(input_tokens=100, output_tokens=50)

    def test_completed_status(self) -> None:
        result = TaskResult(
            task_id="t1",
            status="completed",
            output="done",
            error=None,
            usage=self._usage(),
            work_summary="finished",
            turns_completed=3,
        )
        assert result.status == "completed"
        assert result.output == "done"
        assert result.error is None

    def test_failed_status(self) -> None:
        result = TaskResult(
            task_id="t1",
            status="failed",
            output=None,
            error="something went wrong",
            usage=self._usage(),
            work_summary="failed at step 2",
            turns_completed=2,
        )
        assert result.status == "failed"
        assert result.error == "something went wrong"
        assert result.output is None

    def test_cancelled_status(self) -> None:
        result = TaskResult(
            task_id="t1",
            status="cancelled",
            output=None,
            error="timed out",
            usage=self._usage(),
            work_summary="cancelled",
            turns_completed=0,
        )
        assert result.status == "cancelled"

    def test_with_usage(self) -> None:
        usage = TokenUsage(input_tokens=200, output_tokens=100)
        result = TaskResult(
            task_id="t1",
            status="completed",
            output="ok",
            error=None,
            usage=usage,
            work_summary="done",
            turns_completed=5,
        )
        assert result.usage.input_tokens == 200
        assert result.usage.output_tokens == 100
        assert result.usage.total == 300

    def test_frozen(self) -> None:
        result = TaskResult(
            task_id="t1",
            status="completed",
            output="ok",
            error=None,
            usage=self._usage(),
            work_summary="done",
            turns_completed=1,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.status = "failed"  # type: ignore[misc]

    def test_is_success_completed(self) -> None:
        result = TaskResult(
            task_id="t1",
            status="completed",
            output="ok",
            error=None,
            usage=self._usage(),
            work_summary="done",
            turns_completed=1,
        )
        assert result.is_success is True

    def test_is_success_failed(self) -> None:
        result = TaskResult(
            task_id="t1",
            status="failed",
            output=None,
            error="err",
            usage=self._usage(),
            work_summary="failed",
            turns_completed=1,
        )
        assert result.is_success is False

    def test_is_success_cancelled(self) -> None:
        result = TaskResult(
            task_id="t1",
            status="cancelled",
            output=None,
            error=None,
            usage=self._usage(),
            work_summary="cancelled",
            turns_completed=0,
        )
        assert result.is_success is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(task_id: str, status: str = "completed") -> TaskResult:
    return TaskResult(
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        output="done" if status == "completed" else None,
        error=None,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        work_summary="summary",
        turns_completed=1,
    )


def _make_task(instruction: str = "do work") -> Task:
    return Task.create(instruction=instruction)


def _make_assistant_message(text: str | None = None, tools: list[dict] | None = None) -> Message:
    content: list = []
    if tools:
        for t in tools:
            content.append(ToolUseBlock(id=t["id"], name=t["name"], input=t.get("input", {})))
    if text:
        content.append(TextBlock(text=text))
    return Message(role="assistant", content=content)


# ---------------------------------------------------------------------------
# TaskTracker tests
# ---------------------------------------------------------------------------

class TestTaskTracker:
    def test_track_and_get(self) -> None:
        tracker = TaskTracker()
        task = _make_task()
        tracker.track(task)
        retrieved = tracker.get_task(task.task_id)
        assert retrieved is task

    def test_get_unknown_task_returns_none(self) -> None:
        tracker = TaskTracker()
        assert tracker.get_task("nonexistent") is None

    def test_complete_stores_result(self) -> None:
        tracker = TaskTracker()
        task = _make_task()
        tracker.track(task)
        result = _make_result(task.task_id)
        tracker.complete(task.task_id, result)
        assert tracker.get_result(task.task_id) is result

    def test_get_result_before_complete_returns_none(self) -> None:
        tracker = TaskTracker()
        task = _make_task()
        tracker.track(task)
        assert tracker.get_result(task.task_id) is None

    def test_get_result_unknown_returns_none(self) -> None:
        tracker = TaskTracker()
        assert tracker.get_result("unknown-id") is None

    def test_list_active_before_complete(self) -> None:
        tracker = TaskTracker()
        task = _make_task()
        tracker.track(task)
        active = tracker.list_active()
        assert task.task_id in active

    def test_list_active_after_complete(self) -> None:
        tracker = TaskTracker()
        task = _make_task()
        tracker.track(task)
        tracker.complete(task.task_id, _make_result(task.task_id))
        active = tracker.list_active()
        assert task.task_id not in active

    def test_list_all_includes_status(self) -> None:
        tracker = TaskTracker()
        task = _make_task("long instruction text here")
        tracker.track(task)
        summaries = tracker.list_all()
        assert len(summaries) == 1
        entry = summaries[0]
        assert entry["task_id"] == task.task_id
        assert entry["status"] == "active"
        assert "instruction" in entry

    def test_list_all_status_changes_after_complete(self) -> None:
        tracker = TaskTracker()
        task = _make_task()
        tracker.track(task)
        tracker.complete(task.task_id, _make_result(task.task_id))
        summaries = tracker.list_all()
        entry = next(e for e in summaries if e["task_id"] == task.task_id)
        assert entry["status"] == "completed"

    def test_list_all_instruction_truncated(self) -> None:
        long_instr = "x" * 200
        tracker = TaskTracker()
        task = _make_task(long_instr)
        tracker.track(task)
        entry = tracker.list_all()[0]
        assert len(entry["instruction"]) < 200

    async def test_cancel_active_task(self) -> None:
        tracker = TaskTracker()
        task = _make_task()
        tracker.track(task)

        # Create a real asyncio task that we can cancel
        async def _noop() -> None:
            await asyncio.sleep(100)

        asyncio_task = asyncio.create_task(_noop())
        tracker.set_asyncio_task(task.task_id, asyncio_task)

        result = tracker.cancel(task.task_id)
        assert result is True
        assert asyncio_task.cancelled() or asyncio_task.cancelling() > 0

    def test_cancel_unknown_task_returns_false(self) -> None:
        tracker = TaskTracker()
        assert tracker.cancel("no-such-id") is False

    def test_cancel_completed_task_returns_false(self) -> None:
        tracker = TaskTracker()
        task = _make_task()
        tracker.track(task)
        tracker.complete(task.task_id, _make_result(task.task_id))
        assert tracker.cancel(task.task_id) is False

    def test_multiple_tasks_tracked(self) -> None:
        tracker = TaskTracker()
        t1 = _make_task("task one")
        t2 = _make_task("task two")
        tracker.track(t1)
        tracker.track(t2)
        assert len(tracker.list_active()) == 2
        tracker.complete(t1.task_id, _make_result(t1.task_id))
        active = tracker.list_active()
        assert t1.task_id not in active
        assert t2.task_id in active


# ---------------------------------------------------------------------------
# _extract_work_summary tests
# ---------------------------------------------------------------------------

class TestExtractWorkSummary:
    def test_empty_messages_returns_fallback(self) -> None:
        result = _extract_work_summary([])
        assert result == "（无有效工作摘要）"

    def test_no_assistant_messages_returns_fallback(self) -> None:
        messages = [Message(role="user", content="hello")]
        result = _extract_work_summary(messages)
        assert result == "（无有效工作摘要）"

    def test_with_tool_calls_includes_tool_name(self) -> None:
        msg = _make_assistant_message(tools=[{"id": "t1", "name": "bash", "input": {}}])
        result = _extract_work_summary([msg])
        assert "bash" in result

    def test_last_assistant_text_included(self) -> None:
        msg = _make_assistant_message(text="Final answer here")
        result = _extract_work_summary([msg])
        assert "Final answer here" in result

    def test_truncates_to_last_5_tool_calls(self) -> None:
        tools = [{"id": f"t{i}", "name": f"tool_{i}", "input": {}} for i in range(8)]
        msg = _make_assistant_message(tools=tools)
        result = _extract_work_summary([msg])
        # Only last 5 should appear
        for i in range(3, 8):
            assert f"tool_{i}" in result
        # First 3 should not
        for i in range(3):
            assert f"tool_{i}" not in result

    def test_string_content_message_ignored(self) -> None:
        # Messages with string content (not list) should not crash
        msg = Message(role="assistant", content="plain string output")
        result = _extract_work_summary([msg])
        # Should include the text content
        assert result != "（无有效工作摘要）"

    def test_mixed_messages(self) -> None:
        user_msg = Message(role="user", content="do something")
        tool_msg = _make_assistant_message(tools=[{"id": "t1", "name": "read_file", "input": {}}])
        text_msg = _make_assistant_message(text="Task completed successfully")
        result = _extract_work_summary([user_msg, tool_msg, text_msg])
        assert "read_file" in result
        assert "Task completed successfully" in result


# ---------------------------------------------------------------------------
# _run_worker session integration tests
# ---------------------------------------------------------------------------


class TestRunWorkerSession:
    """Tests that _run_worker uses explicit Session so work_summary is real."""

    @pytest.mark.asyncio
    async def test_cancel_work_summary_contains_real_messages(self) -> None:
        """On CancelledError, work_summary must reflect actual agent messages."""
        import asyncio
        from unittest.mock import MagicMock, patch
        from neoagent.config import NeoAgentConfig
        from neoagent.agent import NeoAgent
        from neoagent.multi.task import Task, TaskTracker, _run_worker
        from neoagent.providers.base import Response

        call_count = 0
        tool_use_block = ToolUseBlock(id="tu1", name="read_file", input={"path": "/foo"})
        first_response = Response(
            content=[tool_use_block],
            stop_reason="tool_use",
            input_tokens=10,
            output_tokens=5,
        )

        async def fake_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            raise asyncio.CancelledError()

        config = NeoAgentConfig(api_key="test", model="claude-haiku-4-5")
        with patch("neoagent.agent._create_provider") as mock_prov:
            provider_mock = MagicMock()
            provider_mock.create = fake_complete
            provider_mock.get_context_window.return_value = 100_000
            mock_prov.return_value = provider_mock
            agent = NeoAgent(config)

        task = Task.create(instruction="read /foo and summarize")
        tracker = TaskTracker()
        tracker.track(task)

        result = await _run_worker(agent, task, tracker)

        assert result.status == "cancelled"
        assert result.work_summary is not None
        assert "read_file" in result.work_summary, (
            f"Expected 'read_file' in work_summary, got: {result.work_summary!r}"
        )

    @pytest.mark.asyncio
    async def test_success_work_summary_populated(self) -> None:
        """On success, work_summary should contain last assistant output."""
        from unittest.mock import MagicMock, patch
        from neoagent.config import NeoAgentConfig
        from neoagent.agent import NeoAgent
        from neoagent.multi.task import Task, TaskTracker, _run_worker
        from neoagent.core.types import TextBlock
        from neoagent.providers.base import Response

        async def fake_complete(*args, **kwargs):
            return Response(
                content=[TextBlock(text="Analysis complete.")],
                stop_reason="end_turn",
                input_tokens=10,
                output_tokens=8,
            )

        config = NeoAgentConfig(api_key="test", model="claude-haiku-4-5")
        with patch("neoagent.agent._create_provider") as mock_prov:
            provider_mock = MagicMock()
            provider_mock.create = fake_complete
            provider_mock.get_context_window.return_value = 100_000
            mock_prov.return_value = provider_mock
            agent = NeoAgent(config)

        task = Task.create(instruction="analyze this")
        tracker = TaskTracker()
        tracker.track(task)

        result = await _run_worker(agent, task, tracker)

        assert result.status == "completed"
        assert result.work_summary is not None
        assert "Analysis complete." in result.work_summary

    @pytest.mark.asyncio
    async def test_success_token_usage_from_session_state(self) -> None:
        """Successful task must report real token usage from session state."""
        from unittest.mock import MagicMock, patch
        from neoagent.config import NeoAgentConfig
        from neoagent.agent import NeoAgent
        from neoagent.multi.task import Task, TaskTracker, _run_worker
        from neoagent.providers.base import Response

        async def fake_complete(*args, **kwargs):
            return Response(
                content=[TextBlock(text="Done.")],
                stop_reason="end_turn",
                input_tokens=123,
                output_tokens=45,
            )

        config = NeoAgentConfig(api_key="test", model="claude-haiku-4-5")
        with patch("neoagent.agent._create_provider") as mock_prov:
            provider_mock = MagicMock()
            provider_mock.create = fake_complete
            provider_mock.get_context_window.return_value = 100_000
            mock_prov.return_value = provider_mock
            agent = NeoAgent(config)

        task = Task.create(instruction="do something")
        tracker = TaskTracker()
        tracker.track(task)

        result = await _run_worker(agent, task, tracker)

        assert result.status == "completed"
        assert result.usage is not None
        assert result.usage.input_tokens > 0, f"Expected >0 input tokens, got {result.usage.input_tokens}"
        assert result.usage.output_tokens > 0, f"Expected >0 output tokens, got {result.usage.output_tokens}"

    @pytest.mark.asyncio
    async def test_cancel_turns_completed_reflects_real_turns(self) -> None:
        """Cancelled task must report actual turns completed, not hardcoded 0."""
        import asyncio
        from unittest.mock import MagicMock, patch
        from neoagent.config import NeoAgentConfig
        from neoagent.agent import NeoAgent
        from neoagent.multi.task import Task, TaskTracker, _run_worker
        from neoagent.providers.base import Response

        call_count = 0
        tool_block = ToolUseBlock(id="t1", name="read_file", input={"path": "/x"})

        async def fake_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Response(
                    content=[tool_block],
                    stop_reason="tool_use",
                    input_tokens=10,
                    output_tokens=5,
                )
            raise asyncio.CancelledError()

        config = NeoAgentConfig(api_key="test", model="claude-haiku-4-5")
        with patch("neoagent.agent._create_provider") as mock_prov:
            provider_mock = MagicMock()
            provider_mock.create = fake_complete
            provider_mock.get_context_window.return_value = 100_000
            mock_prov.return_value = provider_mock
            agent = NeoAgent(config)

        task = Task.create(instruction="work then cancel")
        tracker = TaskTracker()
        tracker.track(task)

        result = await _run_worker(agent, task, tracker)

        assert result.status == "cancelled"
        assert result.turns_completed >= 1, (
            f"Expected ≥1 turns_completed (worker ran before cancel), got {result.turns_completed}"
        )

    @pytest.mark.asyncio
    async def test_max_turns_limits_worker_execution(self) -> None:
        """Task.max_turns must actually cap the worker's turn count."""
        from unittest.mock import MagicMock, patch
        from neoagent.config import NeoAgentConfig
        from neoagent.agent import NeoAgent
        from neoagent.multi.task import Task, TaskTracker, _run_worker
        from neoagent.core.types import ToolUseBlock
        from neoagent.providers.base import Response

        call_count = 0
        tool_block = ToolUseBlock(id="t1", name="bash", input={"cmd": "ls"})

        async def fake_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Always return tool_use to force the loop to keep going
            return Response(
                content=[tool_block],
                stop_reason="tool_use",
                input_tokens=5,
                output_tokens=3,
            )

        # config.max_turns = 50, but task.max_turns = 2
        config = NeoAgentConfig(api_key="test", model="claude-haiku-4-5", max_turns=50)
        with patch("neoagent.agent._create_provider") as mock_prov:
            provider_mock = MagicMock()
            provider_mock.create = fake_complete
            provider_mock.get_context_window.return_value = 100_000
            mock_prov.return_value = provider_mock
            agent = NeoAgent(config)

        # Task with max_turns=2 — should stop after 2 LLM calls, NOT 50
        task = Task.create(instruction="keep working", max_turns=2)
        tracker = TaskTracker()
        tracker.track(task)

        result = await _run_worker(agent, task, tracker)

        # Loop should have stopped after task.max_turns calls
        assert call_count <= 2, f"Expected ≤ 2 LLM calls, got {call_count}"
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# S5: Role validation in _run_worker context parsing
# ---------------------------------------------------------------------------


class TestRunWorkerRoleValidation:
    """S5: context items with invalid roles must be downgraded to 'user'."""

    def _parse_context_role(self, role_value: str) -> str:
        """Simulate the role parsing logic from _run_worker."""
        import json
        ctx_item = json.dumps({"role": role_value, "content": "hello"})
        parsed = json.loads(ctx_item)
        role = parsed.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        return role

    def test_system_role_downgraded_to_user(self) -> None:
        """A context item with role='system' must be treated as 'user'."""
        assert self._parse_context_role("system") == "user"

    def test_tool_role_downgraded_to_user(self) -> None:
        """A context item with role='tool' must be treated as 'user'."""
        assert self._parse_context_role("tool") == "user"

    def test_arbitrary_role_downgraded_to_user(self) -> None:
        """A context item with an arbitrary invalid role must be treated as 'user'."""
        assert self._parse_context_role("attacker_role") == "user"

    def test_user_role_preserved(self) -> None:
        """A context item with role='user' is kept as-is."""
        assert self._parse_context_role("user") == "user"

    def test_assistant_role_preserved(self) -> None:
        """A context item with role='assistant' is kept as-is."""
        assert self._parse_context_role("assistant") == "assistant"

    @pytest.mark.asyncio
    async def test_run_worker_system_role_context_item_uses_user(self) -> None:
        """End-to-end: _run_worker must not pass role='system' to Message."""
        import json
        from unittest.mock import MagicMock, patch
        from neoagent.config import NeoAgentConfig
        from neoagent.agent import NeoAgent
        from neoagent.multi.task import Task, TaskTracker, _run_worker
        from neoagent.providers.base import Response

        captured_messages: list = []

        async def fake_complete(*args, **kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return Response(
                content=[TextBlock(text="Done.")],
                stop_reason="end_turn",
                input_tokens=5,
                output_tokens=3,
            )

        config = NeoAgentConfig(api_key="test", model="claude-haiku-4-5")
        with patch("neoagent.agent._create_provider") as mock_prov:
            provider_mock = MagicMock()
            provider_mock.create = fake_complete
            provider_mock.get_context_window.return_value = 100_000
            mock_prov.return_value = provider_mock
            agent = NeoAgent(config)

        # Context item with role="system" — should be downgraded to "user"
        ctx_item = json.dumps({"role": "system", "content": "you are a bot"})
        task = Task.create(instruction="do work", context=(ctx_item,))
        tracker = TaskTracker()
        tracker.track(task)

        result = await _run_worker(agent, task, tracker)

        assert result.status == "completed"
        # Verify no message has role='system' in the messages sent to provider
        roles = [m.role for m in captured_messages]
        assert "system" not in roles, (
            f"Found 'system' role in messages sent to provider: {roles}"
        )
