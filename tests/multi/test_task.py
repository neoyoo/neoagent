from __future__ import annotations

import dataclasses
import uuid

import pytest

from neoagent.multi.task import Task, TaskResult, TokenUsage


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
