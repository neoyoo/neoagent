from __future__ import annotations

import dataclasses

import pytest

from neoagent.multi.worker import WorkerCard


class TestWorkerCard:
    def test_required_fields(self) -> None:
        card = WorkerCard(
            name="coder",
            description="Writes Python code",
            instruction="You are a skilled Python developer.",
            tags=("python", "code"),
            model="claude-sonnet-4-6",
            tools=("bash", "read"),
        )
        assert card.name == "coder"
        assert card.description == "Writes Python code"
        assert card.instruction == "You are a skilled Python developer."
        assert card.model == "claude-sonnet-4-6"

    def test_optional_model_none(self) -> None:
        card = WorkerCard(
            name="coder",
            description="",
            instruction="You are a coder.",
            tags=(),
            model=None,
            tools=(),
        )
        assert card.model is None

    def test_empty_tools(self) -> None:
        card = WorkerCard(
            name="analyst",
            description="Analyzes data",
            instruction="You analyze data.",
            tags=("data",),
            model=None,
            tools=(),
        )
        assert card.tools == ()

    def test_frozen(self) -> None:
        card = WorkerCard(
            name="coder",
            description="",
            instruction="You are a coder.",
            tags=(),
            model=None,
            tools=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            card.name = "changed"  # type: ignore[misc]

    def test_tags_are_tuple(self) -> None:
        card = WorkerCard(
            name="coder",
            description="",
            instruction="",
            tags=("python", "testing"),
            model=None,
            tools=(),
        )
        assert isinstance(card.tags, tuple)
        assert card.tags == ("python", "testing")

    def test_tools_are_tuple(self) -> None:
        card = WorkerCard(
            name="coder",
            description="",
            instruction="",
            tags=(),
            model=None,
            tools=("bash", "read", "write"),
        )
        assert isinstance(card.tools, tuple)
        assert card.tools == ("bash", "read", "write")
