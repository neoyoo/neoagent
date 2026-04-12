from __future__ import annotations

import dataclasses

import pytest

from neoagent.multi.worker import WorkerCard, WorkerPool


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_card(name: str, description: str = "desc") -> WorkerCard:
    return WorkerCard(
        name=name,
        description=description,
        instruction=f"You are {name}.",
        tags=(),
        model=None,
        tools=(),
    )


# ---------------------------------------------------------------------------
# WorkerPool tests
# ---------------------------------------------------------------------------


class TestWorkerPool:
    def test_worker_pool_register_and_find(self) -> None:
        pool = WorkerPool()
        card = _make_card("alpha")
        pool.register(card)
        assert pool.find("alpha") is card

    def test_worker_pool_find_missing_returns_none(self) -> None:
        pool = WorkerPool()
        assert pool.find("nonexistent") is None

    def test_worker_pool_list_all_empty(self) -> None:
        pool = WorkerPool()
        assert pool.list_all() == []

    def test_worker_pool_list_all_returns_all(self) -> None:
        pool = WorkerPool()
        cards = [_make_card("a"), _make_card("b"), _make_card("c")]
        for card in cards:
            pool.register(card)
        result = pool.list_all()
        assert len(result) == 3
        assert set(c.name for c in result) == {"a", "b", "c"}

    def test_worker_pool_register_collision_overwrites(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        pool = WorkerPool()
        original = _make_card("bot", description="original")
        replacement = _make_card("bot", description="replacement")
        pool.register(original)
        with caplog.at_level(logging.WARNING, logger="neoagent.multi.worker"):
            pool.register(replacement)
        found = pool.find("bot")
        assert found is replacement
        assert any("bot" in record.message for record in caplog.records if record.levelno == logging.WARNING)

    def test_worker_pool_remove_existing(self) -> None:
        pool = WorkerPool()
        pool.register(_make_card("x"))
        assert pool.remove("x") is True
        assert pool.find("x") is None

    def test_worker_pool_remove_missing_returns_false(self) -> None:
        pool = WorkerPool()
        assert pool.remove("ghost") is False

    def test_worker_pool_list_all_returns_copy(self) -> None:
        pool = WorkerPool()
        pool.register(_make_card("sole"))
        copy = pool.list_all()
        copy.clear()
        assert pool.size == 1

    def test_worker_pool_size(self) -> None:
        pool = WorkerPool()
        assert pool.size == 0
        pool.register(_make_card("p"))
        assert pool.size == 1
        pool.register(_make_card("q"))
        assert pool.size == 2
        pool.remove("p")
        assert pool.size == 1
