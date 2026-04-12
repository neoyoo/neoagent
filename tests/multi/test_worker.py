from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from neoagent.multi.worker import WorkerCard, WorkerPool, _create_worker_agent


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


# ---------------------------------------------------------------------------
# TestCreateWorkerAgent
# ---------------------------------------------------------------------------


def _make_orchestrator(model: str = "claude-opus-4", max_depth: int = 2, tool_pool: dict | None = None):
    """Build a minimal orchestrator mock with the attributes _create_worker_agent needs."""
    from neoagent.config import NeoAgentConfig
    from neoagent.events import EventBus

    config = NeoAgentConfig(api_key="test-key", model=model)
    orch = MagicMock()
    orch.config = config
    orch._tool_pool = tool_pool if tool_pool is not None else {}
    orch.max_depth = max_depth
    # Provide a real EventBus so _setup_event_bubble can call subscribe_all
    orch._event_bus = EventBus()
    return orch


def _make_worker_card(
    name: str = "worker",
    model: str | None = None,
    tools: tuple[str, ...] = (),
    instruction: str = "You are a worker.",
) -> WorkerCard:
    return WorkerCard(
        name=name,
        description="Test worker",
        instruction=instruction,
        tags=(),
        model=model,
        tools=tools,
    )


class TestCreateWorkerAgent:
    """Tests for _create_worker_agent factory function."""

    def test_returns_neoagent(self) -> None:
        from neoagent.agent import NeoAgent

        orch = _make_orchestrator()
        card = _make_worker_card()

        with patch("neoagent.agent._create_provider") as mock_provider:
            mock_provider.return_value = MagicMock()
            result = _create_worker_agent(card, orch, depth=0)

        assert isinstance(result, NeoAgent)

    def test_uses_card_model(self) -> None:
        orch = _make_orchestrator(model="claude-opus-4")
        card = _make_worker_card(model="claude-haiku-4")

        with patch("neoagent.agent._create_provider") as mock_provider:
            mock_provider.return_value = MagicMock()
            result = _create_worker_agent(card, orch, depth=0)

        assert result._config.model == "claude-haiku-4"

    def test_falls_back_to_orchestrator_model(self) -> None:
        orch = _make_orchestrator(model="claude-opus-4")
        card = _make_worker_card(model=None)

        with patch("neoagent.agent._create_provider") as mock_provider:
            mock_provider.return_value = MagicMock()
            result = _create_worker_agent(card, orch, depth=0)

        assert result._config.model == "claude-opus-4"

    def test_system_prompt_set_from_card_instruction(self) -> None:
        orch = _make_orchestrator()
        card = _make_worker_card(instruction="You are a precise coder.")

        with patch("neoagent.agent._create_provider") as mock_provider:
            mock_provider.return_value = MagicMock()
            result = _create_worker_agent(card, orch, depth=0)

        assert result._config.system_prompt == "You are a precise coder."

    def test_registers_only_authorized_tools(self) -> None:
        from neoagent.tools.base import BaseTool
        from pydantic import BaseModel
        from neoagent.core.types import ToolResult

        class _FakeTool(BaseTool):
            name = "read"
            description = "reads"
            input_model = BaseModel
            permission = "auto"

            async def execute(self, input):
                return ToolResult(content="ok")

        fake_read = _FakeTool()
        tool_pool = {"read": fake_read}

        orch = _make_orchestrator(tool_pool=tool_pool, max_depth=0)
        card = _make_worker_card(tools=("read",))

        with patch("neoagent.agent._create_provider") as mock_provider:
            mock_provider.return_value = MagicMock()
            result = _create_worker_agent(card, orch, depth=0)

        registered = result._registry.all_tools()
        assert "read" in registered

    def test_missing_tool_in_pool_is_skipped(self) -> None:
        """Card requests a tool that doesn't exist in pool — should not raise."""
        orch = _make_orchestrator(tool_pool={}, max_depth=0)
        card = _make_worker_card(tools=("nonexistent_tool",))

        with patch("neoagent.agent._create_provider") as mock_provider:
            mock_provider.return_value = MagicMock()
            # Should not raise
            result = _create_worker_agent(card, orch, depth=0)

        registered = result._registry.all_tools()
        assert "nonexistent_tool" not in registered

    def test_depth_below_max_registers_spawn_worker(self) -> None:
        orch = _make_orchestrator(max_depth=3)
        card = _make_worker_card()

        with patch("neoagent.agent._create_provider") as mock_provider:
            mock_provider.return_value = MagicMock()
            result = _create_worker_agent(card, orch, depth=2)  # 2 < 3

        schemas = result._registry.get_schemas()
        schema_names = {s["name"] for s in schemas}
        assert "spawn_worker" in schema_names

    def test_depth_at_max_no_spawn_worker(self) -> None:
        orch = _make_orchestrator(max_depth=3)
        card = _make_worker_card()

        with patch("neoagent.agent._create_provider") as mock_provider:
            mock_provider.return_value = MagicMock()
            result = _create_worker_agent(card, orch, depth=3)  # 3 == max_depth

        schemas = result._registry.get_schemas()
        schema_names = {s["name"] for s in schemas}
        assert "spawn_worker" not in schema_names

    def test_create_worker_agent_uses_real_orchestrator_config(self) -> None:
        """_create_worker_agent must read orchestrator.config, not orchestrator._config."""
        from unittest.mock import patch, MagicMock
        from neoagent.config import NeoAgentConfig
        from neoagent.multi.orchestrator import Orchestrator

        config = NeoAgentConfig(api_key="test-key", model="claude-haiku-4-5")
        with patch("neoagent.agent._create_provider") as mock_prov:
            mock_prov.return_value = MagicMock()
            orch = Orchestrator(config)

        card = WorkerCard(
            name="w",
            description="",
            instruction="You help.",
            tags=(),
            model=None,
            tools=(),
        )
        with patch("neoagent.agent._create_provider") as mock_prov:
            mock_prov.return_value = MagicMock()
            with patch("neoagent.multi.events._setup_event_bubble"):
                # Should NOT raise AttributeError
                agent = _create_worker_agent(card, orch, depth=0)
        assert agent is not None

    def test_event_bubble_called(self) -> None:
        orch = _make_orchestrator(max_depth=0)
        card = _make_worker_card(name="my-worker")

        # Patch at the source module where _setup_event_bubble is defined.
        # The function is imported inside _create_worker_agent via
        # "from neoagent.multi.events import _setup_event_bubble", so we
        # intercept it at the module level.
        with patch("neoagent.agent._create_provider") as mock_provider, \
             patch("neoagent.multi.events._setup_event_bubble") as mock_bubble:
            mock_provider.return_value = MagicMock()
            result = _create_worker_agent(card, orch, depth=0)

        mock_bubble.assert_called_once()
        args = mock_bubble.call_args[0]
        # args: (agent, orchestrator, worker_name, task_id, depth)
        from neoagent.agent import NeoAgent
        assert isinstance(args[0], NeoAgent)
        assert args[1] is orch
        assert args[2] == "my-worker"
        # args[3] is a UUID string — just check it's a non-empty string
        assert isinstance(args[3], str) and len(args[3]) > 0
        assert args[4] == 0

    def test_create_worker_agent_tool_instances_are_isolated(self) -> None:
        """Each worker should get its own copy of tools, not the shared instance."""
        import copy
        from neoagent.config import NeoAgentConfig
        from neoagent.core.types import ToolResult
        from neoagent.tools.base import BaseTool
        from pydantic import BaseModel as _BM

        class StatefulTool(BaseTool):
            name = "stateful_test_tool"
            description = "test"
            class _In(_BM):
                x: int = 0
            input_model = _In
            permission = "auto"

            async def execute(self, input):
                return ToolResult(call_id="", output="ok")

        original_tool = StatefulTool()

        orch = MagicMock()
        orch.config = NeoAgentConfig(api_key="test", model="claude-haiku-4-5")
        orch._tool_pool = {"stateful_test_tool": original_tool}
        orch.max_depth = 2

        card = WorkerCard(
            name="w",
            description="",
            instruction="help",
            tags=(),
            model=None,
            tools=("stateful_test_tool",),
        )

        with patch("neoagent.agent._create_provider") as mock_prov:
            mock_prov.return_value = MagicMock()
            with patch("neoagent.multi.worker._setup_event_bubble"):
                agent = _create_worker_agent(card, orch, depth=0)

        registered_tool = agent._registry.get_tool("stateful_test_tool")
        assert registered_tool is not original_tool, "Worker should have its own tool copy"
