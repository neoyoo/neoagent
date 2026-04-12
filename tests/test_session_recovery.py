"""
tests/test_session_recovery.py
Session Recovery 相关测试。
"""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from neoagent.session import Session, JsonFileStorage
from neoagent.core.types import Message, TextBlock, ToolUseBlock, ToolResult
from neoagent.providers.base import Response


def _make_end_turn_response(text="done"):
    return Response(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
    )

def _make_tool_use_response(tool_name="echo", call_id="c1"):
    return Response(
        content=[ToolUseBlock(id=call_id, name=tool_name, input={"msg": "hi"})],
        stop_reason="tool_use",
        input_tokens=15,
        output_tokens=8,
    )


class TestQueryLoopAutoSave:

    @pytest.mark.asyncio
    async def test_auto_save_after_end_turn(self, tmp_path):
        """end_turn: save_if_storage() called at least once."""
        from neoagent.core.loop import QueryLoop
        from neoagent.tools.registry import ToolRegistry
        from neoagent.core.prompt import PromptBuilder, PromptSection

        storage = JsonFileStorage(tmp_path)
        session = Session.create(session_id="end-turn-save")
        session.bind_storage(storage)
        session.messages.append(Message(role="user", content="hello"))

        provider = MagicMock()
        provider.get_context_window.return_value = 200_000
        provider.create = AsyncMock(return_value=_make_end_turn_response("world"))

        registry = ToolRegistry()
        pb = PromptBuilder()
        pb.add_section(PromptSection(name="sys", content="test", priority=0, is_static=True))

        loop = QueryLoop(provider=provider, tool_registry=registry, prompt_builder=pb)

        call_count = [0]
        original = session.save_if_storage
        def counting_save():
            call_count[0] += 1
            original()
        session.save_if_storage = counting_save

        await loop.run(session=session)
        assert call_count[0] >= 1
        assert (tmp_path / "end-turn-save.json").exists()

    @pytest.mark.asyncio
    async def test_auto_save_after_tool_use_turn(self, tmp_path):
        """tool_use + end_turn: save called at least twice."""
        from neoagent.core.loop import QueryLoop
        from neoagent.tools.registry import ToolRegistry
        from neoagent.tools.executor import ToolExecutor
        from neoagent.tools.permission import PermissionChecker
        from neoagent.core.prompt import PromptBuilder, PromptSection
        from neoagent.events import EventBus
        from neoagent.tools.base import BaseTool

        storage = JsonFileStorage(tmp_path)
        session = Session.create(session_id="tool-use-save")
        session.bind_storage(storage)
        session.messages.append(Message(role="user", content="use tool"))

        provider = MagicMock()
        provider.get_context_window.return_value = 200_000
        provider.create = AsyncMock(side_effect=[
            _make_tool_use_response("mock_tool", "c1"),
            _make_end_turn_response("done"),
        ])

        registry = ToolRegistry()
        bus = EventBus()
        permission = PermissionChecker(auto_approve=True)
        executor = ToolExecutor(registry=registry, permission_checker=permission, event_bus=bus)

        class MockInput(BaseModel):
            msg: str = ""

        class MockTool(BaseTool):
            name = "mock_tool"
            description = "test"
            input_model: type[BaseModel] = MockInput
            permission = "auto"
            is_concurrent_safe = True

            async def execute(self, input: BaseModel) -> ToolResult:
                return ToolResult(call_id="", output="ok")

        registry.register(MockTool())

        pb = PromptBuilder()
        pb.add_section(PromptSection(name="sys", content="test", priority=0, is_static=True))

        loop = QueryLoop(provider=provider, tool_registry=registry, tool_executor=executor, prompt_builder=pb, event_bus=bus)

        save_calls = []
        original = session.save_if_storage
        def tracking_save():
            save_calls.append(1)
            original()
        session.save_if_storage = tracking_save

        await loop.run(session=session)
        assert len(save_calls) >= 2

    @pytest.mark.asyncio
    async def test_auto_save_skipped_without_storage(self):
        """No storage bound: loop completes without error."""
        from neoagent.core.loop import QueryLoop
        from neoagent.tools.registry import ToolRegistry
        from neoagent.core.prompt import PromptBuilder, PromptSection

        session = Session.create(session_id="no-storage")
        session.messages.append(Message(role="user", content="hello"))

        provider = MagicMock()
        provider.get_context_window.return_value = 200_000
        provider.create = AsyncMock(return_value=_make_end_turn_response("ok"))

        registry = ToolRegistry()
        pb = PromptBuilder()
        pb.add_section(PromptSection(name="sys", content="test", priority=0, is_static=True))

        loop = QueryLoop(provider=provider, tool_registry=registry, prompt_builder=pb)
        result = await loop.run(session=session)
        assert result.reason == "completed"
