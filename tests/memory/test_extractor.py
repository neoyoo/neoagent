from __future__ import annotations
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from neoagent.core.types import Message, TextBlock
from neoagent.memory.store import MemoryStore
from neoagent.memory.extractor import MemoryExtractor
from neoagent.providers.base import Response


def _make_provider(text: str) -> MagicMock:
    resp = Response(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=10,
    )
    provider = MagicMock()
    provider.create = AsyncMock(return_value=resp)
    return provider


def _msgs(n: int = 3) -> list[Message]:
    return [Message(role="user", content=f"message {i}") for i in range(n)]


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path)


@pytest.mark.asyncio
async def test_extract_below_threshold_returns_zero(store: MemoryStore) -> None:
    provider = _make_provider("[]")
    extractor = MemoryExtractor(provider, store)
    result = await extractor.extract(_msgs(), tool_calls_count=0, token_delta=100)
    assert result == 0
    provider.create.assert_not_called()


@pytest.mark.asyncio
async def test_extract_above_tool_threshold_stores_items(store: MemoryStore) -> None:
    items = [{"filename": "goals.md", "description": "project goals", "content": "# Goals\n- Build agent\n"}]
    provider = _make_provider(json.dumps(items))
    extractor = MemoryExtractor(provider, store)
    result = await extractor.extract(_msgs(), tool_calls_count=5, token_delta=0)
    assert result == 1
    assert "Build agent" in store.read_topic("goals.md")


@pytest.mark.asyncio
async def test_extract_above_token_threshold_stores_items(store: MemoryStore) -> None:
    items = [{"filename": "prefs.md", "description": "user prefs", "content": "# Prefs\n- Python\n"}]
    provider = _make_provider(json.dumps(items))
    extractor = MemoryExtractor(provider, store)
    result = await extractor.extract(_msgs(), tool_calls_count=0, token_delta=4000)
    assert result == 1


@pytest.mark.asyncio
async def test_extract_invalid_json_returns_zero(store: MemoryStore) -> None:
    provider = _make_provider("not json at all")
    extractor = MemoryExtractor(provider, store)
    result = await extractor.extract(_msgs(), tool_calls_count=5, token_delta=0)
    assert result == 0


@pytest.mark.asyncio
async def test_extract_empty_array_returns_zero(store: MemoryStore) -> None:
    provider = _make_provider("[]")
    extractor = MemoryExtractor(provider, store)
    result = await extractor.extract(_msgs(), tool_calls_count=5, token_delta=0)
    assert result == 0


@pytest.mark.asyncio
async def test_extract_rebuilds_index(store: MemoryStore) -> None:
    items = [{"filename": "ctx.md", "description": "project context", "content": "# Ctx\n- details\n"}]
    provider = _make_provider(json.dumps(items))
    extractor = MemoryExtractor(provider, store)
    await extractor.extract(_msgs(), tool_calls_count=5, token_delta=0)
    index = store.read_index()
    assert "ctx.md" in index
