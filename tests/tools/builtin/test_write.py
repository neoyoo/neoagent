from __future__ import annotations
import pytest
from neoagent.tools.builtin.write import WriteTool, WriteInput


class TestWriteTool:
    def test_permission_is_ask(self):
        assert WriteTool().permission == "ask"

    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        f = tmp_path / "new.txt"
        t = WriteTool()
        r = await t.execute(WriteInput(file_path=str(f), content="hello"))
        assert r.is_error is False
        assert f.read_text() == "hello"

    @pytest.mark.asyncio
    async def test_overwrite(self, tmp_path):
        f = tmp_path / "exist.txt"
        f.write_text("old")
        t = WriteTool()
        await t.execute(WriteInput(file_path=str(f), content="new"))
        assert f.read_text() == "new"

    @pytest.mark.asyncio
    async def test_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "a" / "b" / "c.txt"
        t = WriteTool()
        r = await t.execute(WriteInput(file_path=str(f), content="deep"))
        assert r.is_error is False
        assert f.read_text() == "deep"
