from __future__ import annotations
import pytest
from neoagent.tools.builtin.edit import EditTool, EditInput


class TestEditTool:
    def test_permission_is_ask(self):
        assert EditTool().permission == "ask"

    @pytest.mark.asyncio
    async def test_replace(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello world")
        t = EditTool()
        r = await t.execute(EditInput(file_path=str(f), old_string="hello", new_string="goodbye"))
        assert r.is_error is False
        assert f.read_text() == "goodbye world"

    @pytest.mark.asyncio
    async def test_not_found_string(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        t = EditTool()
        r = await t.execute(EditInput(file_path=str(f), old_string="xyz", new_string="abc"))
        assert r.is_error is True
        assert f.read_text() == "hello"  # unchanged

    @pytest.mark.asyncio
    async def test_not_unique(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("aa bb aa")
        t = EditTool()
        r = await t.execute(EditInput(file_path=str(f), old_string="aa", new_string="cc"))
        assert r.is_error is True
        assert f.read_text() == "aa bb aa"  # unchanged

    @pytest.mark.asyncio
    async def test_file_not_exist(self):
        t = EditTool()
        r = await t.execute(EditInput(file_path="/nonexistent.txt", old_string="a", new_string="b"))
        assert r.is_error is True
