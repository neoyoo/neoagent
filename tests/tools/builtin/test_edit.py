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
        t = EditTool(allowed_directories=[tmp_path])
        r = await t.execute(EditInput(file_path=str(f), old_string="hello", new_string="goodbye"))
        assert r.is_error is False
        assert f.read_text() == "goodbye world"

    @pytest.mark.asyncio
    async def test_not_found_string(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        t = EditTool(allowed_directories=[tmp_path])
        r = await t.execute(EditInput(file_path=str(f), old_string="xyz", new_string="abc"))
        assert r.is_error is True
        assert f.read_text() == "hello"  # unchanged

    @pytest.mark.asyncio
    async def test_not_unique(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("aa bb aa")
        t = EditTool(allowed_directories=[tmp_path])
        r = await t.execute(EditInput(file_path=str(f), old_string="aa", new_string="cc"))
        assert r.is_error is True
        assert f.read_text() == "aa bb aa"  # unchanged

    @pytest.mark.asyncio
    async def test_file_not_exist(self, tmp_path):
        t = EditTool(allowed_directories=[tmp_path])
        r = await t.execute(EditInput(file_path=str(tmp_path / "nonexistent.txt"), old_string="a", new_string="b"))
        assert r.is_error is True

    @pytest.mark.asyncio
    async def test_outside_allowed_dir_returns_error(self, tmp_path):
        t = EditTool(allowed_directories=[tmp_path])
        r = await t.execute(EditInput(file_path="/etc/hosts", old_string="a", new_string="b"))
        assert r.is_error is True
        assert "outside allowed" in r.output

    @pytest.mark.asyncio
    async def test_dotdot_traversal_blocked(self, tmp_path):
        t = EditTool(allowed_directories=[tmp_path])
        evil = str(tmp_path / ".." / "sensitive.txt")
        r = await t.execute(EditInput(file_path=evil, old_string="a", new_string="b"))
        assert r.is_error is True
        assert "outside allowed" in r.output

    @pytest.mark.asyncio
    async def test_inside_allowed_dir_succeeds(self, tmp_path):
        f = tmp_path / "target.txt"
        f.write_text("original text")
        t = EditTool(allowed_directories=[tmp_path])
        r = await t.execute(EditInput(file_path=str(f), old_string="original", new_string="updated"))
        assert r.is_error is False
        assert f.read_text() == "updated text"
