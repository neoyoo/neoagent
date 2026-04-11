from __future__ import annotations
import pytest
from neoagent.tools.builtin.write import WriteTool, WriteInput


class TestWriteTool:
    def test_permission_is_ask(self):
        assert WriteTool().permission == "ask"

    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        f = tmp_path / "new.txt"
        t = WriteTool(allowed_directories=[tmp_path])
        r = await t.execute(WriteInput(file_path=str(f), content="hello"))
        assert r.is_error is False
        assert f.read_text() == "hello"

    @pytest.mark.asyncio
    async def test_overwrite(self, tmp_path):
        f = tmp_path / "exist.txt"
        f.write_text("old")
        t = WriteTool(allowed_directories=[tmp_path])
        await t.execute(WriteInput(file_path=str(f), content="new"))
        assert f.read_text() == "new"

    @pytest.mark.asyncio
    async def test_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "a" / "b" / "c.txt"
        t = WriteTool(allowed_directories=[tmp_path])
        r = await t.execute(WriteInput(file_path=str(f), content="deep"))
        assert r.is_error is False
        assert f.read_text() == "deep"

    @pytest.mark.asyncio
    async def test_outside_allowed_dir_returns_error(self, tmp_path):
        t = WriteTool(allowed_directories=[tmp_path])
        r = await t.execute(WriteInput(file_path="/tmp/evil.txt", content="pwned"))
        assert r.is_error is True
        assert "outside allowed" in r.output

    @pytest.mark.asyncio
    async def test_dotdot_traversal_blocked(self, tmp_path):
        t = WriteTool(allowed_directories=[tmp_path])
        evil = str(tmp_path / ".." / "evil.txt")
        r = await t.execute(WriteInput(file_path=evil, content="bad"))
        assert r.is_error is True
        assert "outside allowed" in r.output

    @pytest.mark.asyncio
    async def test_inside_allowed_dir_succeeds(self, tmp_path):
        f = tmp_path / "ok.txt"
        t = WriteTool(allowed_directories=[tmp_path])
        r = await t.execute(WriteInput(file_path=str(f), content="good"))
        assert r.is_error is False
        assert f.read_text() == "good"
