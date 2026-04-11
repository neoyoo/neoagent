from __future__ import annotations
import pytest
from neoagent.tools.builtin.read import ReadTool, ReadInput
from neoagent.core.types import ToolResult


class TestReadTool:
    def test_schema(self):
        t = ReadTool()
        s = t.get_schema()
        assert s["name"] == "read"
        assert "file_path" in s["input_schema"]["properties"]

    def test_permission_is_auto(self):
        assert ReadTool().permission == "auto"

    def test_concurrent_safe(self):
        assert ReadTool().is_concurrent_safe is True

    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        t = ReadTool(allowed_directories=[tmp_path])
        r = await t.execute(ReadInput(file_path=str(f)))
        assert isinstance(r, ToolResult)
        assert r.is_error is False
        assert "1\tline1" in r.output
        assert "2\tline2" in r.output

    @pytest.mark.asyncio
    async def test_offset(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("\n".join(f"line{i}" for i in range(10)))
        t = ReadTool(allowed_directories=[tmp_path])
        r = await t.execute(ReadInput(file_path=str(f), offset=5))
        assert "line0" not in r.output
        assert "line5" in r.output

    @pytest.mark.asyncio
    async def test_limit(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("\n".join(f"line{i}" for i in range(10)))
        t = ReadTool(allowed_directories=[tmp_path])
        r = await t.execute(ReadInput(file_path=str(f), limit=3))
        assert "line0" in r.output
        assert "line2" in r.output
        assert "line3" not in r.output

    @pytest.mark.asyncio
    async def test_file_not_found(self, tmp_path):
        t = ReadTool(allowed_directories=[tmp_path])
        r = await t.execute(ReadInput(file_path=str(tmp_path / "nonexistent.txt")))
        assert r.is_error is True

    @pytest.mark.asyncio
    async def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        t = ReadTool(allowed_directories=[tmp_path])
        r = await t.execute(ReadInput(file_path=str(f)))
        assert r.is_error is False

    @pytest.mark.asyncio
    async def test_outside_allowed_dir_returns_error(self, tmp_path):
        t = ReadTool(allowed_directories=[tmp_path])
        r = await t.execute(ReadInput(file_path="/etc/hostname"))
        assert r.is_error is True
        assert "outside allowed" in r.output

    @pytest.mark.asyncio
    async def test_dotdot_traversal_blocked(self, tmp_path):
        t = ReadTool(allowed_directories=[tmp_path])
        evil = str(tmp_path / ".." / ".." / "etc" / "passwd")
        r = await t.execute(ReadInput(file_path=evil))
        assert r.is_error is True
        assert "outside allowed" in r.output

    @pytest.mark.asyncio
    async def test_inside_allowed_dir_succeeds(self, tmp_path):
        f = tmp_path / "safe.txt"
        f.write_text("safe content")
        t = ReadTool(allowed_directories=[tmp_path])
        r = await t.execute(ReadInput(file_path=str(f)))
        assert r.is_error is False
        assert "safe content" in r.output
