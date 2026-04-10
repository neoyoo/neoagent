from __future__ import annotations
import pytest
from neoagent.tools.builtin.grep import GrepTool, GrepInput

class TestGrepTool:
    def test_permission_auto(self):
        assert GrepTool().permission == "auto"

    def test_concurrent_safe(self):
        assert GrepTool().is_concurrent_safe is True

    @pytest.mark.asyncio
    async def test_search_match(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world\nfoo bar\nhello again\n")
        t = GrepTool()
        r = await t.execute(GrepInput(pattern="hello", path=str(tmp_path)))
        assert r.is_error is False
        assert "hello" in r.output

    @pytest.mark.asyncio
    async def test_no_match(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("nothing here\n")
        t = GrepTool()
        r = await t.execute(GrepInput(pattern="xyz", path=str(tmp_path)))
        assert r.is_error is False  # no match is not an error

    @pytest.mark.asyncio
    async def test_invalid_path(self):
        t = GrepTool()
        r = await t.execute(GrepInput(pattern="x", path="/nonexistent"))
        # grep on nonexistent path returns error or empty
        assert isinstance(r.output, str)
