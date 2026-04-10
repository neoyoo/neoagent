from __future__ import annotations
import pytest
from neoagent.tools.builtin.glob import GlobTool, GlobInput

class TestGlobTool:
    def test_permission_auto(self):
        assert GlobTool().permission == "auto"

    def test_concurrent_safe(self):
        assert GlobTool().is_concurrent_safe is True

    @pytest.mark.asyncio
    async def test_match_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        (tmp_path / "c.txt").write_text("z")
        t = GlobTool()
        r = await t.execute(GlobInput(pattern="*.py", path=str(tmp_path)))
        assert r.is_error is False
        assert "a.py" in r.output
        assert "b.py" in r.output
        assert "c.txt" not in r.output

    @pytest.mark.asyncio
    async def test_no_match(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        t = GlobTool()
        r = await t.execute(GlobInput(pattern="*.rs", path=str(tmp_path)))
        assert r.is_error is False

    @pytest.mark.asyncio
    async def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("x")
        t = GlobTool()
        r = await t.execute(GlobInput(pattern="**/*.py", path=str(tmp_path)))
        assert "deep.py" in r.output
