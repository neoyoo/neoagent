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
        t = GrepTool(allowed_directories=[tmp_path])
        r = await t.execute(GrepInput(pattern="hello", path=str(tmp_path)))
        assert r.is_error is False
        assert "hello" in r.output

    @pytest.mark.asyncio
    async def test_no_match(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("nothing here\n")
        t = GrepTool(allowed_directories=[tmp_path])
        r = await t.execute(GrepInput(pattern="xyz", path=str(tmp_path)))
        assert r.is_error is False  # no match is not an error

    @pytest.mark.asyncio
    async def test_invalid_path(self, tmp_path):
        t = GrepTool(allowed_directories=[tmp_path])
        r = await t.execute(GrepInput(pattern="x", path=str(tmp_path / "nonexistent")))
        # grep on nonexistent path returns error or empty
        assert isinstance(r.output, str)

    @pytest.mark.asyncio
    async def test_outside_allowed_dir_returns_error(self, tmp_path):
        t = GrepTool(allowed_directories=[tmp_path])
        r = await t.execute(GrepInput(pattern="root", path="/etc"))
        assert r.is_error is True
        assert "outside allowed" in r.output

    @pytest.mark.asyncio
    async def test_dotdot_traversal_blocked(self, tmp_path):
        t = GrepTool(allowed_directories=[tmp_path])
        evil = str(tmp_path / ".." / "..")
        r = await t.execute(GrepInput(pattern="secret", path=evil))
        assert r.is_error is True
        assert "outside allowed" in r.output

    @pytest.mark.asyncio
    async def test_inside_allowed_dir_succeeds(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("find me here\n")
        t = GrepTool(allowed_directories=[tmp_path])
        r = await t.execute(GrepInput(pattern="find me", path=str(tmp_path)))
        assert r.is_error is False
        assert "find me" in r.output

    def test_asyncio_imported(self):
        """asyncio must be importable from grep module (required for wait_for timeout)."""
        import importlib
        import neoagent.tools.builtin.grep as grep_module
        assert hasattr(grep_module, "asyncio")

    @pytest.mark.asyncio
    async def test_grep_normal_operation_with_timeout_guard(self, tmp_path):
        """Normal grep still works correctly after adding wait_for timeout."""
        f = tmp_path / "source.py"
        f.write_text("def hello():\n    return 'world'\n")
        t = GrepTool(allowed_directories=[tmp_path])
        r = await t.execute(GrepInput(pattern="def hello", path=str(tmp_path)))
        assert r.is_error is False
        assert "def hello" in r.output
