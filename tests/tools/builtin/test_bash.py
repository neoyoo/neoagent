from __future__ import annotations
import pytest
from neoagent.tools.builtin.bash import BashTool, BashInput

class TestBashTool:
    def test_permission_is_ask(self):
        assert BashTool().permission == "ask"

    def test_not_concurrent_safe(self):
        assert BashTool().is_concurrent_safe is False

    @pytest.mark.asyncio
    async def test_simple_command(self):
        t = BashTool()
        r = await t.execute(BashInput(command="echo hello"))
        assert r.is_error is False
        assert "hello" in r.output

    @pytest.mark.asyncio
    async def test_command_failure(self):
        t = BashTool()
        r = await t.execute(BashInput(command="exit 1"))
        assert r.is_error is True

    @pytest.mark.asyncio
    async def test_timeout(self):
        t = BashTool()
        r = await t.execute(BashInput(command="sleep 10", timeout=1))
        assert r.is_error is True
        assert "timed out" in r.output.lower()
