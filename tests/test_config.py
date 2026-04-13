from __future__ import annotations
import pytest
from neoagent.config import NeoAgentConfig


class TestNeoAgentConfig:
    def test_default_values(self):
        c = NeoAgentConfig(api_key="test-key")
        assert c.provider == "anthropic"
        assert c.max_turns == 30
        assert c.auto_approve_tools == False  # noqa: E712

    def test_repr_masks_api_key(self):
        c = NeoAgentConfig(api_key="sk-secret-key-12345")
        r = repr(c)
        assert "sk-s***" in r
        assert "sk-secret-key-12345" not in r

    def test_repr_handles_empty_api_key(self):
        c = NeoAgentConfig(api_key="")
        r = repr(c)
        assert "api_key=''" in r
