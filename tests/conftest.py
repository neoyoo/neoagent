from __future__ import annotations
"""
Root conftest: inject minimal stubs into sys.modules so every test module
can import neoagent.providers.anthropic and neoagent.providers.openai
without the real SDKs installed.
"""
import sys
from unittest.mock import MagicMock

if "anthropic" not in sys.modules:
    stub = MagicMock()
    stub.AsyncAnthropic = MagicMock
    sys.modules["anthropic"] = stub

if "openai" not in sys.modules:
    stub = MagicMock()
    stub.AsyncOpenAI = MagicMock
    sys.modules["openai"] = stub
