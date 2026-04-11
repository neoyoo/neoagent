from __future__ import annotations
"""
Inject minimal stubs into sys.modules so tests can import
neoagent.providers.anthropic and neoagent.providers.openai
without the real SDKs installed.
"""
import sys
from unittest.mock import MagicMock

# Only stub if the real package isn't available
if "anthropic" not in sys.modules:
    stub = MagicMock()
    stub.AsyncAnthropic = MagicMock
    sys.modules["anthropic"] = stub

if "openai" not in sys.modules:
    stub = MagicMock()
    stub.AsyncOpenAI = MagicMock
    sys.modules["openai"] = stub
