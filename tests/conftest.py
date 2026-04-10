from __future__ import annotations
"""
Root conftest: inject a minimal anthropic stub into sys.modules so every
test module can import neoagent.providers.anthropic without the real SDK.
"""
import sys
from unittest.mock import MagicMock

if "anthropic" not in sys.modules:
    stub = MagicMock()
    stub.AsyncAnthropic = MagicMock
    sys.modules["anthropic"] = stub
