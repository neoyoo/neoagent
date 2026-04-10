from __future__ import annotations
"""
Inject a minimal anthropic stub into sys.modules so tests can import
neoagent.providers.anthropic without the real anthropic SDK installed.
"""
import sys
from unittest.mock import MagicMock

# Only stub if the real package isn't available
if "anthropic" not in sys.modules:
    stub = MagicMock()
    stub.AsyncAnthropic = MagicMock
    sys.modules["anthropic"] = stub
