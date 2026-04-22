# neoagent/v2/strategies/__init__.py
"""Compression strategy implementations for neoagent v2.

Available strategies:
  - OneShotCompressionStrategy: single LLM call, cheapest option (§ 10.3)
"""

from neoagent.v2.strategies.oneshot_compression import OneShotCompressionStrategy

__all__ = ["OneShotCompressionStrategy"]
