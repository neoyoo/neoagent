# neoagent/v2/strategies/__init__.py
"""Strategy implementations for neoagent v2.

Available strategies:
  - OneShotCompressionStrategy: single LLM call compressor (§ 10.3)
  - OneShotMemoryReviewStrategy: single LLM call memory reviewer (§ 11.3)
"""

from neoagent.v2.strategies.oneshot_compression import OneShotCompressionStrategy
from neoagent.v2.strategies.oneshot_memory_review import OneShotMemoryReviewStrategy

__all__ = ["OneShotCompressionStrategy", "OneShotMemoryReviewStrategy"]
