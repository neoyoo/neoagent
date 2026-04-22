# neoagent/v2/errors.py
"""Error types for neoagent v2 strategies.

Spec refs:
  § 10.3a (lines 1710-1766) — Compressor Output Contract
"""


class CompressionError(Exception):
    """Raised when OneShotCompressionStrategy exhausts all retries.

    This is only raised on JSON parse failure / top-level structure error
    after max_retries attempts. Hard contract violations (rules 1-5) use
    degrade strategy (a) — drop working_memory_delta, no raise.
    """
    pass
