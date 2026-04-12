# neoagent/channels/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neoagent.agent import NeoAgent


class Channel(ABC):
    """Abstract base for neoagent communication channels.

    Subclasses implement a specific transport (HTTP, MQ, WebSocket, etc.)
    and wire it to a NeoAgent instance.

    Lifecycle::

        channel = MyChannel(agent, ...)
        await channel.start()          # bind port / connect broker
        await channel.serve_forever()  # block until stop() called
        await channel.stop()           # graceful shutdown
    """

    _agent: "NeoAgent"

    def __init__(self, agent: "NeoAgent") -> None:
        self._agent = agent

    @abstractmethod
    async def start(self) -> None:
        """Initialize the channel (bind port, connect to broker, etc.)."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the channel."""

    @abstractmethod
    async def serve_forever(self) -> None:
        """Block until stop() is called.

        ``stop()`` may be called concurrently from a signal handler or
        another asyncio task. Implementations must handle that safely.
        """
