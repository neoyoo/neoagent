# tests/channels/test_base.py
from __future__ import annotations

import pytest
from neoagent.channels.base import Channel


class ConcreteChannel(Channel):
    """Minimal concrete implementation for testing the ABC contract."""

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def serve_forever(self) -> None:
        pass  # no-op in tests


def test_channel_cannot_be_instantiated_directly():
    """Channel is abstract — direct instantiation must fail."""
    with pytest.raises(TypeError):
        Channel(agent=None)  # type: ignore[arg-type]


def test_concrete_channel_can_be_instantiated():
    """A complete subclass can be instantiated."""
    ch = ConcreteChannel(agent=None)  # type: ignore[arg-type]
    assert ch._agent is None


@pytest.mark.asyncio
async def test_start_stop_lifecycle():
    """ConcreteChannel start/stop lifecycle works."""
    ch = ConcreteChannel(agent=None)  # type: ignore[arg-type]
    await ch.start()
    assert ch._started is True
    await ch.stop()
    assert ch._started is False


@pytest.mark.asyncio
async def test_serve_forever_is_callable():
    """serve_forever() is a no-op in ConcreteChannel — verify the contract."""
    ch = ConcreteChannel(agent=None)  # type: ignore[arg-type]
    await ch.serve_forever()  # should not raise
