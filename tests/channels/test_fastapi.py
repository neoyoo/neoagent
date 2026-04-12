# tests/channels/test_fastapi.py
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import httpx

from neoagent.channels.fastapi_channel import FastAPIChannel, _event_to_sse_data
from neoagent.core.types import ConversationResult, Message, TextBlock, Turn
from neoagent.events import TurnCompleteEvent, ToolCallEvent, ToolResultEvent, ProviderResponseEvent


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_agent(result: ConversationResult | None = None) -> MagicMock:
    """Return a MagicMock NeoAgent with agent.run() → result."""
    agent = MagicMock()
    agent.event_bus = MagicMock()
    agent.event_bus.subscribe = MagicMock()
    agent.event_bus.unsubscribe = MagicMock()
    if result is None:
        result = ConversationResult(
            turns=[Turn(
                response=Message(role="assistant", content=[TextBlock(text="hello")]),
                tool_calls=[],
                tool_results=[],
                stop_reason="end_turn",
            )],
            reason="completed",
        )
    agent.run = AsyncMock(return_value=result)
    return agent


def _make_channel(agent=None) -> FastAPIChannel:
    if agent is None:
        agent = _make_agent()
    return FastAPIChannel(agent)


def _async_client(channel: FastAPIChannel) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=channel.app),
        base_url="http://test",
    )


# ── health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_ok():
    channel = _make_channel()
    async with _async_client(channel) as client:
        resp = await client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── /v1/run (sync) ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_returns_conversation_result():
    agent = _make_agent()
    channel = _make_channel(agent)
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    async with _async_client(channel) as client:
        resp = await client.post("/v1/run", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["reason"] == "completed"
    assert len(body["turns"]) == 1
    assert body["turns"][0]["text"] == "hello"
    assert body["turns"][0]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_run_passes_max_turns_to_agent():
    agent = _make_agent()
    channel = _make_channel(agent)
    payload = {"messages": [{"role": "user", "content": "hi"}], "max_turns": 5}
    async with _async_client(channel) as client:
        await client.post("/v1/run", json=payload)
    _, kwargs = agent.run.call_args
    assert kwargs.get("max_turns") == 5


@pytest.mark.asyncio
async def test_run_converts_messages_to_Message_objects():
    agent = _make_agent()
    channel = _make_channel(agent)
    payload = {"messages": [{"role": "user", "content": "hello world"}]}
    async with _async_client(channel) as client:
        await client.post("/v1/run", json=payload)
    msgs = agent.run.call_args.kwargs["messages"]
    assert len(msgs) == 1
    assert msgs[0].role == "user"
    assert msgs[0].content == "hello world"


# ── _event_to_sse_data ────────────────────────────────────────────────────────

def test_event_to_sse_data_turn_complete():
    ev = TurnCompleteEvent(turn_index=0, stop_reason="end_turn", tool_call_count=0)
    data = json.loads(_event_to_sse_data(ev))
    assert data["type"] == "turn_complete"
    assert data["turn_index"] == 0
    assert data["stop_reason"] == "end_turn"


def test_event_to_sse_data_tool_call():
    ev = ToolCallEvent(name="bash", input_data={"cmd": "ls"}, call_id="abc")
    data = json.loads(_event_to_sse_data(ev))
    assert data["type"] == "tool_call"
    assert data["name"] == "bash"
    assert data["input"]["cmd"] == "ls"


def test_event_to_sse_data_tool_result():
    ev = ToolResultEvent(name="bash", call_id="abc", output="ok", is_error=False)
    data = json.loads(_event_to_sse_data(ev))
    assert data["type"] == "tool_result"
    assert data["is_error"] is False
    assert data["output"] == "ok"


def test_event_to_sse_data_provider_response():
    content_tuple = (TextBlock(text="hi"),)
    ev = ProviderResponseEvent(
        content=content_tuple,
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
        turn=0,
    )
    data = json.loads(_event_to_sse_data(ev))
    assert data["type"] == "response"
    assert data["input_tokens"] == 10


# ── /v1/run/stream (SSE) ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_emits_done_event():
    """SSE stream must end with a 'done' event."""
    agent = _make_agent()
    channel = _make_channel(agent)

    async with _async_client(channel) as client:
        async with client.stream(
            "POST",
            "/v1/run/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            assert resp.status_code == 200
            lines = []
            async for line in resp.aiter_lines():
                lines.append(line)

    data_lines = [l for l in lines if l.startswith("data: ")]
    assert len(data_lines) >= 1
    last = json.loads(data_lines[-1][6:])
    assert last["type"] == "done"
    assert last["reason"] == "completed"


@pytest.mark.asyncio
async def test_stream_subscribes_and_unsubscribes_handlers():
    """Channel must subscribe to EventBus before run and unsubscribe after."""
    agent = _make_agent()
    subscribe_calls = []
    unsubscribe_calls = []

    def fake_subscribe(event_type, handler):
        subscribe_calls.append(event_type)

    def fake_unsubscribe(event_type, handler):
        unsubscribe_calls.append(event_type)

    agent.event_bus.subscribe = fake_subscribe
    agent.event_bus.unsubscribe = fake_unsubscribe
    channel = _make_channel(agent)

    async with _async_client(channel) as client:
        async with client.stream(
            "POST",
            "/v1/run/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            async for _ in resp.aiter_lines():
                pass

    assert len(subscribe_calls) > 0
    assert subscribe_calls == unsubscribe_calls  # symmetric cleanup


@pytest.mark.asyncio
async def test_stream_forwards_events_from_event_bus():
    """Events pushed to the EventBus during agent.run() must appear in the SSE stream."""
    import asyncio as _asyncio
    from neoagent.events import TurnCompleteEvent

    agent = _make_agent()
    captured_handlers: dict = {}

    def fake_subscribe(event_type, handler):
        captured_handlers[event_type] = handler

    def fake_unsubscribe(event_type, handler):
        pass

    agent.event_bus.subscribe = fake_subscribe
    agent.event_bus.unsubscribe = fake_unsubscribe

    # Patch agent.run to emit a TurnCompleteEvent via the captured handler
    # before completing, by running the handler just before returning.
    original_run = agent.run.side_effect

    async def run_with_event(messages, max_turns=None, session=None):
        # Simulate the event being emitted mid-run
        ev = TurnCompleteEvent(turn_index=0, stop_reason="end_turn", tool_call_count=0)
        if TurnCompleteEvent in captured_handlers:
            captured_handlers[TurnCompleteEvent](ev)
        return _make_agent().run.return_value  # reuse the default ConversationResult

    agent.run = AsyncMock(side_effect=run_with_event)

    channel = _make_channel(agent)

    async with _async_client(channel) as client:
        async with client.stream(
            "POST",
            "/v1/run/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            assert resp.status_code == 200
            lines = []
            async for line in resp.aiter_lines():
                lines.append(line)

    data_lines = [l for l in lines if l.startswith("data: ")]
    # Should have at least the turn_complete event + done event
    assert len(data_lines) >= 2

    # Find the turn_complete event
    events = [json.loads(l[6:]) for l in data_lines]
    turn_events = [e for e in events if e.get("type") == "turn_complete"]
    assert len(turn_events) == 1
    assert turn_events[0]["stop_reason"] == "end_turn"

    # Last event must be done
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_stream_endpoint_not_registered_when_streaming_false():
    """When streaming=False, POST /v1/run/stream must return 404."""
    agent = _make_agent()
    channel = FastAPIChannel(agent, streaming=False)
    async with _async_client(channel) as client:
        resp = await client.post(
            "/v1/run/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stream_emits_error_on_agent_exception():
    """When agent.run() raises, SSE stream must emit an error event."""
    agent = _make_agent()
    agent.event_bus.subscribe = MagicMock()
    agent.event_bus.unsubscribe = MagicMock()
    agent.run = AsyncMock(side_effect=RuntimeError("boom"))
    channel = _make_channel(agent)

    async with _async_client(channel) as client:
        async with client.stream(
            "POST",
            "/v1/run/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            assert resp.status_code == 200
            lines = []
            async for line in resp.aiter_lines():
                lines.append(line)

    data_lines = [l for l in lines if l.startswith("data: ")]
    assert len(data_lines) >= 1
    last = json.loads(data_lines[-1][6:])
    assert last["type"] == "error"
    assert "boom" in last["message"]


# ── S2: Security defaults ────────────────────────────────────────────────────

def test_default_host_is_localhost():
    """Default host must be 127.0.0.1, not 0.0.0.0."""
    agent = _make_agent()
    channel = FastAPIChannel(agent)
    assert channel._host == "127.0.0.1"


@pytest.mark.asyncio
async def test_api_key_rejects_unauthenticated():
    """When api_key is set, requests without auth header get 401."""
    agent = _make_agent()
    channel = FastAPIChannel(agent, api_key="test-secret-key")
    async with _async_client(channel) as client:
        resp = await client.post("/v1/run", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_api_key_accepts_valid_bearer():
    """When api_key is set, requests with correct Bearer token succeed."""
    agent = _make_agent()
    channel = FastAPIChannel(agent, api_key="test-secret-key")
    async with _async_client(channel) as client:
        resp = await client.post(
            "/v1/run",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer test-secret-key"},
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_api_key_rejects_wrong_bearer():
    """Wrong API key gets 401."""
    agent = _make_agent()
    channel = FastAPIChannel(agent, api_key="test-secret-key")
    async with _async_client(channel) as client:
        resp = await client.post(
            "/v1/run",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong-key"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_health_bypasses_api_key():
    """Health endpoint must work without API key even when auth is enabled."""
    agent = _make_agent()
    channel = FastAPIChannel(agent, api_key="test-secret-key")
    async with _async_client(channel) as client:
        resp = await client.get("/v1/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_no_api_key_allows_all():
    """When api_key is None (default), all requests pass."""
    agent = _make_agent()
    channel = FastAPIChannel(agent)  # no api_key
    async with _async_client(channel) as client:
        resp = await client.post("/v1/run", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200


# ── S3: Run serialization ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_lock_exists():
    """FastAPIChannel must have an asyncio.Lock for run serialization."""
    agent = _make_agent()
    channel = FastAPIChannel(agent)
    assert hasattr(channel, '_run_lock')
    assert isinstance(channel._run_lock, asyncio.Lock)


@pytest.mark.asyncio
async def test_concurrent_runs_are_serialized():
    """Two concurrent /v1/run requests must not overlap execution."""
    import asyncio as _asyncio

    execution_log = []

    async def slow_run(messages, max_turns=None, session=None):
        execution_log.append("start")
        await _asyncio.sleep(0.05)
        execution_log.append("end")
        return ConversationResult(
            turns=[Turn(
                response=Message(role="assistant", content=[TextBlock(text="ok")]),
                tool_calls=[], tool_results=[], stop_reason="end_turn",
            )],
            reason="completed",
        )

    agent = _make_agent()
    agent.run = AsyncMock(side_effect=slow_run)
    channel = FastAPIChannel(agent)

    async with _async_client(channel) as client:
        tasks = [
            client.post("/v1/run", json={"messages": [{"role": "user", "content": "a"}]}),
            client.post("/v1/run", json={"messages": [{"role": "user", "content": "b"}]}),
        ]
        results = await _asyncio.gather(*tasks)

    # Both must succeed
    assert all(r.status_code == 200 for r in results)
    # Execution must be serialized: start, end, start, end (not start, start, end, end)
    assert execution_log == ["start", "end", "start", "end"]
