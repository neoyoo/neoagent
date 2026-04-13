# neoagent/channels/fastapi_channel.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, AsyncGenerator

from neoagent.channels.base import Channel
from neoagent.core.types import ConversationResult, Message, TextBlock
from neoagent.events import (
    Event,
    ProviderResponseEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
)

if TYPE_CHECKING:
    from neoagent.agent import NeoAgent

logger = logging.getLogger(__name__)

# ── lazy imports ──────────────────────────────────────────────────────────────

def _get_fastapi():
    try:
        import fastapi
        return fastapi
    except ImportError:
        raise ImportError(
            "FastAPIChannel requires the 'fastapi' extra: "
            "pip install neoagent[fastapi]"
        )


def _get_uvicorn():
    try:
        import uvicorn
        return uvicorn
    except ImportError:
        raise ImportError(
            "FastAPIChannel requires the 'fastapi' extra: "
            "pip install neoagent[fastapi]"
        )


# ── Pydantic schemas ──────────────────────────────────────────────────────────

def _build_schemas():
    """Build Pydantic schemas. Called lazily to avoid eager fastapi import."""
    from pydantic import BaseModel

    class MessageIn(BaseModel):
        role: str
        content: str

    class RunRequest(BaseModel):
        messages: list[MessageIn]
        max_turns: int | None = None

    class RunResponse(BaseModel):
        turns: list[dict]
        reason: str

        @classmethod
        def from_result(cls, result: ConversationResult) -> "RunResponse":
            turns = []
            for turn in result.turns:
                content = turn.response.content
                if isinstance(content, list):
                    text = "\n".join(b.text for b in content if isinstance(b, TextBlock))
                else:
                    text = content if isinstance(content, str) else ""
                turns.append({
                    "stop_reason": turn.stop_reason,
                    "text": text,
                    "tool_calls": [{"name": tc.name, "id": tc.id} for tc in turn.tool_calls],
                })
            return cls(turns=turns, reason=result.reason)

    return MessageIn, RunRequest, RunResponse


# ── SSE event serialisation ───────────────────────────────────────────────────

_STREAM_EVENT_TYPES = (
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    ProviderResponseEvent,
)


def _event_to_sse_data(event: Event) -> str:
    """Serialize *event* to a JSON string suitable for an SSE ``data:`` line."""
    if isinstance(event, TurnCompleteEvent):
        payload = {
            "type": "turn_complete",
            "turn_index": event.turn_index,
            "stop_reason": event.stop_reason,
            "tool_call_count": event.tool_call_count,
        }
    elif isinstance(event, ToolCallEvent):
        payload = {
            "type": "tool_call",
            "name": event.name,
            "call_id": event.call_id,
            "input": event.input_data,
        }
    elif isinstance(event, ToolResultEvent):
        payload = {
            "type": "tool_result",
            "name": event.name,
            "call_id": event.call_id,
            "output": event.output,
            "is_error": event.is_error,
        }
    elif isinstance(event, ProviderResponseEvent):
        payload = {
            "type": "response",
            "stop_reason": event.stop_reason,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
            "turn": event.turn,
        }
    else:
        payload = {"type": "event", "class": type(event).__name__}
    return json.dumps(payload)


# ── FastAPIChannel ────────────────────────────────────────────────────────────

class FastAPIChannel(Channel):
    """HTTP channel built on FastAPI + uvicorn.

    Exposes two endpoints:

    - ``GET  /v1/health``      — liveness probe
    - ``POST /v1/run``         — synchronous, returns full ConversationResult JSON
    - ``POST /v1/run/stream``  — SSE stream of agent events

    Each request creates a fresh, stateless session (no cross-request memory).

    Usage::

        agent = NeoAgent(config)
        channel = FastAPIChannel(agent, host="127.0.0.1", port=8000)
        await channel.serve_forever()
    """

    def __init__(
        self,
        agent: "NeoAgent",
        host: str = "127.0.0.1",
        port: int = 8000,
        streaming: bool = True,
        api_key: str | None = None,
        cors_origins: list[str] | None = None,
    ) -> None:
        super().__init__(agent)
        self._host = host
        self._port = port
        self._streaming = streaming
        self._api_key = api_key
        self._cors_origins = cors_origins
        self._server: object | None = None
        self._app: object | None = None
        self._run_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Channel lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Build the ASGI app and configure uvicorn (does not bind port yet)."""
        fastapi = _get_fastapi()
        uvicorn = _get_uvicorn()

        self._app = self._build_app(fastapi)
        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)

    async def stop(self) -> None:
        """Signal uvicorn to shut down."""
        if self._server is not None:
            self._server.should_exit = True

    async def serve_forever(self) -> None:
        """Start uvicorn and block until stop() is called.

        Calls start() automatically if it hasn't been called yet.
        ``stop()`` may be called concurrently from a signal handler or another asyncio task.
        """
        if self._server is None:
            await self.start()
        await self._server.serve()

    # ------------------------------------------------------------------
    # ASGI app
    # ------------------------------------------------------------------

    def _build_app(self, fastapi_module) -> object:
        FastAPI = fastapi_module.FastAPI

        MessageIn, RunRequest, RunResponse = _build_schemas()

        app = FastAPI(title="neoagent", version="1.0")

        # CORS middleware
        if self._cors_origins is not None:
            from starlette.middleware.cors import CORSMiddleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self._cors_origins,
                allow_methods=["GET", "POST"],
                allow_headers=["Authorization", "Content-Type"],
                allow_credentials=False,
            )

        # API key authentication dependency
        if self._api_key is not None:
            expected_key = self._api_key

            async def _verify_api_key(request):
                """Check Authorization: Bearer <key> header."""
                from starlette.responses import JSONResponse
                auth = request.headers.get("Authorization", "")
                if not auth.startswith("Bearer ") or auth[7:] != expected_key:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key"},
                    )
                return None

            @app.middleware("http")
            async def auth_middleware(request, call_next):
                # Skip auth for health endpoint
                if request.url.path == "/v1/health":
                    return await call_next(request)
                error_response = await _verify_api_key(request)
                if error_response is not None:
                    return error_response
                return await call_next(request)

        @app.get("/v1/health")
        async def health():
            return {"status": "ok"}

        # NOTE: We must NOT rely on PEP 563 deferred annotation resolution here
        # because `RunRequest` only exists in this local scope. FastAPI cannot
        # resolve a string annotation like 'RunRequest' without the local
        # namespace. We use `__annotations__` injection to force the real type.
        async def run(req):
            msgs = [Message(role=m.role, content=m.content) for m in req.messages]
            async with self._run_lock:
                result = await self._agent.run(messages=msgs, max_turns=req.max_turns)
            return RunResponse.from_result(result)

        run.__annotations__ = {"req": RunRequest, "return": RunResponse}
        app.post("/v1/run", response_model=RunResponse)(run)

        if self._streaming:
            StreamingResponse = fastapi_module.responses.StreamingResponse

            async def run_stream(req):
                msgs = [Message(role=m.role, content=m.content) for m in req.messages]
                return StreamingResponse(
                    self._sse_generator(msgs, req.max_turns),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            run_stream.__annotations__ = {"req": RunRequest, "return": object}
            app.post("/v1/run/stream")(run_stream)

        return app

    @property
    def app(self) -> object:
        """Return the ASGI app (build it if needed). Useful for testing."""
        if self._app is None:
            fastapi = _get_fastapi()
            self._app = self._build_app(fastapi)
        return self._app

    # ------------------------------------------------------------------
    # SSE streaming
    # ------------------------------------------------------------------

    async def _sse_generator(
        self,
        messages: list[Message],
        max_turns: int | None,
    ) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted strings for each agent event.

        Uses asyncio.Queue to bridge the synchronous EventBus handlers
        into this async generator. Each subscribed event type pushes an
        event onto the queue; the generator drains it until a sentinel
        (None = success, Exception = error) is received.
        """
        queue: asyncio.Queue[Event | BaseException | None] = asyncio.Queue()

        # Register sync handlers that push events to the async queue.
        # Each handler only needs to capture `queue` (same for all event types).
        handlers: dict[type, object] = {}
        for event_type in _STREAM_EVENT_TYPES:
            def _make_handler():
                def _handler(ev: Event) -> None:
                    queue.put_nowait(ev)
                return _handler
            h = _make_handler()
            handlers[event_type] = h
            self._agent.event_bus.subscribe(event_type, h)

        async def _run() -> ConversationResult:
            async with self._run_lock:
                return await self._agent.run(messages=messages, max_turns=max_turns)

        agent_task = asyncio.create_task(_run())

        def _on_done(fut: asyncio.Future) -> None:
            if fut.cancelled():
                queue.put_nowait(asyncio.CancelledError("agent task was cancelled"))
            else:
                queue.put_nowait(fut.exception())  # None on success, Exception on error

        agent_task.add_done_callback(_on_done)

        try:
            while True:
                item = await queue.get()
                if item is None:
                    # Agent completed successfully
                    result = agent_task.result()
                    yield f"data: {json.dumps({'type': 'done', 'reason': result.reason})}\n\n"
                    break
                elif isinstance(item, BaseException):
                    yield f"data: {json.dumps({'type': 'error', 'message': str(item)})}\n\n"
                    break
                else:
                    yield f"data: {_event_to_sse_data(item)}\n\n"
        finally:
            for event_type, handler in handlers.items():
                self._agent.event_bus.unsubscribe(event_type, handler)
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
