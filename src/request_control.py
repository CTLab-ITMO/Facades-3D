from __future__ import annotations

import asyncio
from threading import Event, Lock
from typing import Literal


CancellationOrigin = Literal[
    "server_signal",
    "client_cancel",
    "connection_drop",
    "handler",
    "internal",
]


class GenerationCancelledError(RuntimeError):
    pass


class RequestCancellation:
    def __init__(self) -> None:
        self.thread_event = Event()
        self._state_lock = Lock()
        self._reason = "Generation was cancelled"
        self._origin: CancellationOrigin = "internal"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_event: asyncio.Event | None = None

    @property
    def reason(self) -> str:
        with self._state_lock:
            return self._reason

    @property
    def origin(self) -> CancellationOrigin:
        with self._state_lock:
            return self._origin

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._state_lock:
            self._loop = loop
            if self._async_event is None:
                self._async_event = asyncio.Event()
            async_event = self._async_event
            already_cancelled = self.thread_event.is_set()

        if already_cancelled:
            loop.call_soon_threadsafe(async_event.set)

    def cancel(
        self,
        reason: str = "Generation was cancelled",
        origin: CancellationOrigin = "internal",
    ) -> None:
        with self._state_lock:
            if not self.thread_event.is_set():
                self._reason = reason
                self._origin = origin
            loop = self._loop
            async_event = self._async_event

        self.thread_event.set()
        if loop is not None and async_event is not None:
            loop.call_soon_threadsafe(async_event.set)

    def is_cancelled(self) -> bool:
        return self.thread_event.is_set()

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()
        self.bind_loop(loop)
        with self._state_lock:
            async_event = self._async_event
        assert async_event is not None
        await async_event.wait()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise GenerationCancelledError(self.reason)


def raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelledError("Generation was cancelled")
