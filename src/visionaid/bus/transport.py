"""ZeroMQ pub/sub transport: envelope = topic | send timestamp | msgpack message."""

from __future__ import annotations

import struct
import time
from typing import Any

import zmq

from visionaid.bus.messages import decode, encode

_SEP = b"|"


def endpoint(port: int, host: str = "127.0.0.1") -> str:
    return f"tcp://{host}:{port}"


class Publisher:
    """Publishes typed messages on a port (bind)."""

    def __init__(self, address: str, hwm: int = 10000, ctx: zmq.Context | None = None) -> None:
        self.ctx = ctx or zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, hwm)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(address)
        self.address = address

    def send(self, topic: str, msg: Any) -> None:
        env = topic.encode() + _SEP + struct.pack("<d", time.time()) + encode(msg)
        self.sock.send(env, copy=False)

    def close(self) -> None:
        self.sock.close(0)


class Subscriber:
    """Subscriber to one or more topics.

    `conflate=True`: only the latest value is kept (heavy streams). ZMQ_CONFLATE
    only accepts one topic per socket; so one socket per topic is created in this mode.
    """

    def __init__(
        self,
        address: str,
        topics: list[str],
        conflate: bool = False,
        hwm: int = 10000,
        ctx: zmq.Context | None = None,
    ) -> None:
        self.ctx = ctx or zmq.Context.instance()
        self.conflate = conflate
        self.socks: list[zmq.Socket] = []
        groups = [[t] for t in topics] if conflate else [topics]
        for grp in groups:
            s = self.ctx.socket(zmq.SUB)
            s.setsockopt(zmq.LINGER, 0)
            if conflate:
                s.setsockopt(zmq.CONFLATE, 1)
            else:
                s.setsockopt(zmq.RCVHWM, hwm)
            for t in grp:
                s.setsockopt(zmq.SUBSCRIBE, t.encode() + _SEP)
            s.connect(address)
            self.socks.append(s)
        self.poller = zmq.Poller()
        for s in self.socks:
            self.poller.register(s, zmq.POLLIN)

    @staticmethod
    def _unpack(raw: bytes) -> tuple[str, float, Any]:
        i = raw.index(_SEP)
        topic = raw[:i].decode()
        (sent,) = struct.unpack("<d", raw[i + 1 : i + 9])
        return topic, sent, decode(raw[i + 9 :])

    def recv(self, timeout_ms: int = 100) -> tuple[str, float, Any] | None:
        """Receives a message (blocking until timeout). Returns (topic, t_send, msg)."""
        events = dict(self.poller.poll(timeout_ms))
        for s in self.socks:
            if s in events:
                return self._unpack(s.recv())
        return None

    def drain(self, timeout_ms: int = 0) -> dict[str, tuple[float, Any]]:
        """Empties the queues without blocking (after an optional poll): latest message per topic."""
        out: dict[str, tuple[float, Any]] = {}
        if timeout_ms > 0 and not self.poller.poll(timeout_ms):
            return out
        for s in self.socks:
            while True:
                try:
                    raw = s.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                topic, sent, msg = self._unpack(raw)
                out[topic] = (sent, msg)
        return out

    def drain_all(self) -> list[tuple[str, float, Any]]:
        """Empties the queues without blocking: all received messages, in order of arrival."""
        out: list[tuple[str, float, Any]] = []
        for s in self.socks:
            while True:
                try:
                    raw = s.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                out.append(self._unpack(raw))
        return out

    def close(self) -> None:
        for s in self.socks:
            s.close(0)
