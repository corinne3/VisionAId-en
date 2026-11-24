"""Generic bus node: loop, target rate, latency measurement, clean stop."""

from __future__ import annotations

import importlib
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from visionaid.bus.transport import Publisher, Subscriber, endpoint
from visionaid.utils.timing import LatencyStats


class Block(Protocol):
    """Business block: receives the latest message of each input, returns {topic: message}."""

    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None: ...


@dataclass
class NodeSpec:
    """Description of a graph node."""

    name: str
    factory: str  # "module:callable" returning a Block; receives (cfg_raw: dict)
    inputs: list[str] = field(default_factory=list)  # sujets lus
    outputs: list[str] = field(default_factory=list)  # published topics
    trigger: str | None = None  # triggering topic; None = fixed rate (rate_hz)
    rate_hz: float = 0.0  # max rate (0 = unlimited if trigger)
    heavy_inputs: list[str] = field(default_factory=list)  # inputs in "latest value" mode
    accumulate_inputs: list[str] = field(default_factory=list)  # inputs delivered as a list (IMU)


def load_factory(path: str):
    mod, _, attr = path.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _t_of(msg: Any) -> float | None:
    return getattr(msg, "t", None)


def run_node(
    spec: NodeSpec,
    ports: dict[str, int],
    cfg_raw: dict,
    stop_event,
    stats_queue=None,
    recv_timeout_ms: int = 50,
) -> None:
    """Loop of a node (run in a subprocess)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    block = load_factory(spec.factory)(cfg_raw)
    pub = Publisher(endpoint(ports[spec.name])) if spec.outputs else None
    subs: list[Subscriber] = []
    by_producer: dict[str, list[str]] = {}
    for topic in spec.inputs:
        by_producer.setdefault(ports["topic:" + topic], []).append(topic)
    for port, topics in by_producer.items():
        light = [t for t in topics if t not in spec.heavy_inputs]
        heavy = [t for t in topics if t in spec.heavy_inputs]
        if light:
            subs.append(Subscriber(endpoint(port), light, conflate=False))
        if heavy:
            subs.append(Subscriber(endpoint(port), heavy, conflate=True))
    stats = LatencyStats()
    latest: dict[str, Any] = {}
    acc: dict[str, list] = {}
    period = 1.0 / spec.rate_hz if spec.rate_hz > 0 else 0.0
    next_tick = time.perf_counter()
    n_out = 0
    try:
        while not stop_event.is_set():
            fresh_trigger = False
            trig_sent = None
            for s in subs:
                for topic, sent, msg in s.drain_all():
                    if topic in spec.accumulate_inputs:
                        acc.setdefault(topic, []).append(msg)
                    else:
                        latest[topic] = msg
                    stats.add(f"age_in:{topic}", time.time() - sent)
                    if topic == spec.trigger:
                        fresh_trigger = True
                        trig_sent = sent
            now = time.perf_counter()
            if spec.trigger is not None:
                ready = fresh_trigger and (period == 0 or now >= next_tick)
            else:
                ready = now >= next_tick
            if not ready:
                if subs:
                    for s in subs:
                        s.poller.poll(max(1, recv_timeout_ms // max(len(subs), 1)))
                else:
                    time.sleep(min(max(next_tick - now, 0.0), recv_timeout_ms / 1000))
                continue
            next_tick = max(next_tick + period, now) if period else now
            t0 = time.perf_counter()
            ins = dict(latest)
            for topic in spec.accumulate_inputs:  # everything that arrived since the last call
                ins[topic] = acc.pop(topic, [])
            out = block.process(ins)
            stats.add("process", time.perf_counter() - t0)
            if out and pub is not None:
                for topic, msg in out.items():
                    if msg is not None:
                        pub.send(topic, msg)
                        n_out += 1
            if trig_sent is not None:
                stats.add("e2e_from_trigger_send", time.time() - trig_sent)
    finally:
        close = getattr(block, "close", None)
        if callable(close):
            close()
        if stats_queue is not None:
            summ = stats.summary()
            summ["_n_out"] = {"n": n_out}
            extra = getattr(block, "stats", None)
            if callable(extra):
                summ["_block"] = extra()
            try:
                stats_queue.put((spec.name, summ), timeout=1.0)
            except Exception:
                pass
        if pub is not None:
            pub.close()
        for s in subs:
            s.close()
