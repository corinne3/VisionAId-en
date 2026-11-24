"""Starts and stops a graph of nodes (one subprocess per node)."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

from visionaid.bus.nodes import NodeSpec, run_node
from visionaid.utils.logging import get_logger

log = get_logger("runner")


def assign_ports(specs: list[NodeSpec], base_port: int) -> dict[str, int]:
    """One port per producer node; `topic:<topic>` → port of the producer."""
    ports: dict[str, int] = {}
    for i, s in enumerate(specs):
        ports[s.name] = base_port + i
        for t in s.outputs:
            if "topic:" + t in ports:
                raise ValueError(f"Topic {t} published by several nodes")
            ports["topic:" + t] = base_port + i
    for s in specs:
        for t in s.inputs:
            if "topic:" + t not in ports:
                raise ValueError(f"Topic {t} read by {s.name} but never published")
    return ports


def specs_from_config(graph: list[dict]) -> list[NodeSpec]:
    return [NodeSpec(**g) for g in graph]


class GraphRunner:
    """Starts the nodes in separate processes (spawn) and stops them cleanly."""

    def __init__(
        self,
        specs: list[NodeSpec],
        cfg_raw: dict,
        base_port: int = 5550,
        stop_timeout_s: float = 2.0,
    ) -> None:
        self.specs = specs
        self.cfg_raw = cfg_raw
        self.ports = assign_ports(specs, base_port)
        self.stop_timeout_s = stop_timeout_s
        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        self.stats_queue = self.ctx.Queue()
        self.procs: list[mp.Process] = []
        self.stats: dict[str, dict] = {}

    def start(self) -> None:
        # Consumers first (avoids losing the first messages), producers afterwards.
        for spec in reversed(self.specs):
            p = self.ctx.Process(
                target=run_node,
                args=(spec, self.ports, self.cfg_raw, self.stop_event, self.stats_queue),
                name=f"node-{spec.name}",
                daemon=True,
            )
            p.start()
            self.procs.append(p)
        log.info("Graph started: %s", [s.name for s in self.specs])

    def alive(self) -> list[str]:
        return [p.name for p in self.procs if p.is_alive()]

    def stop(self) -> float:
        """Stops all nodes; returns the stop duration (s)."""
        t0 = time.perf_counter()
        self.stop_event.set()
        deadline = t0 + self.stop_timeout_s
        while time.perf_counter() < deadline and len(self.stats) < len(self.procs):
            try:
                name, summ = self.stats_queue.get(timeout=0.05)
                self.stats[name] = summ
            except queue.Empty:
                if not self.alive():
                    break
        for p in self.procs:
            p.join(timeout=max(0.0, deadline - time.perf_counter()))
        for p in self.procs:
            if p.is_alive():
                log.warning("Node %s does not stop: terminate", p.name)
                p.terminate()
                p.join(0.5)
            if p.is_alive():
                p.kill()
                p.join(0.5)
        while True:
            try:
                name, summ = self.stats_queue.get_nowait()
                self.stats[name] = summ
            except (queue.Empty, OSError, ValueError):
                break
        self.stats_queue.close()
        self.stats_queue.join_thread()
        return time.perf_counter() - t0

    def __enter__(self) -> GraphRunner:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class SequentialRunner:
    """Deterministic single-process mode: same blocks, called in sequence at each tick.

    Nodes are called in the order of `specs`. A node without a trigger (source) is
    called at each tick; a triggered node only if its topic was produced during
    the tick. Rate-limited nodes (`rate_hz`) are subsampled according to the time of the
    messages (`t`), not the wall clock, hence the determinism.
    """

    def __init__(self, specs: list[NodeSpec], cfg_raw: dict, blocks: dict | None = None) -> None:
        from visionaid.bus.nodes import load_factory

        assign_ports(specs, 0)  # graph validation
        self.specs = specs
        self.blocks = blocks or {s.name: load_factory(s.factory)(cfg_raw) for s in specs}
        self.latest: dict = {}
        self._last_run_t: dict[str, float] = {}

    def tick(self) -> dict:
        """Runs one tick; returns the messages produced (None if the source is exhausted)."""
        produced: dict = {}
        for s in self.specs:
            if s.trigger is not None and s.trigger not in produced:
                continue
            if s.trigger is not None and s.rate_hz > 0:
                t = getattr(produced[s.trigger], "t", None)
                last = self._last_run_t.get(s.name)
                if t is not None and last is not None and t - last < 1.0 / s.rate_hz - 1e-9:
                    continue
                if t is not None:
                    self._last_run_t[s.name] = t
            out = self.blocks[s.name].process(dict(self.latest))
            if s.trigger is None and not out:
                return {}
            for k, v in (out or {}).items():
                if v is not None:
                    self.latest[k] = v
                    produced[k] = v
        return produced

    def run(self, max_ticks: int | None = None) -> int:
        n = 0
        while max_ticks is None or n < max_ticks:
            if not self.tick():
                break
            n += 1
        for b in self.blocks.values():
            close = getattr(b, "close", None)
            if callable(close):
                close()
        return n
