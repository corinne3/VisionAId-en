"""Time and latency measurement tools."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


def now() -> float:
    """Monotonic clock (s)."""
    return time.perf_counter()


@dataclass
class LatencyStats:
    """Accumulates durations per stage and provides p50/p95."""

    samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, name: str, seconds: float) -> None:
        self.samples[name].append(float(seconds))

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for k, v in self.samples.items():
            if not v:
                continue
            a = np.asarray(v)
            out[k] = {
                "n": int(a.size),
                "p50_ms": float(np.percentile(a, 50) * 1e3),
                "p95_ms": float(np.percentile(a, 95) * 1e3),
                "mean_ms": float(a.mean() * 1e3),
            }
        return out


class Timer:
    """Context manager that adds the duration to a LatencyStats."""

    def __init__(self, stats: LatencyStats | None, name: str) -> None:
        self.stats = stats
        self.name = name
        self.t0 = 0.0
        self.dt = 0.0

    def __enter__(self) -> Timer:
        self.t0 = now()
        return self

    def __exit__(self, *exc: object) -> None:
        self.dt = now() - self.t0
        if self.stats is not None:
            self.stats.add(self.name, self.dt)
