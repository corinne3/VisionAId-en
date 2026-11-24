"""Minimal blocks to test the bus (importable by the subprocesses)."""

from __future__ import annotations

import time

import numpy as np

from visionaid.bus.messages import Frame, GuidanceCue


class CounterSource:
    """Publishes numbered frames at a fixed rate."""

    def __init__(self, cfg: dict) -> None:
        self.i = 0
        self.size = tuple(cfg.get("test_size", (48, 64, 3)))

    def process(self, inputs):
        f = Frame(t=time.time(), frame_idx=self.i, image=np.full(self.size, self.i % 255, np.uint8))
        self.i += 1
        return {"frame": f}


class SlowEcho:
    """Consumes the frames slowly and returns a cue with the received index."""

    def __init__(self, cfg: dict) -> None:
        self.delay = float(cfg.get("test_delay", 0.05))

    def process(self, inputs):
        f = inputs.get("frame")
        if f is None:
            return None
        time.sleep(self.delay)
        return {"cue": GuidanceCue(t=f.t, frame_idx=f.frame_idx, state="TEST")}


class Collector:
    """Puits silencieux."""

    def __init__(self, cfg: dict) -> None:
        self.n = 0

    def process(self, inputs):
        self.n += 1
        return None

    def stats(self):
        return {"n": self.n}
