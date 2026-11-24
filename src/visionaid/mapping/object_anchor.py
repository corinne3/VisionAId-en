"""3D anchoring of the target: measurement (mask + depth), Kalman filter, memory."""

from __future__ import annotations

import cv2
import numpy as np

from visionaid.bus.messages import ObjectEstimate
from visionaid.mapping.geometry import backproject, transform


def measure_object(
    mask: np.ndarray | None,
    box: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    shrink: float,
    min_points: int,
) -> np.ndarray | None:
    """Camera position of the object = median of the 3D points under the eroded mask (or shrunk box)."""
    h, w = depth.shape
    if mask is not None and mask.sum() >= min_points:
        m = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if m.sum() < min_points:
            m = mask
    else:
        x0, y0, x1, y1 = box
        cx, cy, bw, bh = (
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            (x1 - x0) * (1 - shrink),
            (y1 - y0) * (1 - shrink),
        )
        m = np.zeros((h, w), bool)
        m[
            int(max(cy - bh / 2, 0)) : int(min(cy + bh / 2, h)),
            int(max(cx - bw / 2, 0)) : int(min(cx + bw / 2, w)),
        ] = True
    pts, _ = backproject(depth, K, 1, m, 0.1, 20.0)
    if len(pts) < min_points:
        return None
    # depth: median; x,y: mask centre at this depth
    return np.median(pts, axis=0)


class ObjectAnchor:
    """Constant-position Kalman filter + unknown/seen/remembered/stale states."""

    def __init__(self, cfg_anchor) -> None:
        c = cfg_anchor
        self.q = float(c.process_noise)
        self.r = float(c.meas_noise)
        self.shrink = float(c.shrink_box)
        self.seen_timeout = float(c.seen_timeout_s)
        self.stale_after = float(c.stale_after_s)
        self.min_points = int(c.min_points)
        self.x: np.ndarray | None = None
        self.P = np.eye(3)
        self.t_last_seen = -np.inf
        self.t_last = None
        self.n_meas = 0
        self.n_rejected = 0
        self._reject_streak = 0

    def predict(self, t: float) -> None:
        if self.x is None:
            self.t_last = t
            return
        dt = max(0.0, t - (self.t_last if self.t_last is not None else t))
        self.P = self.P + np.eye(3) * self.q * dt
        self.t_last = t

    def update_world(
        self, z: np.ndarray, t: float, confidence: float = 1.0, range_m: float = 2.0
    ) -> bool:
        """Update with a measurement in the world frame; noise growing with distance."""
        self.predict(t)
        R = np.eye(3) * self.r * max(range_m / 2.0, 0.5) ** 2 / max(confidence, 0.1)
        if self.x is None:
            self.x, self.P = z.copy(), R.copy()
        else:
            S = self.P + R
            innov = z - self.x
            m2 = float(innov @ np.linalg.solve(S, innov))
            if m2 > 16.0 and np.linalg.norm(innov) > 0.5:  # aberrante (~4σ)
                self._reject_streak += 1
                self.n_rejected += 1
                if self._reject_streak < 5:
                    return False
                self.x, self.P = z.copy(), R.copy()  # the object has moved: reset
            else:
                Kg = self.P @ np.linalg.inv(S)
                self.x = self.x + Kg @ innov
                self.P = (np.eye(3) - Kg) @ self.P
        self._reject_streak = 0
        self.t_last_seen = t
        self.n_meas += 1
        return True

    def update(
        self,
        t: float,
        T_world_cam: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        mask,
        box,
        confidence: float = 1.0,
    ) -> bool:
        pc = measure_object(mask, box, depth, K, self.shrink, self.min_points)
        if pc is None:
            self.predict(t)
            return False
        return self.update_world(transform(T_world_cam, pc[None])[0], t, confidence, float(pc[2]))

    def state_at(self, t: float) -> str:
        if self.x is None:
            return "unknown"
        age = t - self.t_last_seen
        if age <= self.seen_timeout:
            return "seen"
        if age <= self.stale_after:
            return "remembered"
        return "stale"

    def estimate(self, t: float, frame_idx: int) -> ObjectEstimate:
        self.predict(t)
        st = self.state_at(t)
        pos = self.x.copy() if self.x is not None else np.full(3, np.nan)
        conf = 0.0 if self.x is None else float(1.0 / (1.0 + np.sqrt(np.trace(self.P))))
        return ObjectEstimate(t, frame_idx, pos, self.P.copy(), st, conf)
