"""Camera intrinsics: default K from a field of view, or OpenCV calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Intrinsics:
    """Pinhole model without distortion."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def scaled(self, width: int, height: int) -> Intrinsics:
        """Intrinsics for a resized image."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)

    def rotated90(self, clockwise: bool = True) -> Intrinsics:
        """Intrinsics after rotating the image by 90°."""
        if clockwise:
            return Intrinsics(
                self.fy, self.fx, self.height - self.cy, self.cx, self.height, self.width
            )
        return Intrinsics(self.fy, self.fx, self.cy, self.width - self.cx, self.height, self.width)


def from_hfov(
    width: int, height: int, hfov_deg: float, landscape_reference: bool = True
) -> Intrinsics:
    """K from a horizontal field of view.

    If `landscape_reference`, the field is related to the longest side (sensor orientation),
    which stays correct for a video shot in portrait.
    """
    ref = max(width, height) if landscape_reference else width
    f = (ref / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return Intrinsics(f, f, width / 2.0, height / 2.0, width, height)


def load_calibration(path: str | Path) -> Intrinsics:
    """Loads an OpenCV calibration (npz with K, width, height; or yaml with camera_matrix)."""
    p = Path(path)
    if p.suffix == ".npz":
        d = np.load(p)
        K = np.asarray(d["K"], dtype=np.float64)
        w, h = int(d["width"]), int(d["height"])
    else:
        d = yaml.safe_load(p.read_text())
        K = np.asarray(d.get("K", d.get("camera_matrix")), dtype=np.float64).reshape(3, 3)
        w, h = int(d["width"]), int(d["height"])
    return Intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2], w, h)


def make_intrinsics(cfg_intr, width: int, height: int) -> Intrinsics:
    """Intrinsics for an image (width, height) according to the config."""
    if getattr(cfg_intr, "calibration_file", None):
        return load_calibration(cfg_intr.calibration_file).scaled(width, height)
    return from_hfov(width, height, float(cfg_intr.hfov_deg))
