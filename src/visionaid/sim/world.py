"""Ground-truth 2D world (top-down view) and simulated sensor."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class SimWorld:
    """Ground-truth occupancy grid + object + start."""

    occ: np.ndarray  # (H,W) bool, True = obstacle
    origin: np.ndarray  # (2,) (x,y) corner of cell (0,0)
    res: float
    obj: np.ndarray  # (3,) position of the object
    start: np.ndarray  # (3,) x, y, cap (rad)
    name: str = ""
    known_object: bool = False  # the object is already in memory at the start
    events: list = field(default_factory=list)  # (t, kind, data)
    obj_support: np.ndarray | None = None  # box (x0,y0,x1,y1) of the furniture carrying the object
    _dist: np.ndarray | None = None

    def cell(self, xy: np.ndarray) -> tuple[int, int]:
        return int(math.floor((xy[1] - self.origin[1]) / self.res)), int(
            math.floor((xy[0] - self.origin[0]) / self.res)
        )

    def inside(self, ij: tuple[int, int]) -> bool:
        return 0 <= ij[0] < self.occ.shape[0] and 0 <= ij[1] < self.occ.shape[1]

    def is_occ(self, xy: np.ndarray) -> bool:
        ij = self.cell(xy)
        return (not self.inside(ij)) or bool(self.occ[ij])

    @property
    def dist(self) -> np.ndarray:
        """Distance (m) from each cell to the closest obstacle."""
        if self._dist is None:
            self._dist = distance_transform_edt(~self.occ) * self.res
        return self._dist

    def clearance(self, xy: np.ndarray) -> float:
        ij = self.cell(xy)
        if not self.inside(ij):
            return 0.0
        return float(self.dist[ij])

    def add_box(self, x0: float, y0: float, x1: float, y1: float) -> None:
        i0, j0 = self.cell(np.array([x0, y0]))
        i1, j1 = self.cell(np.array([x1, y1]))
        self.occ[max(i0, 0) : i1 + 1, max(j0, 0) : j1 + 1] = True
        self._dist = None

    def raycast(self, xy: np.ndarray, angle: float, max_range: float) -> float:
        """Distance to the first obstacle along a ray (max_range if none)."""
        step = self.res * 0.5
        n = int(max_range / step)
        d = np.arange(1, n + 1) * step
        px = xy[0] + d * math.cos(angle)
        py = xy[1] + d * math.sin(angle)
        ii = np.floor((py - self.origin[1]) / self.res).astype(int)
        jj = np.floor((px - self.origin[0]) / self.res).astype(int)
        out = (ii < 0) | (ii >= self.occ.shape[0]) | (jj < 0) | (jj >= self.occ.shape[1])
        hit = out.copy()
        ok = ~out
        hit[ok] = self.occ[ii[ok], jj[ok]]
        k = np.argmax(hit) if hit.any() else -1
        return float(d[k]) if k >= 0 else max_range

    def visible(
        self, xy: np.ndarray, target: np.ndarray, heading: float, fov_deg: float, max_range: float
    ) -> bool:
        """Is the object in the field of view, in range and in line of sight (the carrying furniture does not hide it)?"""
        v = target[:2] - xy
        d = float(np.linalg.norm(v))
        if d > max_range or d < 1e-6:
            return d < 1e-6
        ang = math.atan2(v[1], v[0])
        if (
            abs(math.atan2(math.sin(ang - heading), math.cos(ang - heading)))
            > math.radians(fov_deg) / 2
        ):
            return False
        free_d = self.raycast(xy, ang, d)
        # the object is on a piece of furniture: the ray may hit that furniture just before the object
        return free_d >= d - 0.6 if self.obj_support is not None else free_d >= d - self.res


class SimSensor:
    """Simulated depth sensor: horizontal cone, range, noise → point cloud (world)."""

    def __init__(self, cfg_sim, rng: np.random.Generator, n_rays: int = 71) -> None:
        self.fov = math.radians(float(cfg_sim.fov_deg))
        self.range = float(cfg_sim.range_m)
        self.noise = float(cfg_sim.sensor_noise_m)
        self.rng = rng
        self.n_rays = n_rays

    def scan(self, world: SimWorld, x: float, y: float, heading: float) -> np.ndarray:
        pts = []
        xy = np.array([x, y])
        for a in heading + np.linspace(-self.fov / 2, self.fov / 2, self.n_rays):
            d = world.raycast(xy, a, self.range)
            c, s = math.cos(a), math.sin(a)
            # visible free floor: from 0.8 m (the floor very close is out of view) up to the obstacle
            for r in np.arange(0.8, d - 0.1, 0.15):
                pts.append((x + r * c, y + r * s, self.rng.normal(0, self.noise)))
            if d < self.range:
                for z in (0.3, 0.7, 1.2):  # vertical face of the obstacle
                    rr = d + abs(self.rng.normal(0, self.noise))
                    pts.append(
                        (
                            x + rr * c + self.rng.normal(0, self.noise),
                            y + rr * s + self.rng.normal(0, self.noise),
                            z,
                        )
                    )
        return np.asarray(pts, float).reshape(-1, 3)
