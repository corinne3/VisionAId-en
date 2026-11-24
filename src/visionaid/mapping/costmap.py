"""Top-down 2D occupancy grid (log-odds), inflation and proximity cost."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from visionaid.bus.messages import Costmap

UNKNOWN, FREE, OCC = -1, 0, 1


class OccupancyGrid:
    """Log-odds occupancy in the aligned world frame (z up, floor at z=0), dynamic size."""

    def __init__(self, cfg_costmap) -> None:
        c = cfg_costmap
        self.res = float(c.resolution_m)
        n = int(round(float(c.init_size_m) / self.res))
        self.L = np.zeros((n, n), np.float32)
        self.origin = np.array([-n * self.res / 2, -n * self.res / 2])
        self.band = tuple(float(v) for v in c.obstacle_band_m)
        self.floor_band = float(c.floor_band_m)
        self.l_occ, self.l_free = float(c.l_occ), float(c.l_free)
        self.l_min, self.l_max = float(c.l_min), float(c.l_max)
        self.p_occ, self.p_free = float(c.occ_threshold), float(c.free_threshold)
        self.half_life = float(c.decay_half_life_s)
        self.radius = float(c.robot_radius_m)
        self.cost_decay = float(c.cost_decay_m)
        self.cost_scale = float(c.cost_scale)
        self.max_range = float(c.max_range_m)
        self.slope = float(getattr(c, "obstacle_height_slope", 0.0))
        self.raycast = bool(c.raycast)
        # obstacles marked only within reliable range (beyond that, monocular depth makes
        # the distant floor "rise" and closes passages that do not exist)
        self.occ_range = float(getattr(c, "occ_range_m", None) or self.max_range)
        self.walked = np.zeros(self.L.shape, bool)  # cells the user has walked through
        self.walk_clearance = float(getattr(c, "carve_radius_m", 0.2) or 0.2)
        self.walked_bias = float(getattr(c, "walked_free_bias", 0.0) or 0.0)
        self.n_updates = 0

    # ------------------------------------------------------------ indexation
    @property
    def shape(self) -> tuple[int, int]:
        return self.L.shape

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        """(N,2) x,y → (N,2) i (row, along y), j (column, along x)."""
        ij = np.floor((np.asarray(xy)[..., ::-1] - self.origin[::-1]) / self.res).astype(int)
        return ij

    def cell_to_world(self, ij: np.ndarray) -> np.ndarray:
        ij = np.asarray(ij, float)
        return np.stack(
            [
                self.origin[0] + (ij[..., 1] + 0.5) * self.res,
                self.origin[1] + (ij[..., 0] + 0.5) * self.res,
            ],
            -1,
        )

    def ensure(self, xy: np.ndarray, margin: float = 1.0) -> None:
        """Grows the grid to contain the xy points (+ margin)."""
        if len(xy) == 0:
            return
        lo = xy.min(0) - margin
        hi = xy.max(0) + margin
        h, w = self.L.shape
        x0, y0 = self.origin
        x1, y1 = x0 + w * self.res, y0 + h * self.res
        if lo[0] >= x0 and lo[1] >= y0 and hi[0] <= x1 and hi[1] <= y1:
            return
        pad = 2.0
        nx0 = min(x0, lo[0] - pad) if lo[0] < x0 else x0
        ny0 = min(y0, lo[1] - pad) if lo[1] < y0 else y0
        nx1 = max(x1, hi[0] + pad) if hi[0] > x1 else x1
        ny1 = max(y1, hi[1] + pad) if hi[1] > y1 else y1
        dj = int(round((x0 - nx0) / self.res))
        di = int(round((y0 - ny0) / self.res))
        nw = dj + w + int(round((nx1 - x1) / self.res))
        nh = di + h + int(round((ny1 - y1) / self.res))
        L = np.zeros((nh, nw), np.float32)
        L[di : di + h, dj : dj + w] = self.L
        self.L = L
        W = np.zeros((nh, nw), bool)
        W[di : di + h, dj : dj + w] = self.walked
        self.walked = W
        self.origin = np.array([x0 - dj * self.res, y0 - di * self.res])

    # ---------------------------------------------------------- update
    def update(self, pts_world: np.ndarray, cam_pos: np.ndarray, dt: float = 1.0 / 30.0) -> None:
        """Integrates a point cloud (aligned world frame) seen from `cam_pos`.

        Forgetting: log-odds decay towards 0 (unknown) with a half-life `decay_half_life_s`.
        """
        self.n_updates += 1
        if self.half_life > 0 and dt > 0:
            self.L *= np.float32(0.5 ** (dt / self.half_life))
        if len(pts_world) == 0:
            return
        horiz = np.linalg.norm(pts_world[:, :2] - cam_pos[:2], axis=1)
        pts = pts_world[horiz < self.max_range]
        z = pts[:, 2]
        r = horiz[horiz < self.max_range]
        # tolerance growing with distance (depth noise, estimated floor tilt)
        lo = self.band[0] + self.slope * r
        obst = pts[(z >= lo) & (z <= self.band[1]) & (r < self.occ_range)]
        floor = pts[np.abs(z) < self.floor_band + self.slope * r]
        self.ensure(np.concatenate([pts[:, :2], cam_pos[None, :2]]))
        h, w = self.L.shape
        free_cells = self._cells(floor[:, :2])
        occ_cells = self._cells(obst[:, :2])
        if self.raycast:
            ends = np.concatenate([obst[:, :2], floor[:, :2]])
            ray_cells = self._ray_cells(cam_pos[:2], ends)
            free_cells = (
                np.unique(np.concatenate([free_cells, ray_cells]), axis=0)
                if len(ray_cells)
                else free_cells
            )
        # a cell hit by an obstacle is not freed during this update
        if len(occ_cells) and len(free_cells):
            occ_lin = occ_cells[:, 0] * w + occ_cells[:, 1]
            free_lin = free_cells[:, 0] * w + free_cells[:, 1]
            free_cells = free_cells[~np.isin(free_lin, occ_lin)]
        if len(free_cells):
            self.L[free_cells[:, 0], free_cells[:, 1]] += self.l_free
        if len(occ_cells):
            self.L[occ_cells[:, 0], occ_cells[:, 1]] += self.l_occ
        if self.walked_bias > 0 and self.walked.any():
            # walked cells: slight "free" bias at each update. Occasional noise
            # is absorbed; a real obstacle (seen at each update, +l_occ) quickly wins
            self.L[self.walked] -= self.walked_bias
        np.clip(self.L, self.l_min, self.l_max, out=self.L)

    def carve(self, xy: np.ndarray, radius: float) -> None:
        """Marks as free, at the time of passing, the cells within `radius` of the user.

        Freeing is not permanent: an obstacle appearing later at the same place
        (moved chair, person) is marked again by the following measurements.
        """
        xy = np.asarray(xy, float)[:2]
        self.ensure(xy[None, :])
        r = int(np.ceil(radius / self.res))
        ci, cj = self.world_to_cell(xy[None, :])[0]
        h, w = self.L.shape
        ii, jj = np.mgrid[max(ci - r, 0) : min(ci + r + 1, h), max(cj - r, 0) : min(cj + r + 1, w)]
        m = (ii - ci) ** 2 + (jj - cj) ** 2 <= (radius / self.res) ** 2
        self.walked[ii[m], jj[m]] = True
        self.L[ii[m], jj[m]] = np.minimum(self.L[ii[m], jj[m]], self.l_min)

    def _cells(self, xy: np.ndarray) -> np.ndarray:
        if len(xy) == 0:
            return np.zeros((0, 2), int)
        ij = self.world_to_cell(xy)
        h, w = self.L.shape
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < h) & (ij[:, 1] >= 0) & (ij[:, 1] < w)
        return np.unique(ij[ok], axis=0)

    def _ray_cells(self, start: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Cells crossed between the camera and each point (end excluded)."""
        if len(ends) == 0:
            return np.zeros((0, 2), int)
        ends_c = np.unique(self.world_to_cell(ends), axis=0)
        ends = self.cell_to_world(ends_c)
        d = np.linalg.norm(ends - start, axis=1)
        n = np.maximum((d / (0.5 * self.res)).astype(int), 1)
        nmax = int(n.max())
        s = np.arange(nmax)[None, :] / n[:, None]
        valid = s < 1.0 - 1e-9
        pts = start + (ends - start)[:, None, :] * s[..., None]
        cells = self.world_to_cell(pts[valid] if valid.any() else np.zeros((0, 2)))
        # remove the final cell of each ray
        return self._cells(self.cell_to_world(cells)) if len(cells) else np.zeros((0, 2), int)

    # ---------------------------------------------------------------- lecture
    def prob(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.L))

    def state(self) -> np.ndarray:
        p = self.prob()
        s = np.full(p.shape, UNKNOWN, np.int8)
        s[p > self.p_occ] = OCC
        s[p < self.p_free] = FREE
        return s

    def cost(self, state: np.ndarray | None = None) -> np.ndarray:
        """Cost: inf within the inflated radius, exponential decay beyond, 0 far from obstacles."""
        s = self.state() if state is None else state
        occ = s == OCC
        if not occ.any():
            return np.zeros(s.shape, np.float32)
        dist = distance_transform_edt(~occ) * self.res
        c = self.cost_scale * np.exp(
            -np.maximum(dist - self.radius, 0.0) / max(self.cost_decay, 1e-6)
        )
        c = c.astype(np.float32)
        # untraversable inflation, except on already walked cells that are not
        # obstacles and stay more than `walk_clearance` from an obstacle (high but finite cost):
        # the user is never "blocked" where they have just walked
        blocked = dist < self.radius
        if self.walked.shape == blocked.shape:
            ok = self.walked & ~occ & (dist >= self.walk_clearance)
            c[blocked & ok] = 2.0 * self.cost_scale
            blocked &= ~ok
        c[blocked] = np.inf
        return c

    def to_msg(self, t: float, frame_idx: int) -> Costmap:
        s = self.state()
        return Costmap(t, frame_idx, s, self.cost(s), self.origin.copy(), self.res)
