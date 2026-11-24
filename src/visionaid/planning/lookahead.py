"""Path smoothing (line-of-sight shortcuts) and look-ahead point."""

from __future__ import annotations

import numpy as np


def line_of_sight(
    cell_cost: np.ndarray, a: tuple[int, int], b: tuple[int, int], max_cost: float = np.inf
) -> bool:
    """True if the segment a→b only crosses traversable cells (cost ≤ max_cost)."""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    ii = np.round(np.linspace(a[0], b[0], n)).astype(int)
    jj = np.round(np.linspace(a[1], b[1], n)).astype(int)
    c = cell_cost[ii, jj]
    return bool(np.all(np.isfinite(c)) and np.all(c <= max_cost))


def smooth_path(cell_cost: np.ndarray, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedy shortcuts: links each point to the farthest visible one without increasing the max cost."""
    if path is None or len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        seg_max = 0.0
        j_best = i + 1
        for j in range(i + 1, len(path)):
            seg_max = max(seg_max, float(cell_cost[path[j]]))
            if line_of_sight(cell_cost, path[i], path[j], max_cost=seg_max + 1e-9):
                j_best = j
        out.append(path[j_best])
        i = j_best
    return out


def polyline_length(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def lookahead_point(pts: np.ndarray, distance: float) -> np.ndarray:
    """Point at `distance` (curvilinear abscissa) along the polyline, or its end."""
    if len(pts) == 0:
        return np.full(2, np.nan)
    if len(pts) == 1:
        return pts[0].copy()
    seg = np.diff(pts, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    if distance >= cum[-1]:
        return pts[-1].copy()
    k = int(np.searchsorted(cum, distance, side="right") - 1)
    r = (distance - cum[k]) / max(L[k], 1e-12)
    return pts[k] + r * seg[k]
