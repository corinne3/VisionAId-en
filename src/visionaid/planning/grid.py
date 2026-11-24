"""Movement cost model on the grid, shared by A*, D* Lite and Dijkstra."""

from __future__ import annotations

import math

import numpy as np

from visionaid.mapping.costmap import UNKNOWN

SQRT2 = math.sqrt(2.0)
# (di, dj, longueur)
MOVES = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, SQRT2),
    (-1, 1, SQRT2),
    (1, -1, SQRT2),
    (1, 1, SQRT2),
]


def traversal_cost(
    cost: np.ndarray, state: np.ndarray | None, unknown_cost: float, allow_unknown: bool
) -> np.ndarray:
    """Cost per cell (≥ 0): proximity + unknown penalty; inf = forbidden."""
    c = np.asarray(cost, np.float64).copy()
    if state is not None:
        unk = state == UNKNOWN
        if allow_unknown:
            c[unk & np.isfinite(c)] += unknown_cost
        else:
            c[unk] = np.inf
    return c


def edge_cost(cell_cost: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    """Cost of a step a → b (neighbours): length × (1 + mean cost of the two cells).

    Diagonal moves do not cut occupied corners.
    """
    ca, cb = cell_cost[a], cell_cost[b]
    if not (math.isfinite(ca) and math.isfinite(cb)):
        return math.inf
    di, dj = b[0] - a[0], b[1] - a[1]
    if di and dj:
        if not (
            math.isfinite(cell_cost[a[0] + di, a[1]]) and math.isfinite(cell_cost[a[0], a[1] + dj])
        ):
            return math.inf
        return SQRT2 * (1.0 + 0.5 * (ca + cb))
    return 1.0 + 0.5 * (ca + cb)


def octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    di, dj = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (di + dj) + (SQRT2 - 2.0) * min(di, dj)


def path_cost(cell_cost: np.ndarray, path: list[tuple[int, int]]) -> float:
    return float(sum(edge_cost(cell_cost, path[k], path[k + 1]) for k in range(len(path) - 1)))
