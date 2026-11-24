"""8-connected A* on a cost grid (numba kernel), and reference Dijkstra.

Cost of a step a → b = length × (1 + (c[a] + c[b]) / 2); no diagonal corner cutting.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

SQRT2 = math.sqrt(2.0)


@njit(cache=True)
def _heap_push(keys, vals, n, k, v):
    i = n
    keys[i] = k
    vals[i] = v
    while i > 0:
        p = (i - 1) >> 1
        if keys[p] <= keys[i]:
            break
        keys[p], keys[i] = keys[i], keys[p]
        vals[p], vals[i] = vals[i], vals[p]
        i = p
    return n + 1


@njit(cache=True)
def _heap_pop(keys, vals, n):
    k0 = keys[0]
    v0 = vals[0]
    n -= 1
    keys[0] = keys[n]
    vals[0] = vals[n]
    i = 0
    while True:
        lft = 2 * i + 1
        r = lft + 1
        m = i
        if lft < n and keys[lft] < keys[m]:
            m = lft
        if r < n and keys[r] < keys[m]:
            m = r
        if m == i:
            break
        keys[m], keys[i] = keys[i], keys[m]
        vals[m], vals[i] = vals[i], vals[m]
        i = m
    return k0, v0, n


@njit(cache=True)
def _astar_kernel(cc, h, w, s, goal, use_h):
    N = h * w
    g = np.full(N, np.inf)
    parent = np.full(N, -1, np.int64)
    closed = np.zeros(N, np.bool_)
    cap = 8 * N + 8
    keys = np.empty(cap)
    vals = np.empty(cap, np.int64)
    gi = goal // w
    gj = goal % w
    k2 = SQRT2 - 2.0
    g[s] = 0.0
    si = s // w
    sj = s % w
    di0 = abs(si - gi)
    dj0 = abs(sj - gj)
    h0 = (di0 + dj0) + k2 * min(di0, dj0) if use_h else 0.0
    n = _heap_push(keys, vals, 0, h0, s)
    DI = np.array([-1, 1, 0, 0, -1, -1, 1, 1])
    DJ = np.array([0, 0, -1, 1, -1, 1, -1, 1])
    LN = np.array([1.0, 1.0, 1.0, 1.0, SQRT2, SQRT2, SQRT2, SQRT2])
    while n > 0:
        f, u, n = _heap_pop(keys, vals, n)
        if closed[u]:
            continue
        if u == goal:
            break
        closed[u] = True
        ui = u // w
        uj = u % w
        cu = cc[u]
        for k in range(8):
            vi = ui + DI[k]
            vj = uj + DJ[k]
            if vi < 0 or vi >= h or vj < 0 or vj >= w:
                continue
            v = vi * w + vj
            if closed[v] or not np.isfinite(cc[v]):
                continue
            if k >= 4:
                if not (np.isfinite(cc[vi * w + uj]) and np.isfinite(cc[ui * w + vj])):
                    continue
            ng = g[u] + LN[k] * (1.0 + 0.5 * (cu + cc[v]))
            if ng < g[v]:
                g[v] = ng
                parent[v] = u
                hv = 0.0
                if use_h:
                    a = abs(vi - gi)
                    b = abs(vj - gj)
                    hv = (a + b) + k2 * min(a, b)
                n = _heap_push(keys, vals, n, ng + hv, v)
    return g, parent


def astar(
    cell_cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], heuristic: bool = True
) -> tuple[list[tuple[int, int]] | None, float]:
    """Shortest path. Returns (path [(i,j)…], cost) or (None, inf) if unreachable."""
    h, w = cell_cost.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and 0 <= goal[0] < h and 0 <= goal[1] < w):
        return None, math.inf
    cc = np.ascontiguousarray(cell_cost, np.float64).ravel()
    s = start[0] * w + start[1]
    gl = goal[0] * w + goal[1]
    if not (np.isfinite(cc[s]) and np.isfinite(cc[gl])):
        return None, math.inf
    g, parent = _astar_kernel(cc, h, w, s, gl, heuristic)
    if not np.isfinite(g[gl]):
        return None, math.inf
    path = []
    u = gl
    while u != -1:
        path.append((int(u // w), int(u % w)))
        u = int(parent[u])
    path.reverse()
    return path, float(g[gl])


def dijkstra(cell_cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    """Brute-force reference: same kernel without heuristic."""
    return astar(cell_cost, start, goal, heuristic=False)


def warmup() -> None:
    """Compiles the kernel (numba disk cache)."""
    astar(np.zeros((4, 4)), (0, 0), (3, 3))
