"""D* Lite (Koenig & Likhachev 2002, optimised version) on an 8-connected grid, numba kernels.

Same cost model as A*: step a → b = length × (1 + (c[a] + c[b]) / 2), no corner cutting.
The search starts from the goal; the start (the user) can move and the costs change between two
calls to `plan()`, only the affected areas are recomputed.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

SQRT2 = math.sqrt(2.0)
DI = np.array([-1, 1, 0, 0, -1, -1, 1, 1], np.int64)
DJ = np.array([0, 0, -1, 1, -1, 1, -1, 1], np.int64)
LN = np.array([1.0, 1.0, 1.0, 1.0, SQRT2, SQRT2, SQRT2, SQRT2])


@njit(cache=True)
def _less(a1, a2, b1, b2):
    return a1 < b1 or (a1 == b1 and a2 < b2)


@njit(cache=True)
def _push(K1, K2, V, n, k1, k2, v):
    i = n
    K1[i] = k1
    K2[i] = k2
    V[i] = v
    while i > 0:
        p = (i - 1) >> 1
        if not _less(K1[i], K2[i], K1[p], K2[p]):
            break
        K1[p], K1[i] = K1[i], K1[p]
        K2[p], K2[i] = K2[i], K2[p]
        V[p], V[i] = V[i], V[p]
        i = p
    return n + 1


@njit(cache=True)
def _sift_down(K1, K2, V, n, i):
    while True:
        lft = 2 * i + 1
        r = lft + 1
        m = i
        if lft < n and _less(K1[lft], K2[lft], K1[m], K2[m]):
            m = lft
        if r < n and _less(K1[r], K2[r], K1[m], K2[m]):
            m = r
        if m == i:
            return
        K1[m], K1[i] = K1[i], K1[m]
        K2[m], K2[i] = K2[i], K2[m]
        V[m], V[i] = V[i], V[m]
        i = m


@njit(cache=True)
def _pop(K1, K2, V, n):
    k1, k2, v = K1[0], K2[0], V[0]
    n -= 1
    K1[0], K2[0], V[0] = K1[n], K2[n], V[n]
    _sift_down(K1, K2, V, n, 0)
    return k1, k2, v, n


@njit(cache=True)
def _valid(K1, K2, V, i, inq, qk1, qk2):
    v = V[i]
    return inq[v] and qk1[v] == K1[i] and qk2[v] == K2[i]


@njit(cache=True)
def _clean_top(K1, K2, V, n, inq, qk1, qk2):
    while n > 0 and not _valid(K1, K2, V, 0, inq, qk1, qk2):
        _k1, _k2, _v, n = _pop(K1, K2, V, n)
    return n


@njit(cache=True)
def _compact(K1, K2, V, n, inq, qk1, qk2):
    m = 0
    for i in range(n):
        if _valid(K1, K2, V, i, inq, qk1, qk2):
            K1[m], K2[m], V[m] = K1[i], K2[i], V[i]
            m += 1
    for i in range(m // 2 - 1, -1, -1):
        _sift_down(K1, K2, V, m, i)
    return m


@njit(cache=True)
def _h(a, b, w):
    di = abs(a // w - b // w)
    dj = abs(a % w - b % w)
    return (di + dj) + (SQRT2 - 2.0) * min(di, dj)


@njit(cache=True)
def _cost(cc, h, w, u, k):
    ui = u // w
    uj = u % w
    vi = ui + DI[k]
    vj = uj + DJ[k]
    if vi < 0 or vi >= h or vj < 0 or vj >= w:
        return np.inf, -1
    v = vi * w + vj
    if not (np.isfinite(cc[u]) and np.isfinite(cc[v])):
        return np.inf, v
    if k >= 4 and not (np.isfinite(cc[vi * w + uj]) and np.isfinite(cc[ui * w + vj])):
        return np.inf, v
    return LN[k] * (1.0 + 0.5 * (cc[u] + cc[v])), v


@njit(cache=True)
def _update_vertex(cc, h, w, g, rhs, inq, qk1, qk2, K1, K2, V, n, u, goal, start, km):
    if u != goal:
        best = np.inf
        for k in range(8):
            c, v = _cost(cc, h, w, u, k)
            if v >= 0 and c + g[v] < best:
                best = c + g[v]
        rhs[u] = best
    if g[u] != rhs[u]:
        m = min(g[u], rhs[u])
        k1 = m + _h(start, u, w) + km
        k2 = m
        inq[u] = True
        qk1[u] = k1
        qk2[u] = k2
        if n >= K1.shape[0]:
            n = _compact(K1, K2, V, n, inq, qk1, qk2)
        n = _push(K1, K2, V, n, k1, k2, u)
    else:
        inq[u] = False
    return n


@njit(cache=True)
def _compute(cc, h, w, g, rhs, inq, qk1, qk2, K1, K2, V, n, goal, start, km, max_iter):
    it = 0
    while True:
        n = _clean_top(K1, K2, V, n, inq, qk1, qk2)
        ms = min(g[start], rhs[start])
        sk1 = ms + km
        if n == 0:
            break
        if not (_less(K1[0], K2[0], sk1, ms) or rhs[start] != g[start]):
            break
        it += 1
        if it > max_iter:
            break
        kold1, kold2, u, n = _pop(K1, K2, V, n)
        inq[u] = False
        mu = min(g[u], rhs[u])
        knew1 = mu + _h(start, u, w) + km
        knew2 = mu
        if _less(kold1, kold2, knew1, knew2):
            inq[u] = True
            qk1[u] = knew1
            qk2[u] = knew2
            n = _push(K1, K2, V, n, knew1, knew2, u)
        elif g[u] > rhs[u]:
            g[u] = rhs[u]
            for k in range(8):
                _c, v = _cost(cc, h, w, u, k)
                if v >= 0:
                    n = _update_vertex(
                        cc, h, w, g, rhs, inq, qk1, qk2, K1, K2, V, n, v, goal, start, km
                    )
        else:
            g[u] = np.inf
            n = _update_vertex(cc, h, w, g, rhs, inq, qk1, qk2, K1, K2, V, n, u, goal, start, km)
            for k in range(8):
                _c, v = _cost(cc, h, w, u, k)
                if v >= 0:
                    n = _update_vertex(
                        cc, h, w, g, rhs, inq, qk1, qk2, K1, K2, V, n, v, goal, start, km
                    )
    return n, it


@njit(cache=True)
def _update_cells(cc, h, w, g, rhs, inq, qk1, qk2, K1, K2, V, n, cells, newc, goal, start, km):
    for idx in range(cells.shape[0]):
        cc[cells[idx]] = newc[idx]
    for idx in range(cells.shape[0]):
        u = cells[idx]
        n = _update_vertex(cc, h, w, g, rhs, inq, qk1, qk2, K1, K2, V, n, u, goal, start, km)
        for k in range(8):
            _c, v = _cost(cc, h, w, u, k)
            if v >= 0:
                n = _update_vertex(
                    cc, h, w, g, rhs, inq, qk1, qk2, K1, K2, V, n, v, goal, start, km
                )
    return n


@njit(cache=True)
def _extract(cc, h, w, g, start, goal, max_len):
    path = np.empty(max_len, np.int64)
    path[0] = start
    m = 1
    u = start
    total = 0.0
    while u != goal and m < max_len:
        best = np.inf
        bv = -1
        bc = 0.0
        for k in range(8):
            c, v = _cost(cc, h, w, u, k)
            if v >= 0 and c + g[v] < best:
                best = c + g[v]
                bv = v
                bc = c
        if bv < 0 or not np.isfinite(best):
            return path[:0], np.inf
        u = bv
        total += bc
        path[m] = u
        m += 1
    if u != goal:
        return path[:0], np.inf
    return path[:m], total


class DStarLite:
    """Incremental planner. `set_costs()` then `plan(start)` at each cycle."""

    def __init__(self, cell_cost: np.ndarray, goal: tuple[int, int]) -> None:
        self.h, self.w = cell_cost.shape
        N = self.h * self.w
        self.cc = np.ascontiguousarray(cell_cost, np.float64).ravel().copy()
        self.goal = goal[0] * self.w + goal[1]
        self.g = np.full(N, np.inf)
        self.rhs = np.full(N, np.inf)
        self.inq = np.zeros(N, np.bool_)
        self.qk1 = np.zeros(N)
        self.qk2 = np.zeros(N)
        cap = 16 * N + 64
        self.K1 = np.empty(cap)
        self.K2 = np.empty(cap)
        self.V = np.empty(cap, np.int64)
        self.n = 0
        self.km = 0.0
        self.last: int | None = None
        self.start: int | None = None
        self.rhs[self.goal] = 0.0
        self.last_iters = 0

    def _ensure_capacity(self) -> None:
        if self.n > self.K1.shape[0] // 2:
            self.n = _compact(self.K1, self.K2, self.V, self.n, self.inq, self.qk1, self.qk2)
        if self.n > self.K1.shape[0] // 2:
            cap = 2 * self.K1.shape[0]
            for name in ("K1", "K2", "V"):
                old = getattr(self, name)
                new = np.empty(cap, old.dtype)
                new[: self.n] = old[: self.n]
                setattr(self, name, new)

    def set_costs(self, cell_cost: np.ndarray) -> int:
        """Replaces the cost grid; only updates the modified cells. Returns their number."""
        new = np.ascontiguousarray(cell_cost, np.float64).ravel()
        diff = ~((new == self.cc) | (np.isinf(new) & np.isinf(self.cc)))
        cells = np.nonzero(diff)[0].astype(np.int64)
        if len(cells) == 0:
            return 0
        if self.start is None:
            self.cc[cells] = new[cells]
            return len(cells)
        self._ensure_capacity()
        self.n = _update_cells(
            self.cc,
            self.h,
            self.w,
            self.g,
            self.rhs,
            self.inq,
            self.qk1,
            self.qk2,
            self.K1,
            self.K2,
            self.V,
            self.n,
            cells,
            new[cells].copy(),
            self.goal,
            self.start,
            self.km,
        )
        return len(cells)

    def plan(
        self, start: tuple[int, int], max_iter: int = 10_000_000
    ) -> tuple[list[tuple[int, int]] | None, float]:
        s = start[0] * self.w + start[1]
        if self.start is None:
            self.start = s
            self.last = s
            self.n = 0
            self.inq[self.goal] = True
            self.qk1[self.goal] = _h(s, self.goal, self.w)
            self.qk2[self.goal] = 0.0
            self.n = _push(self.K1, self.K2, self.V, 0, self.qk1[self.goal], 0.0, self.goal)
        elif s != self.start:
            self.km += _h(self.last, s, self.w)
            self.last = s
            self.start = s
        self._ensure_capacity()
        self.n, self.last_iters = _compute(
            self.cc,
            self.h,
            self.w,
            self.g,
            self.rhs,
            self.inq,
            self.qk1,
            self.qk2,
            self.K1,
            self.K2,
            self.V,
            self.n,
            self.goal,
            self.start,
            self.km,
            max_iter,
        )
        if not np.isfinite(self.g[s]) and not np.isfinite(self.rhs[s]):
            return None, math.inf
        p, total = _extract(self.cc, self.h, self.w, self.g, s, self.goal, self.h * self.w + 1)
        if len(p) == 0:
            return None, math.inf
        return [(int(u // self.w), int(u % self.w)) for u in p], float(total)


def warmup() -> None:
    d = DStarLite(np.zeros((5, 5)), (4, 4))
    d.plan((0, 0))
    c = np.zeros((5, 5))
    c[2, 1:4] = np.inf
    d.set_costs(c)
    d.plan((0, 1))
