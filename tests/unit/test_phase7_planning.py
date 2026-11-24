import time

import numpy as np
import pytest

from visionaid.bus.messages import Costmap, ObjectEstimate
from visionaid.mapping.costmap import FREE, OCC, OccupancyGrid
from visionaid.planning.astar import astar, dijkstra
from visionaid.planning.astar import warmup as wa
from visionaid.planning.dstar_lite import DStarLite
from visionaid.planning.dstar_lite import warmup as wd
from visionaid.planning.grid import path_cost
from visionaid.planning.lookahead import lookahead_point, polyline_length, smooth_path
from visionaid.planning.planner import Planner
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit
wa()
wd()


def rand_grid(rng, n=60, blocks=15, size=(2, 10)):
    c = np.zeros((n, n))
    for _ in range(blocks):
        i, j = rng.integers(0, n, 2)
        c[i : i + rng.integers(*size), j : j + rng.integers(*size)] = np.inf
    fin = np.isfinite(c)
    c[fin] += rng.uniform(0, 1.0, fin.sum())
    c[:2, :2] = 0
    c[-2:, -2:] = 0
    return c


def test_astar_equals_dijkstra():
    rng = np.random.default_rng(0)
    for _ in range(30):
        c = rand_grid(rng)
        s, g = (0, 0), (59, 59)
        p, L = astar(c, s, g)
        p2, L2 = dijkstra(c, s, g)
        if p is None:
            assert p2 is None
            continue
        assert L == pytest.approx(L2, abs=1e-6)
        assert path_cost(c, p) == pytest.approx(L, abs=1e-6)


def test_unreachable_returns_none():
    c = np.zeros((30, 30))
    c[:, 15] = np.inf
    t0 = time.perf_counter()
    p, L = astar(c, (5, 5), (5, 25))
    assert p is None and np.isinf(L)
    d = DStarLite(c, (5, 25))
    p, L = d.plan((5, 5))
    assert p is None and np.isinf(L)
    assert time.perf_counter() - t0 < 1.0


def test_no_corner_cutting():
    c = np.zeros((3, 3))
    c[0, 1] = np.inf
    c[1, 0] = np.inf
    p, _ = astar(c, (0, 0), (1, 1))
    assert p is None


def test_dstar_matches_astar_after_changes():
    rng = np.random.default_rng(1)
    n_checked = 0
    for _ in range(50):
        c = rand_grid(rng)
        goal = (59, 59)
        d = DStarLite(c, goal)
        start = (0, 0)
        p, L = d.plan(start)
        for _ in range(3):
            c2 = c.copy()
            i, j = rng.integers(0, 60, 2)
            if rng.random() < 0.5:
                c2[i : i + 6, j : j + 6] = np.inf  # insertion
            else:
                c2[i : i + 8, j : j + 8] = rng.uniform(0, 0.5)  # suppression
            c2[:2, :2] = 0
            c2[-2:, -2:] = 0
            if p is not None and len(p) > 4:
                start = p[2]
                c2[start] = min(c2[start], 0.0) if np.isfinite(c2[start]) else 0.0
            d.set_costs(c2)
            p, L = d.plan(start)
            pa, La = astar(c2, start, goal)
            if pa is None:
                assert p is None
            else:
                assert L == pytest.approx(La, abs=1e-6)
                n_checked += 1
            c = c2
    assert n_checked > 50


def test_performance_200():
    rng = np.random.default_rng(2)
    ta, td = [], []
    for _ in range(10):
        c = rand_grid(rng, 200, 50, (3, 20))
        t = time.perf_counter()
        astar(c, (0, 0), (199, 199))
        ta.append(time.perf_counter() - t)
        d = DStarLite(c, (199, 199))
        p, _ = d.plan((0, 0))
        start = (0, 0)
        for _ in range(5):
            c2 = c.copy()
            i, j = rng.integers(0, 200, 2)
            c2[i : i + 8, j : j + 8] = np.inf
            c2[:2, :2] = 0
            c2[-2:, -2:] = 0
            if p is not None and len(p) > 3:
                start = p[2]
            t = time.perf_counter()
            d.set_costs(c2)
            p, _ = d.plan(start)
            td.append(time.perf_counter() - t)
            c = c2
    assert np.mean(ta) < 0.050
    assert np.mean(td) < 0.010


def make_cm(cfg, state):
    g = OccupancyGrid(cfg.costmap)
    g.L = np.where(state == OCC, 3.0, np.where(state == FREE, -3.0, 0.0)).astype(np.float32)
    g.origin = np.array([0.0, 0.0])
    s = g.state()
    return Costmap(0.0, 0, s, g.cost(s), g.origin.copy(), g.res), g


def test_path_avoids_inflated_cells():
    cfg = load_config()
    st = np.full((80, 80), FREE, np.int8)
    st[20:60, 38:42] = OCC  # wall
    cm, g = make_cm(cfg, st)
    pl = Planner(cfg)
    obj = ObjectEstimate(0, 0, np.array([3.5, 2.0, 0.8]), np.eye(3), "seen", 1.0)
    path = pl.plan(cm, np.array([0.5, 2.0]), obj)
    assert path.valid
    for p in np.linspace(path.points[:-1], path.points[1:], 20).reshape(-1, 2):
        i, j = int(p[1] / g.res), int(p[0] / g.res)
        assert np.isfinite(cm.cost[i, j])


def test_goal_is_free_cell_near_object_on_furniture():
    cfg = load_config()
    st = np.full((80, 80), FREE, np.int8)
    st[30:50, 50:70] = OCC  # table
    cm, g = make_cm(cfg, st)
    obj = ObjectEstimate(0, 0, np.array([3.0, 2.0, 0.8]), np.eye(3), "seen", 1.0)  # on the table
    path = Planner(cfg).plan(cm, np.array([0.5, 2.0]), obj)
    assert path.valid
    gi, gj = int(path.goal[1] / g.res), int(path.goal[0] / g.res)
    assert st[gi, gj] != OCC
    # phase 27: goal "within arm's reach" (≤ 0.70 m from the object), margin reduced to 0.20 m near
    # the object (instead of 0.35 m): at the edge of the table, on the user's side
    # object 0.50 m inside the table: no accessible cell is within arm's reach
    # (0.55 m), fallback to the closest traversable cell, at the edge of the margin (~0.73 m)
    assert np.linalg.norm(path.goal - obj.position[:2]) <= 0.80
    assert 2.5 - path.goal[0] >= cfg.planning.approach_clearance_m - 0.05  # ≥ 20 cm from the table
    assert 2.0 < path.goal[0] < 2.5 + 1e-6


def test_lookahead():
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    la = lookahead_point(pts, 1.0)
    assert np.allclose(la, [1.0, 0.0])
    la = lookahead_point(pts, 2.5)
    assert np.allclose(la, [2.0, 0.5])
    assert np.allclose(lookahead_point(pts, 10.0), [2.0, 2.0])  # shorter path → target
    assert polyline_length(pts) == pytest.approx(4.0)
    # on the path: zero distance to the segment
    la = lookahead_point(pts, 3.3)
    assert abs(la[0] - 2.0) < 1e-9 and 0 <= la[1] <= 2


def test_smoothing_shortens_and_stays_valid():
    c = np.zeros((40, 40))
    c[10:30, 20] = np.inf
    p, _ = astar(c, (20, 0), (20, 39))
    sp = smooth_path(c, p)
    assert len(sp) < len(p)
    assert sp[0] == p[0] and sp[-1] == p[-1]
    from visionaid.planning.lookahead import line_of_sight

    assert all(line_of_sight(c, a, b) for a, b in zip(sp[:-1], sp[1:], strict=True))
