"""Phase 25 (C): path stability (commitment) and aiming at the object at the goal."""

from __future__ import annotations

import numpy as np
import pytest

from visionaid.bus.messages import Costmap, ObjectEstimate
from visionaid.mapping.costmap import FREE, OCC, OccupancyGrid
from visionaid.planning.planner import Planner
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit
RES = 0.05


def make_cm(extra_left: bool, block_left: bool = False) -> Costmap:
    """Free 6 × 4 m room, central 1 m obstacle ahead; a small obstacle (noise) on one side."""
    cfg = load_config()
    h, w = int(4 / RES), int(6 / RES)
    st = np.full((h, w), FREE, np.int8)
    origin = np.array([-1.0, -2.0])

    def box(x0, x1, y0, y1):
        j0, j1 = int((x0 - origin[0]) / RES), int((x1 - origin[0]) / RES)
        i0, i1 = int((y0 - origin[1]) / RES), int((y1 - origin[1]) / RES)
        st[i0:i1, j0:j1] = OCC

    box(1.8, 2.2, -0.5, 0.5)  # central obstacle, in front of the user (x axis)
    if extra_left:  # noise: lengthens the detour on the left a little (y > 0)
        box(1.5, 1.6, 0.7, 0.8)
    else:
        box(1.5, 1.6, -0.8, -0.7)
    if block_left:  # real obstacle: the left becomes impracticable
        box(1.9, 2.1, 0.5, 2.0)
    g = OccupancyGrid(cfg.costmap)
    return Costmap(0.0, 0, st, g.cost(st), origin, RES)


def side(path) -> int:
    mid = path.points[np.argmin(np.abs(path.points[:, 0] - 2.0))]
    return int(np.sign(mid[1]))


def obj():
    return ObjectEstimate(0.0, 0, np.array([4.0, 0.0, 0.8]), np.eye(3) * 0.01, "seen", 0.9)


def run_seq(commit: bool, n: int = 10):
    # isolated commitment mechanism: D* planner without the "within arm's reach" goal (phase 27)
    ov = {"planning.reach_m": 0.0}
    if not commit:
        ov["planning.commit_ratio"] = 0.0
    pl = Planner(load_config(overrides=ov))
    sides = []
    for k in range(n):
        p = pl.plan(make_cm(extra_left=bool(k % 2)), np.array([0.0, 0.0]), obj())
        assert p.valid
        sides.append(side(p))
    return sides, pl


def test_noise_flips_path_without_commitment():
    sides, _ = run_seq(commit=False)
    flips = sum(a != b for a, b in zip(sides, sides[1:], strict=False))
    assert flips >= 5  # without commitment: the path changes side almost at every computation


def test_commitment_keeps_side():
    sides, pl = run_seq(commit=True)
    assert len(set(sides)) == 1  # same side from start to end
    assert pl.n_kept >= 8


def test_commitment_released_when_path_blocked():
    pl = Planner(load_config())
    p0 = pl.plan(make_cm(extra_left=False), np.array([0.0, 0.0]), obj())
    s0 = side(p0)
    assert s0 == 1  # the left is the shortest at the start
    p1 = pl.plan(make_cm(extra_left=False, block_left=True), np.array([0.0, 0.0]), obj())
    assert p1.valid and side(p1) == -1  # the left is blocked: new path on the right


def test_aim_points_to_object_at_goal():
    pl = Planner(load_config())
    cm = make_cm(extra_left=False)
    o = obj()
    # user at the goal point (free cell closest to the object, ~ in front of them)
    p = pl.plan(cm, np.array([3.8, 0.0]), o)
    assert p.valid and p.length < 0.35
    np.testing.assert_allclose(p.lookahead, o.position[:2])
