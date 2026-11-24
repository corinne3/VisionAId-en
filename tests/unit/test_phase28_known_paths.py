"""Phase 28: paths through known areas only, sound direction without jumps.

Live trial gourde_son_live3 (22/11, log_tick 2688 of the Rerun view): the path cut through
a never-seen area (black in the world view) although a known passage (white) existed; and the
sound direction jumped from one ear to the other when the path changed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from visionaid.bus.messages import Costmap, ObjectEstimate, Path, Pose
from visionaid.guidance.cues import GuidanceEngine
from visionaid.mapping.costmap import FREE, OCC, UNKNOWN, OccupancyGrid
from visionaid.planning.planner import Planner
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit
RES = 0.05
ORIGIN = np.array([-1.0, -3.0])


def ij(x, y):
    return int((y - ORIGIN[1]) / RES), int((x - ORIGIN[0]) / RES)


def fill(st, x0, x1, y0, y1, v):
    i0, j0 = ij(x0, y0)
    i1, j1 = ij(x1, y1)
    st[i0:i1, j0:j1] = v


def costmap(st, cfg):
    return Costmap(0.0, 0, st, OccupancyGrid(cfg.costmap).cost(st), ORIGIN.copy(), RES)


def unknown_len(st, pts, skip=0.3):
    seg = np.diff(pts, axis=0)
    cum = np.r_[0.0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
    s = np.arange(skip, cum[-1], 0.02)
    x, y = np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])
    cells = [ij(a, b) for a, b in zip(x, y, strict=True)]
    return sum(st[c] == UNKNOWN for c in cells) * 0.02


def detour_scene():
    """User at (0, 0), object at (4, 1). Wall at x≈2 (y from −2.2 to 0); above the wall,
    a never-seen 60 cm band (the shortcut); below (y ≈ −2.5), a known passage."""
    st = np.full((int(6 / RES), int(6 / RES)), UNKNOWN, np.int8)
    fill(st, -0.5, 4.8, -2.8, 2.8, FREE)  # everything has been seen…
    fill(st, 1.7, 2.3, 0.0, 2.8, UNKNOWN)  # … except this band, above the wall
    fill(st, 1.8, 2.2, -2.2, 0.0, OCC)  # wall
    obj = ObjectEstimate(0.0, 0, np.array([4.0, 1.0, 0.8]), np.eye(3) * 0.01, "seen", 0.9)
    return st, obj


def test_path_never_goes_through_unknown():
    cfg = load_config()
    st, obj = detour_scene()
    p = Planner(cfg).plan(costmap(st, cfg), np.array([0.0, 0.0]), obj)
    assert p.valid
    assert unknown_len(st, p.points) == 0.0
    assert p.points[:, 1].min() < -1.4  # goes through the known passage, at the bottom
    # before: the shortcut through the unknown was chosen
    old = load_config(overrides={"planning.allow_unknown": True})
    q = Planner(old).plan(costmap(st, old), np.array([0.0, 0.0]), obj)
    assert q.valid and unknown_len(st, q.points) > 0.5


def test_no_known_route_goes_to_frontier():
    """No known path to the object: path up to the edge of the known area."""
    cfg = load_config()
    st = np.full((int(6 / RES), int(6 / RES)), UNKNOWN, np.int8)
    fill(st, -0.5, 1.5, -0.6, 0.6, FREE)  # only 1.5 m ahead of the user is known
    obj = ObjectEstimate(0.0, 0, np.array([4.0, 0.0, 0.8]), np.eye(3) * 0.01, "seen", 0.9)
    pl = Planner(cfg)
    p = pl.plan(costmap(st, cfg), np.array([0.0, 0.0]), obj)
    assert p.valid and pl.n_frontier == 1
    assert unknown_len(st, p.points, skip=0.0) == 0.0
    assert 1.2 < p.points[-1, 0] <= 1.5  # stops at the edge of the known area, towards the object
    assert np.linalg.norm(p.goal - p.points[-1]) < 1e-6
    off = load_config(overrides={"planning.frontier_fallback": False})
    assert not Planner(off).plan(costmap(st, off), np.array([0.0, 0.0]), obj).valid


def _pose(yaw: float) -> Pose:
    """Head at (0, 0, 1.4) looking in the direction `yaw` (rad, 0 = +x)."""
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.stack([[s, -c, 0.0], [0.0, 0.0, -1.0], [c, s, 0.0]], 1)  # x_cam, y_cam, z_cam
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.0, 0.0, 1.4]
    return Pose(0.0, 0, T, "ok")


def _inputs(la_xy):
    st = np.full((100, 100), FREE, np.int8)
    cm = Costmap(0.0, 0, st, np.zeros(st.shape, np.float32), np.array([-2.5, -2.5]), 0.05)
    obj = ObjectEstimate(0.0, 0, np.array([3.0, 3.0, 0.8]), np.eye(3), "seen", 1.0)
    la = np.asarray(la_xy, float)
    path = Path(0.0, 0, np.array([[0.0, 0.0], la * 3]), la, np.array([3.0, 3.0]), True, 4.0)
    return obj, path, cm


def test_direction_glides_when_path_flips():
    """The path swings by 120°: the sound turns at most 90°/s, without a jump."""
    cfg = load_config()
    e = GuidanceEngine(cfg)
    obj, p_left, cm = _inputs([math.cos(1.0), math.sin(1.0)])  # path at +57°
    _, p_right, _ = _inputs([math.cos(-1.1), math.sin(-1.1)])  # then at −63°
    pose = _pose(0.0)
    azs = []
    for k in range(60):
        t = k * 0.05
        azs.append(e.update(t, k, pose, obj, p_left if k < 10 else p_right, cm).guide_azimuth)
    steps = np.degrees(np.abs(np.diff(azs)))
    assert steps.max() <= cfg.guidance.direction_max_rate_dps * 0.05 + 1e-6
    assert azs[9] == pytest.approx(1.0, abs=1e-3)
    assert azs[-1] == pytest.approx(-1.1, abs=0.05)  # reaches the new direction


def test_body_rotation_not_delayed():
    """Smoothing is in the world frame: turning the body changes the sound immediately."""
    e = GuidanceEngine(load_config())
    obj, path, cm = _inputs([1.0, 0.0])  # path straight ahead (+x)
    a0 = e.update(0.0, 0, _pose(0.0), obj, path, cm).guide_azimuth
    a1 = e.update(0.05, 1, _pose(math.radians(60)), obj, path, cm).guide_azimuth
    assert a0 == pytest.approx(0.0, abs=1e-6)
    assert a1 == pytest.approx(-math.radians(60), abs=1e-3)  # path now on the right


def test_smoothing_never_points_into_obstacle():
    """Path replanned to avoid an obstacle that appeared ahead: no gliding, the new
    direction is taken immediately (otherwise the sound would still lead onto the obstacle)."""
    e = GuidanceEngine(load_config())
    obj, p_ahead, cm = _inputs([1.0, 0.0])  # path straight ahead (+x)
    _, p_side, _ = _inputs([math.cos(1.2), math.sin(1.2)])  # detour on the left
    e.update(0.0, 0, _pose(0.0), obj, p_ahead, cm)
    cm.state[50 - 4 : 50 + 5, 50 + 10 : 50 + 13] = OCC  # obstacle 0.5 m ahead
    a = e.update(0.05, 1, _pose(0.0), obj, p_side, cm).guide_azimuth
    assert a == pytest.approx(1.2, abs=1e-3) and e.n_dir_snaps == 1
