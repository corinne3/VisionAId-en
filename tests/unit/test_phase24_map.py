"""Phase 24 (B): reliable map — obstacles within reliable range, user trajectory free."""

from __future__ import annotations

import numpy as np
import pytest

from visionaid.mapping.costmap import FREE, OCC, OccupancyGrid
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit


def grid(**ov):
    return OccupancyGrid(load_config(overrides={f"costmap.{k}": v for k, v in ov.items()}).costmap)


def wall(x, y0=-1.0, y1=1.0, n=400):
    ys = np.linspace(y0, y1, n)
    return np.stack([np.full(n, x), ys, np.full(n, 0.8)], 1)


def test_far_points_do_not_create_obstacles():
    g = grid(occ_range_m=3.0)
    cam = np.array([0.0, 0.0, 1.4])
    for _ in range(10):
        g.update(np.concatenate([wall(2.0), wall(4.0)]), cam, 0.1)
    st = g.state()
    near = g.world_to_cell(np.array([[2.0, 0.0]]))[0]
    far = g.world_to_cell(np.array([[4.0, 0.0]]))[0]
    assert st[tuple(near)] == OCC  # at 2 m: obstacle marked
    assert st[tuple(far)] != OCC  # at 4 m: no obstacle (beyond reliable range)


def test_walked_cells_are_free_and_never_blocked():
    g = grid(occ_range_m=3.0)
    cam = np.array([0.0, 0.0, 1.4])
    # noisy obstacle right next to the trajectory
    obs = np.array([[0.25, 0.0, 0.8]] * 50)
    for _ in range(10):
        g.update(obs, cam, 0.1)
    c0 = g.cost()
    me = tuple(g.world_to_cell(np.array([[0.0, 0.0]]))[0])
    assert not np.isfinite(c0[me])  # within the 0.35 m margin: blocked
    g.carve(np.array([0.0, 0.0]), 0.20)
    g.update(obs, cam, 0.1)
    assert g.state()[me] == FREE
    assert np.isfinite(g.cost()[me])  # where we walked: never untraversable
    # the obstacle itself stays an obstacle
    ob = tuple(g.world_to_cell(np.array([[0.25, 0.0]]))[0])
    assert g.state()[ob] == OCC


def test_walked_mask_survives_grid_growth():
    g = grid()
    g.carve(np.array([0.0, 0.0]), 0.2)
    n0 = int(g.walked.sum())
    g.ensure(np.array([[20.0, 20.0]]))  # grid enlargement
    assert g.walked.shape == g.L.shape and int(g.walked.sum()) == n0
    me = tuple(g.world_to_cell(np.array([[0.0, 0.0]]))[0])
    assert g.walked[me]


def test_obstacle_appearing_on_walked_cells_is_seen():
    """Safety: a chair placed where the user walked must reappear on the map."""
    g = grid(occ_range_m=3.0)
    cam = np.array([0.0, 0.0, 1.4])
    g.carve(np.array([1.0, 0.0]), 0.20)  # walked through (1, 0)
    g.carve(np.array([0.0, 0.0]), 0.20)
    chair = np.array([[1.0, 0.0, 0.5]] * 50)
    for _ in range(10):
        g.update(chair, cam, 0.1)
    cell = tuple(g.world_to_cell(np.array([[1.0, 0.0]]))[0])
    assert g.walked[cell]
    assert g.state()[cell] == OCC
    assert not np.isfinite(g.cost()[cell])  # untraversable despite the walk
