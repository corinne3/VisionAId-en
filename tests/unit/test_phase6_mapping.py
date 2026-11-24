import numpy as np
import pytest

from visionaid.mapping.costmap import FREE, OCC, UNKNOWN, OccupancyGrid
from visionaid.mapping.floor import FloorTracker, fit_plane_ransac
from visionaid.mapping.geometry import inv_T, make_T, rot_between, transform
from visionaid.mapping.object_anchor import ObjectAnchor
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit


def cfg_(**ov):
    return load_config(overrides=ov)


def floor_points(rng, n=4000, size=4.0, noise=0.01):
    return np.stack(
        [rng.uniform(-size, size, n), rng.uniform(-size, size, n), rng.normal(0, noise, n)], 1
    )


def test_plane_ransac_noisy():
    rng = np.random.default_rng(0)
    pts = floor_points(rng)
    R = rot_between(
        np.array([0, 0, 1.0]), np.array([0.1, -0.2, 1.0]) / np.linalg.norm([0.1, -0.2, 1.0])
    )
    pts = pts @ R.T + np.array([0, 0, -1.4])
    pts = np.concatenate([pts, rng.uniform(-3, 3, (800, 3))])  # 17 % d'outliers
    n, d, _ = fit_plane_ransac(pts, 200, 0.03, rng, np.array([0, 0, 1.0]))
    true_n = R @ np.array([0, 0, 1.0])
    ang = np.degrees(np.arccos(abs(np.dot(n, true_n))))
    assert ang < 2.0
    assert abs(abs(d) - abs(np.dot(true_n, [0, 0, -1.4]))) < 0.03


def test_floor_tracker_camera_height():
    """Camera at 1.4 m, tilted 30° downwards, above a flat floor (VO frame = camera)."""
    cfg = cfg_()
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    h, w = 240, 320
    p = np.radians(30)
    v, u = np.mgrid[0:h, 0:w]
    y = (v - 120) / 300.0
    den = np.cos(p) * y + np.sin(p)  # up_cam = (0, -cos p, -sin p); up·X = -1.4 on the floor
    depth = np.where(den > 1e-3, 1.4 / np.maximum(den, 1e-3), 50.0)
    depth += np.random.default_rng(0).normal(0, 0.005, depth.shape)
    ft = FloorTracker(cfg.floor)
    assert ft.update(depth.astype(np.float32), K, np.eye(4))
    assert ft.height(np.zeros(3)) == pytest.approx(1.4, abs=0.03)
    up_true = np.array([0, -np.cos(p), -np.sin(p)])
    assert np.degrees(np.arccos(np.dot(ft.n, up_true))) < 2.0
    T_w = ft.T_world_vo()
    cam_world = T_w @ np.array([0, 0, 0, 1.0])
    assert cam_world[2] == pytest.approx(1.4, abs=0.03)
    assert np.allclose(cam_world[:2], 0, atol=1e-6)  # origin on the floor under the 1st camera
    fwd = T_w[:3, :3] @ np.array([0, 0, 1.0])
    assert fwd[0] > 0.8 and abs(fwd[1]) < 1e-6  # cap initial = +x


def scan_box(grid, box_min, box_max, cam, rng, n=3000):
    """Points: free floor around + faces of a box."""
    floor = np.stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), np.zeros(n)], 1)
    inside = (
        (floor[:, 0] > box_min[0])
        & (floor[:, 0] < box_max[0])
        & (floor[:, 1] > box_min[1])
        & (floor[:, 1] < box_max[1])
    )
    floor = floor[~inside]
    box = np.stack(
        [
            rng.uniform(box_min[0], box_max[0], n),
            rng.uniform(box_min[1], box_max[1], n),
            rng.uniform(0.0, 0.5, n),
        ],
        1,
    )
    ceil = np.stack([rng.uniform(-3, 3, 500), rng.uniform(-3, 3, 500), np.full(500, 2.3)], 1)
    grid.update(np.concatenate([floor, box, ceil]), cam)


def test_box_occupancy_iou():
    cfg = cfg_(**{"costmap.decay_half_life_s": 0.0, "costmap.raycast": False})
    g = OccupancyGrid(cfg.costmap)
    rng = np.random.default_rng(1)
    bmin, bmax = np.array([1.0, -0.25]), np.array([1.5, 0.25])
    for _ in range(3):
        scan_box(g, bmin, bmax, np.array([0, 0, 1.4]), rng)
    s = g.state()
    occ = s == OCC
    ij = np.indices(s.shape).reshape(2, -1).T
    xy = g.cell_to_world(ij).reshape(*s.shape, 2)
    gt = (
        (xy[..., 0] > bmin[0])
        & (xy[..., 0] < bmax[0])
        & (xy[..., 1] > bmin[1])
        & (xy[..., 1] < bmax[1])
    )
    iou = (occ & gt).sum() / (occ | gt).sum()
    assert iou > 0.8
    assert (s == FREE).sum() > 1000


def test_ceiling_ignored():
    cfg = cfg_()
    g = OccupancyGrid(cfg.costmap)
    pts = np.stack([np.linspace(-1, 1, 500), np.zeros(500), np.full(500, 2.3)], 1)
    for _ in range(5):
        g.update(pts, np.array([0, 0, 1.4]))
    assert (g.state() == OCC).sum() == 0


def test_inflation_radius():
    cfg = cfg_()
    g = OccupancyGrid(cfg.costmap)
    g.L[:] = -3.0
    c0 = np.array(g.shape) // 2
    g.L[c0[0], c0[1]] = 3.0
    cost = g.cost()
    row = cost[c0[0], c0[1] :]
    first_free = int(np.argmax(np.isfinite(row)))
    radius_cells = cfg.costmap.robot_radius_m / cfg.costmap.resolution_m
    assert abs(first_free - radius_cells) <= 1
    assert row[first_free] > row[first_free + 5] > 0  # decreasing cost


def test_decay_removes_obstacle():
    cfg = cfg_()
    g = OccupancyGrid(cfg.costmap)
    rng = np.random.default_rng(2)
    for _ in range(3):
        scan_box(g, np.array([1.0, -0.25]), np.array([1.5, 0.25]), np.array([0, 0, 1.4]), rng)
    ij = g.world_to_cell(np.array([[1.25, 0.0]]))[0]
    assert g.state()[ij[0], ij[1]] == OCC
    n = 0
    while g.state()[ij[0], ij[1]] == OCC and n < 200:
        g.update(np.zeros((0, 3)), np.array([0, 0, 1.4]), dt=1.0)  # plus rien vu : oubli (1 Hz)
        n += 1
    assert n < 100
    assert g.state()[ij[0], ij[1]] in (UNKNOWN, FREE)


def test_raycast_clears_removed_obstacle():
    cfg = cfg_(**{"costmap.decay_half_life_s": 0.0})
    g = OccupancyGrid(cfg.costmap)
    rng = np.random.default_rng(3)
    for _ in range(3):
        scan_box(g, np.array([1.0, -0.25]), np.array([1.5, 0.25]), np.array([0, 0, 1.4]), rng)
    ij = g.world_to_cell(np.array([[1.25, 0.0]]))[0]
    wall = np.stack(
        [np.full(2000, 3.0), rng.uniform(-0.5, 0.5, 2000), rng.uniform(0.1, 1.5, 2000)], 1
    )
    n = 0
    while g.state()[ij[0], ij[1]] == OCC and n < 50:
        g.update(wall, np.array([0, 0, 1.4]))
        n += 1
    assert n < 15


def test_grid_grows():
    cfg = cfg_(**{"costmap.init_size_m": 2.0})
    g = OccupancyGrid(cfg.costmap)
    g.update(np.array([[8.0, -6.0, 0.5]]), np.array([7.0, -5.0, 1.4]))
    ij = g.world_to_cell(np.array([[8.0, -6.0]]))[0]
    assert 0 <= ij[0] < g.shape[0] and 0 <= ij[1] < g.shape[1]
    assert g.L[ij[0], ij[1]] > 0


def test_anchor_noisy_poses():
    cfg = cfg_()
    rng = np.random.default_rng(4)
    a = ObjectAnchor(cfg.anchor)
    obj = np.array([2.0, 1.0, 0.8])
    for k in range(5):
        T = make_T(
            rot_between(
                np.array([0, 0, 1.0]),
                np.array([0.2 * k, 0, 1.0]) / np.linalg.norm([0.2 * k, 0, 1.0]),
            ),
            [0.1 * k, 0, 1.4],
        )
        T_noisy = T.copy()
        T_noisy[:3, 3] += rng.normal(0, 0.02, 3)
        pc = transform(inv_T(T), obj[None])[0]
        a.update_world(transform(T_noisy, pc[None])[0], k * 0.1, 0.9, float(np.linalg.norm(pc)))
    est = a.estimate(0.4, 4)
    assert np.linalg.norm(est.position - obj) < 0.05
    assert est.state == "seen"


def test_anchor_memory():
    cfg = cfg_()
    a = ObjectAnchor(cfg.anchor)
    assert a.estimate(0.0, 0).state == "unknown"
    a.update_world(np.array([1.0, 2.0, 0.7]), 0.0, 1.0, 2.0)
    tr0 = np.trace(a.P)
    e = a.estimate(3.0, 90)
    assert e.state == "remembered"
    assert np.allclose(e.position, [1.0, 2.0, 0.7])
    assert np.trace(e.cov) > tr0  # incertitude croissante
    assert a.estimate(100.0, 3000).state == "stale"


def test_anchor_rejects_outlier_then_relocates():
    cfg = cfg_()
    a = ObjectAnchor(cfg.anchor)
    for k in range(10):
        a.update_world(np.array([1.0, 0.0, 0.7]), k * 0.03, 1.0, 1.0)
    assert not a.update_world(np.array([3.0, 0.0, 0.7]), 0.4, 1.0, 1.0)
    assert np.allclose(a.x, [1.0, 0.0, 0.7], atol=0.01)
    for k in range(5):
        a.update_world(np.array([3.0, 0.0, 0.7]), 0.5 + k * 0.03, 1.0, 1.0)
    assert np.linalg.norm(a.x - [3.0, 0, 0.7]) < 0.1
