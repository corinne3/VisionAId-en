import numpy as np
import pytest

from visionaid.bus.messages import Frame
from visionaid.mapping.geometry import inv_T, make_T, rodrigues, rot_angle_deg, transform
from visionaid.pose.source import ArkitPoseSource, PoseSource, VisualOdometry
from visionaid.pose.vo import Observation, RGBDOdometry
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit

W, H = 480, 640
K = np.array([[450.0, 0, W / 2], [0, 450.0, H / 2], [0, 0, 1]])


def world(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.stack([rng.uniform(-6, 6, n), rng.uniform(-2.5, 2.5, n), rng.uniform(3, 10, n)], 1)
    desc = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    return pts, desc


def trajectory(n=100, step=0.04):
    """Cercle (rayon 0,5 m) + avance, lacet oscillant ±15°."""
    Ts = []
    for k in range(n):
        a = 2 * np.pi * k / n
        t = np.array([0.5 * np.sin(a), 0.0, 0.5 * (1 - np.cos(a)) + step * k * 0.3])
        R = rodrigues(np.array([0.0, np.radians(15) * np.sin(a), 0.0]))
        Ts.append(make_T(R, t))
    return Ts


def observe(pts, desc, T_wc, rng, noise=0.5, outliers=0.0):
    pc = transform(inv_T(T_wc), pts)
    ok = pc[:, 2] > 0.2
    uv = pc[ok, :2] / pc[ok, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv, z, d = uv[inside], pc[ok][inside, 2], desc[ok][inside]
    uv = uv + rng.normal(0, noise, uv.shape)
    if outliers > 0:
        m = rng.random(len(uv)) < outliers
        uv[m] = np.stack([rng.uniform(0, W, m.sum()), rng.uniform(0, H, m.sum())], 1)
    return Observation(uv=uv, desc=d, depth=z)


def run_vo(Ts, outliers=0.0, seed=0):
    cfg = load_config()
    vo = RGBDOdometry(cfg.vo)
    pts, desc = world(seed=seed)
    rng = np.random.default_rng(seed + 1)
    poses = [
        vo.step(observe(pts, desc, T, rng, outliers=outliers), K, k / 30, k)
        for k, T in enumerate(Ts)
    ]
    return vo, poses


def _rel(Ta, Tb):
    return inv_T(Ta) @ Tb


def check_pairs(Ts, poses):
    t_err, r_err = [], []
    T0inv = inv_T(Ts[0])
    gt = [T0inv @ T for T in Ts]  # VO world frame = first camera
    for k in range(len(Ts) - 1):
        a, b = _rel(gt[k], gt[k + 1]), _rel(poses[k].T_world_cam, poses[k + 1].T_world_cam)
        d = np.linalg.norm(a[:3, 3])
        t_err.append(np.linalg.norm(a[:3, 3] - b[:3, 3]) / d)
        r_err.append(rot_angle_deg(a[:3, :3].T @ b[:3, :3]))
    return np.array(t_err), np.array(r_err), gt


def check_kf_pairs(Ts, poses):
    """Relative error between consecutive keyframes (baseline ≥ kf_trans_m), see DECISIONS.md."""
    kfi = [k for k, p in enumerate(poses) if p.is_keyframe]
    te, re = [], []
    for a, b in zip(kfi[:-1], kfi[1:], strict=True):
        g = inv_T(Ts[a]) @ Ts[b]
        e = inv_T(poses[a].T_world_cam) @ poses[b].T_world_cam
        te.append(np.linalg.norm(g[:3, 3] - e[:3, 3]) / np.linalg.norm(g[:3, 3]))
        re.append(rot_angle_deg(g[:3, :3].T @ e[:3, :3]))
    return np.array(te), np.array(re)


def test_pairwise_accuracy():
    Ts = trajectory()
    vo, poses = run_vo(Ts)
    assert all(p.quality in ("ok", "init") for p in poses)
    te, re = check_kf_pairs(Ts, poses)
    assert len(te) >= 5
    assert np.median(te) < 0.03
    assert re.max() < 1.0
    # frame-by-frame safeguard (3.4 cm step: measurement noise regime)
    t_err, r_err, _ = check_pairs(Ts, poses)
    assert np.median(t_err) < 0.15 and r_err.max() < 1.0


def test_cumulative_drift():
    Ts = trajectory()
    _, poses = run_vo(Ts)
    _, _, gt = check_pairs(Ts, poses)
    dist = sum(np.linalg.norm(gt[k + 1][:3, 3] - gt[k][:3, 3]) for k in range(len(gt) - 1))
    err = np.linalg.norm(gt[-1][:3, 3] - poses[-1].T_world_cam[:3, 3])
    assert err / dist < 0.05


def test_outliers_30pct():
    Ts = trajectory()
    _, poses = run_vo(Ts, outliers=0.30)
    assert all(p.quality in ("ok", "init") for p in poses)
    te, re = check_kf_pairs(Ts, poses)
    assert np.median(te) < 0.03 and re.max() < 1.0
    t_err, r_err, gt = check_pairs(Ts, poses)
    assert r_err.max() < 1.0
    dist = sum(np.linalg.norm(gt[k + 1][:3, 3] - gt[k][:3, 3]) for k in range(len(gt) - 1))
    assert np.linalg.norm(gt[-1][:3, 3] - poses[-1].T_world_cam[:3, 3]) / dist < 0.05


def test_keyframes_created():
    vo, poses = run_vo(trajectory())
    assert 3 <= len(vo.keyframes) < 100
    assert sum(p.is_keyframe for p in poses) == len(vo.keyframes)


def test_textureless_is_lost_without_exception():
    cfg = load_config()
    src = VisualOdometry(cfg)
    assert isinstance(src, PoseSource)
    img = np.full((H, W, 3), 128, np.uint8)
    from visionaid.bus.messages import Depth

    for k in range(3):
        p = src.update(Frame(k / 30, k, img, K), Depth(k / 30, k, np.full((H, W), 2.0, np.float32)))
        assert p.quality == "lost"
    assert src.lost_fraction == 1.0


def test_lost_then_recover():
    cfg = load_config()
    vo = RGBDOdometry(cfg.vo)
    pts, desc = world()
    rng = np.random.default_rng(3)
    Ts = trajectory(40)
    empty = Observation(np.zeros((0, 2)), np.zeros((0, 32), np.uint8), np.zeros(0))
    q = []
    for k, T in enumerate(Ts):
        obs = empty if 15 <= k < 18 else observe(pts, desc, T, rng)
        q.append(vo.step(obs, K, k / 30, k).quality)
    assert q[15:18] == ["lost"] * 3
    assert q[19] == "ok" and q[-1] == "ok"


def test_arkit_stub():
    with pytest.raises(NotImplementedError):
        ArkitPoseSource()
