"""Phase 18: fused pose VO + gyroscope + gravity, and scale re-anchoring by camera height."""

from __future__ import annotations

import numpy as np
import pytest

from visionaid.bus.messages import Depth, ImuBatch
from visionaid.calib.device import ImuCalib
from visionaid.calib.rotation import so3_log
from visionaid.mapping.geometry import inv_T, make_T, rodrigues, rot_angle_deg, transform
from visionaid.pose.imu import GravityFilter, ImuBuffer
from visionaid.pose.scale import DepthScaleCorrector
from visionaid.pose.vio import VisualInertialOdometry
from visionaid.pose.vo import Observation, RGBDOdometry
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit

W, H = 480, 640
K = np.array([[450.0, 0, W / 2], [0, 450.0, H / 2], [0, 0, 1]])
R_CAM_IMU = np.diag([1.0, -1.0, -1.0])
UP_WORLD = np.array([0.0, -1.0, 0.0])  # world of the phase 5 tests: y down
FPS = 30.0


def calib(delta=0.0):
    return ImuCalib(R_CAM_IMU, np.zeros(3), delta, 1.0, "test")


def world_front(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.stack([rng.uniform(-6, 6, n), rng.uniform(-2.5, 2.5, n), rng.uniform(3, 10, n)], 1)
    return pts, rng.integers(0, 256, (n, 32), dtype=np.uint8)


def world_around(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(4, 9, n)
    pts = np.stack([r * np.sin(a), rng.uniform(-2.5, 2.5, n), r * np.cos(a)], 1)
    return pts, rng.integers(0, 256, (n, 32), dtype=np.uint8)


def trajectory(n=100, step=0.04):
    Ts = []
    for k in range(n):
        a = 2 * np.pi * k / n
        t = np.array([0.5 * np.sin(a), 0.0, 0.5 * (1 - np.cos(a)) + step * k * 0.3])
        R = rodrigues(np.array([0.0, np.radians(15) * np.sin(a), 0.0]))
        Ts.append(make_T(R, t))
    return Ts


def fast_yaw_trajectory(n=150, peak_dps=150.0, freq=0.5):
    """Rotation on the spot, sinusoidal yaw (peak speed `peak_dps`), slight forward motion."""
    amp = np.radians(peak_dps) / (2 * np.pi * freq)
    Ts = []
    for k in range(n):
        t = k / FPS
        yaw = amp * np.sin(2 * np.pi * freq * t)
        Ts.append(make_T(rodrigues(np.array([0.0, yaw, 0.0])), np.array([0.0, 0.0, 0.01 * k])))
    return Ts


def observe(pts, desc, T_wc, rng, noise=0.5, keep=None):
    pc = transform(inv_T(T_wc), pts)
    ok = pc[:, 2] > 0.2
    uv = pc[ok, :2] / pc[ok, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv, z, d = uv[inside], pc[ok][inside, 2], desc[ok][inside]
    uv = uv + rng.normal(0, noise, uv.shape)
    if keep is not None and keep < len(uv):  # motion blur: few points detected
        s = rng.choice(len(uv), keep, replace=False)
        uv, z, d = uv[s], z[s], d[s]
    return Observation(uv=uv, desc=d, depth=z)


def imu_for(Ts, rate=120.0, acc_noise=0.05, gyro_noise=0.005, walk_acc=0.0, seed=0, delta=0.0):
    """IMU (device frame) consistent with the camera poses (images at FPS)."""
    rng = np.random.default_rng(seed)
    n = len(Ts)
    t_imu = np.arange(-0.6, (n - 1) / FPS + 0.2, 1 / rate)
    w_cam = np.zeros((len(t_imu), 3))
    a_cam = np.zeros((len(t_imu), 3))
    for i, t in enumerate(t_imu):
        k = int(np.clip(np.floor(t * FPS), 0, n - 2))
        Ra, Rb = Ts[k][:3, :3], Ts[k + 1][:3, :3]
        w_cam[i] = so3_log(Ra.T @ Rb) * FPS
        Rk = Ts[int(np.clip(round(t * FPS), 0, n - 1))][:3, :3]
        a_cam[i] = Rk.T @ UP_WORLD * 9.81
    a_cam += rng.normal(0, acc_noise, a_cam.shape) + rng.normal(0, walk_acc, a_cam.shape)
    w_cam += rng.normal(0, gyro_noise, w_cam.shape)
    # device frame: ω_imu = Rᵀ ω_cam; IMU clock = image clock + delta
    return ImuBatch(t_imu + delta, w_cam @ R_CAM_IMU, a_cam @ R_CAM_IMU)


def run(Ts, world, use_imu=True, blur_dps=None, cut=None, seed=0, imu=None):
    cfg = load_config()
    pts, desc = world
    rng = np.random.default_rng(seed + 1)
    est = VisualInertialOdometry(cfg, calib()) if use_imu else RGBDOdometry(cfg.vo)
    if use_imu:
        est.add_imu(imu if imu is not None else imu_for(Ts, seed=seed))
    poses = []
    for k, T in enumerate(Ts):
        keep = None
        if blur_dps is not None and k > 0:
            rate = rot_angle_deg(Ts[k - 1][:3, :3].T @ T[:3, :3]) * FPS
            if rate > blur_dps:
                keep = 15
        if cut is not None and cut[0] <= k < cut[1]:
            keep = 0
        poses.append(est.step(observe(pts, desc, T, rng, keep=keep), K, k / FPS, k))
    return est, poses


def test_vertical_from_gravity_under_2deg():
    Ts = trajectory()
    _, poses = run(Ts, world_front(), imu=imu_for(trajectory(), walk_acc=0.8))
    errs = []
    for p, T in zip(poses, Ts, strict=True):
        if p.quality in ("ok", "init"):
            up_cam_true = T[:3, :3].T @ UP_WORLD
            errs.append(
                np.degrees(np.arccos(np.clip((p.T_world_cam[:3, :3] @ up_cam_true)[2], -1, 1)))
            )
            assert p.up is not None
            np.testing.assert_allclose(p.up, [0, 0, 1], atol=0.05)
    assert len(errs) > 90 and max(errs) < 2.0


def test_fast_rotation_fewer_losses_than_vo():
    Ts = fast_yaw_trajectory()
    w = world_around()
    vo, vo_poses = run(Ts, w, use_imu=False, blur_dps=100)
    vio, vio_poses = run(Ts, w, use_imu=True, blur_dps=100)
    vo_lost = np.mean([p.quality == "lost" for p in vo_poses])
    vio_lost = np.mean([p.quality == "lost" for p in vio_poses])
    assert vo_lost > 0.1, "the scenario must make VO alone lose tracking"
    assert vio_lost <= 0.5 * vo_lost
    # orientation stays right despite the blur
    T0 = Ts[0]
    yaw_err = [
        rot_angle_deg((inv_T(T0) @ T)[:3, :3].T @ _vio_cam_rotation(p, vio_poses[0]))
        for p, T in zip(vio_poses, Ts, strict=True)
    ]
    assert np.max(yaw_err) < 3.0


def _vio_cam_rotation(p, p0):
    """Rotation relative to the first pose (comparable to the truth expressed in the first camera)."""
    return p0.T_world_cam[:3, :3].T @ p.T_world_cam[:3, :3]


def test_translation_at_least_as_good_as_vo():
    Ts = trajectory()
    w = world_front()

    def drift(poses):
        gt = [inv_T(Ts[0]) @ T for T in Ts]
        est = [inv_T(poses[0].T_world_cam) @ p.T_world_cam for p in poses]
        dist = sum(np.linalg.norm(gt[k + 1][:3, 3] - gt[k][:3, 3]) for k in range(len(gt) - 1))
        return np.linalg.norm(gt[-1][:3, 3] - est[-1][:3, 3]) / dist

    _, vo_p = run(Ts, w, use_imu=False)
    _, vio_p = run(Ts, w, use_imu=True)
    assert all(p.quality in ("ok", "init") for p in vio_p)
    assert drift(vio_p) <= max(1.2 * drift(vo_p), 0.01) and drift(vio_p) < 0.05


def test_vo_cut_one_second_keeps_orientation():
    Ts = fast_yaw_trajectory(n=120, peak_dps=60)
    w = world_around()
    cut = (40, 70)  # 1 s without any point (camera covered)
    est, poses = run(Ts, w, cut=cut)
    during = [p.quality for p in poses[cut[0] : cut[1]]]
    assert all(q == "degraded" for q in during), during
    err = rot_angle_deg(
        (inv_T(Ts[0]) @ Ts[cut[1] - 1])[:3, :3].T @ _vio_cam_rotation(poses[cut[1] - 1], poses[0])
    )
    assert err < 3.0
    after = [p.quality for p in poses[cut[1] : cut[1] + 6]]
    assert "ok" in after[:4], after
    assert est.stats()["degraded_fraction"] > 0.2


def test_degraded_becomes_lost_after_timeout():
    Ts = fast_yaw_trajectory(n=120, peak_dps=30)
    _, poses = run(Ts, world_around(), cut=(20, 110))  # 3 s without vision (> 2 s)
    qs = [p.quality for p in poses[20:110]]
    assert qs[0] == "degraded" and qs[-1] == "lost"
    first_lost = qs.index("lost")
    assert 2.0 * FPS - 2 <= first_lost <= 2.0 * FPS + 2


def test_without_imu_behaves_like_vo():
    Ts = trajectory(n=40)
    cfg = load_config()
    est = VisualInertialOdometry(cfg, calib())
    pts, desc = world_front()
    rng = np.random.default_rng(1)
    poses = [est.step(observe(pts, desc, T, rng), K, k / FPS, k) for k, T in enumerate(Ts)]
    assert all(p.quality in ("ok", "init") for p in poses) and poses[-1].up is None


def test_imu_buffer_time_offset_and_rotation():
    Ts = fast_yaw_trajectory(n=60, peak_dps=90)
    b = ImuBuffer(calib(delta=0.04))
    b.add(imu_for(Ts, delta=0.04))
    for k in (10, 30, 45):
        R = b.delta_R(k / FPS, (k + 1) / FPS)
        R_true = Ts[k][:3, :3].T @ Ts[k + 1][:3, :3]
        assert rot_angle_deg(R.T @ R_true) < 0.15
    assert b.delta_R(100.0, 101.0) is None  # outside coverage


def test_gravity_filter_tracks_rotation():
    g = GravityFilter(gain=0.05)
    assert g.init(np.array([0.0, -9.81, 0.0]))
    R = rodrigues(np.array([np.radians(30), 0, 0]))
    g.predict(R)
    np.testing.assert_allclose(g.up, R.T @ [0, -1, 0], atol=1e-9)
    g.correct(np.array([0.0, -30.0, 0.0]))  # shock: ignored
    assert g.n_corrections == 0


def floor_depth(h=480, w=320, pitch_deg=15.0, cam_h=1.3, box=True):
    Kf = np.array([[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]])
    v, u = np.mgrid[0:h, 0:w]
    y = (v - Kf[1, 2]) / Kf[1, 1]
    p = np.radians(pitch_deg)
    den = np.cos(p) * y + np.sin(p)
    d = np.where(den > 1e-3, cam_h / np.maximum(den, 1e-3), 8.0)
    d = np.minimum(d, 6.0)
    if box:  # furniture in the foreground at the bottom left
        d[int(h * 0.7) :, : int(w * 0.3)] = 1.2
    up_cam = -np.array([0.0, np.cos(p), np.sin(p)])
    return d.astype(np.float32), Kf, up_cam


def test_depth_scale_converges_to_camera_height():
    d, Kf, up = floor_depth()
    sc = DepthScaleCorrector(camera_height_m=1.3, ema=0.2)
    biased = Depth(0.0, 0, d * 1.3, None)  # depth overestimated by 30%
    for _ in range(30):
        out = sc(biased, Kf, up)
    assert sc.scale == pytest.approx(1 / 1.3, rel=0.05)
    np.testing.assert_allclose(out.depth, d, rtol=0.06)
    # without gravity: factor unchanged
    assert DepthScaleCorrector(1.3).update(d, Kf, None) == 1.0


def test_depth_scale_perfectly_flat_floor():
    """Edge case: noiseless floor (all heights in a single histogram bin)."""
    d, Kf, up = floor_depth(box=False)
    sc = DepthScaleCorrector(camera_height_m=1.2, ema=1.0, n_init=1)
    sc.update(d, Kf, up)
    assert sc.n_updates == 1 and sc.scale == pytest.approx(1.2 / 1.3, rel=0.01)


def test_depth_scale_ignores_dominant_table():
    """Real trial: the camera points at a table covering the bottom of the image; the floor, lower, is
    only a secondary mode. The floor (not the table) must give the scale, and an isolated
    outlier measurement must not make the factor jump."""
    d, Kf, up = floor_depth(box=False, pitch_deg=30.0, cam_h=1.5)
    h = d.shape[0]
    table = d.copy()
    # table top 0.75 m below the camera over 70% of the bottom of the image
    v, _u = np.mgrid[0:h, 0 : d.shape[1]]
    y = (v - Kf[1, 2]) / Kf[1, 1]
    p = np.radians(30.0)
    den = np.cos(p) * y + np.sin(p)
    tab = np.where(den > 1e-3, 0.75 / np.maximum(den, 1e-3), 8.0)
    table[int(h * 0.6) :, :] = tab[int(h * 0.6) :, :]
    sc = DepthScaleCorrector(camera_height_m=1.5, ema=0.2)
    for _ in range(10):
        sc.update(table, Kf, up)
    assert sc.scale == pytest.approx(1.0, abs=0.05)
