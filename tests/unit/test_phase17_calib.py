"""Phase 17: camera (checkerboard) and IMU ↔ camera calibration on synthetic data."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
import yaml

from visionaid.calib.camera import calibrate_images, render_board_view, square_size_from_notes
from visionaid.calib.device import load_imu_calib, phone_intrinsics
from visionaid.calib.imu_cam import calibrate_imu_cam, static_mask
from visionaid.calib.rotation import VisualRotation, rotation_angle_deg, so3_exp
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit

W, H = 480, 640
K_TRUE = np.array([[520.0, 0, 243.0], [0, 523.0, 318.0], [0, 0, 1]])


def test_camera_calibration_recovers_K():
    rng = np.random.default_rng(1)
    views = []
    for _ in range(14):
        rvec = rng.uniform(-0.5, 0.5, 3)
        rvec[2] = rng.uniform(-0.3, 0.3)
        R, _ = cv2.Rodrigues(rvec)
        center = np.array([0.1, 0.0625, 0.0])  # centre of the checkerboard (checkerboard frame)
        t = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.05, 0.05), rng.uniform(0.35, 0.55)])
        tvec = t - R @ center
        views.append(render_board_view(K_TRUE, rvec, tvec, (W, H)))
    views.append(np.full((H, W, 3), 128, np.uint8))  # uniform view: rejected (blurred)
    views.append(_texture(5))  # sharp view without checkerboard: rejected
    cal = calibrate_images(views, (9, 6), 0.025)
    assert cal.n_views >= 12
    assert cal.rejected.get("vue_14") == "floue"
    assert cal.rejected.get("vue_15") == "checkerboard not found"
    for k, (i, j) in enumerate([(0, 0), (1, 1), (0, 2), (1, 2)]):
        assert cal.K[i, j] == pytest.approx(K_TRUE[i, j], rel=0.02), k
    assert cal.rms_px < 0.5
    assert 40 < cal.hfov_long_side_deg < 80


def test_square_size_from_notes():
    assert square_size_from_notes("case = 24,5 mm, écran 14 pouces") == pytest.approx(0.0245)
    assert square_size_from_notes("Case: 30mm") == pytest.approx(0.030)
    assert square_size_from_notes("rien") is None


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.random((H, W)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_visual_rotation_sign_convention(axis):
    """The camera turns by ω·dt: the visual estimate must return ω in the camera frame."""
    img1 = _texture(axis)
    w_true = np.zeros(3)
    w_true[axis] = 0.6  # rad/s
    dt = 1 / 15
    R_c1_c2 = so3_exp(w_true * dt)
    R_c2_c1 = R_c1_c2.T
    Hm = K_TRUE @ R_c2_c1 @ np.linalg.inv(K_TRUE)
    img2 = cv2.warpPerspective(img1, Hm, (W, H), borderMode=cv2.BORDER_REFLECT)
    w, n = VisualRotation().angular_velocity(img1, img2, K_TRUE, dt)
    assert n > 50
    np.testing.assert_allclose(w, w_true, atol=0.03)


def _profile(t):
    """Camera angular velocity (rad/s): still for 3 s, varied movements, still for 3 s."""
    w = np.zeros((len(t), 3))
    m = (t > 3) & (t < 23)
    tt = t[m]
    w[m, 0] = 0.4 * np.sin(2 * np.pi * 0.31 * tt) + 0.2 * np.sin(2 * np.pi * 1.1 * tt)
    w[m, 1] = 1.2 * np.sin(2 * np.pi * 0.23 * tt + 1) + 0.4 * np.sin(2 * np.pi * 0.9 * tt)
    w[m, 2] = 0.3 * np.sin(2 * np.pi * 0.41 * tt + 2)
    env = np.clip(np.minimum(t - 3, 23 - t), 0, 1)  # smooth rise / fall
    return w * env[:, None]


def _synthetic(delta=0.04, bias=(0.01, -0.02, 0.015), gyro_units_factor=1.0, seed=0):
    rng = np.random.default_rng(seed)
    R_prior = np.diag([1.0, -1.0, -1.0])
    R_true = so3_exp(np.radians([2.0, -1.5, 2.5])) @ R_prior
    t_imu = np.arange(0, 26, 1 / 60) + 1000.0
    w_cam = _profile(t_imu - 1000.0)
    w_imu = w_cam @ R_true  # ω_cam = R ω_imu  ⇔  ω_imu = Rᵀ ω_cam
    gyro = (w_imu + np.array(bias) + rng.normal(0, 0.01, w_imu.shape)) * gyro_units_factor
    acc = np.array([0.0, 9.81, 0.0]) + rng.normal(0, 0.03, (len(t_imu), 3))
    moving = np.linalg.norm(w_cam, axis=1) > 0.05
    acc[moving] += rng.normal(0, 0.8, (moving.sum(), 3))
    # images at 15 Hz; the image timestamped t corresponds to the IMU instant t + delta
    t_f = np.arange(0.5, 25.5, 1 / 15) + 1000.0
    t0, t1 = t_f[:-1], t_f[1:]
    w_vis = []
    fine = np.arange(0, 26, 1 / 1000) + 1000.0
    wc_fine = _profile(fine - 1000.0)
    for a, b in zip(t0 + delta, t1 + delta, strict=True):
        m = (fine >= a) & (fine <= b)
        w_vis.append(wc_fine[m].mean(0))
    w_vis = np.array(w_vis) + rng.normal(0, 0.01, (len(t0), 3))
    return t_imu, gyro, acc, t0, t1, w_vis, R_true, R_prior


def test_imu_cam_calibration_recovers_bias_offset_rotation():
    bias = np.array([0.01, -0.02, 0.015])
    t_imu, gyro, acc, t0, t1, w_vis, R_true, R_prior = _synthetic(delta=0.04, bias=bias)
    cal = calibrate_imu_cam(t_imu, gyro, acc, t0, t1, w_vis, R_prior)
    assert cal.time_offset_s == pytest.approx(0.04, abs=0.005)
    np.testing.assert_allclose(cal.gyro_bias, bias, rtol=0.1, atol=0.001)
    assert rotation_angle_deg(cal.R_cam_imu @ R_true.T) < 2.0
    assert cal.units_ok and cal.units_fix == 1.0
    assert cal.correlation > 0.9 and 2 < cal.prior_angle_deg < 6
    assert cal.static_s > 4


def test_imu_cam_detects_wrong_gyro_units():
    # gyroscope in deg/s treated as rad/s: values 57.3× too large
    t_imu, gyro, acc, t0, t1, w_vis, R_true, R_prior = _synthetic(gyro_units_factor=180 / math.pi)
    cal = calibrate_imu_cam(t_imu, gyro, acc, t0, t1, w_vis, R_prior)
    assert not cal.units_ok
    assert cal.units_fix == pytest.approx(math.pi / 180)
    assert any("units" in n for n in cal.notes)
    assert rotation_angle_deg(cal.R_cam_imu @ R_true.T) < 2.0


def test_static_mask():
    t = np.arange(0, 4, 0.01)
    g = np.zeros((len(t), 3))
    g[(t > 1) & (t < 3)] = np.random.default_rng(0).normal(0, 0.5, (199, 3))
    m = static_mask(t, g, np.tile([0, 9.81, 0], (len(t), 1)))
    assert m[t < 0.9].mean() > 0.9 and m[(t > 1.2) & (t < 2.8)].mean() < 0.05


def test_device_calibration_loading(tmp_path):
    cfg = load_config(overrides={"phone.calib_dir": str(tmp_path), "phone.device": "testphone"})
    # without a file: field of view and prior
    k0 = phone_intrinsics(cfg, 480, 640)
    imu0 = load_imu_calib(cfg)
    assert imu0.source == "a priori" and imu0.time_offset_s == 0.0
    # 480×640 portrait calibration: applied with scaling, and rotated for landscape
    (tmp_path / "testphone_camera.yaml").write_text(
        yaml.safe_dump({"K": K_TRUE.tolist(), "width": W, "height": H})
    )
    k1 = phone_intrinsics(cfg, 240, 320)
    assert k1.fx == pytest.approx(260) and k1.cy == pytest.approx(159)
    k2 = phone_intrinsics(cfg, 640, 480)
    assert k2.width == 640 and k2.fx == pytest.approx(523)
    assert phone_intrinsics(cfg, 500, 500).fx == pytest.approx(
        load_config().intrinsics.hfov_deg and k0.fx * 500 / 640, rel=1e-6
    )
    (tmp_path / "testphone_imu.yaml").write_text(
        yaml.safe_dump(
            {
                "R_cam_imu": np.eye(3).tolist(),
                "gyro_bias": [0.1, 0, 0],
                "time_offset_s": 0.03,
                "units_fix": 1.0,
            }
        )
    )
    imu1 = load_imu_calib(cfg)
    assert imu1.source == "calibration" and imu1.time_offset_s == 0.03


def test_units_in_cm_or_mm():
    from visionaid.utils.config import length_to_m

    assert square_size_from_notes("Case =2.2cm") == pytest.approx(0.022)
    assert square_size_from_notes("case=22mm") == pytest.approx(0.022)
    assert square_size_from_notes("case: 0,022 m") == pytest.approx(0.022)
    assert square_size_from_notes("case = 22") == pytest.approx(0.022)  # without a unit: mm
    assert square_size_from_notes("case 2,2") == pytest.approx(0.022)  # small number: cm
    assert length_to_m(150) == pytest.approx(1.5)
    assert length_to_m(1.5) == pytest.approx(1.5)
    assert length_to_m(1500) == pytest.approx(1.5)
    assert length_to_m(None) is None


def test_safari_ios_axes_and_accel_sign():
    """Safari iOS: gyroscope [alpha, beta, gamma] = axes (x, y, z) instead of (z, x, y), and
    accelerometer with inverted sign. Calibration absorbs the permutation in R_cam_imu and
    measures the accelerometer sign, applied separately by the VIO."""
    from visionaid.calib.device import ImuCalib
    from visionaid.pose.imu import ImuBuffer

    t_imu, gyro, acc, t0, t1, w_vis, R_true, R_prior = _synthetic(bias=(0, 0, 0))
    # the page sends [beta, gamma, alpha]; under iOS these are the axes (y, z, x)
    gyro_page = gyro[:, [1, 2, 0]]
    acc_ios = -acc  # gravity with inverted sign
    cal = calibrate_imu_cam(t_imu, gyro_page, acc_ios, t0, t1, w_vis, R_prior)
    assert cal.units_ok and cal.correlation > 0.9
    assert 100 < cal.prior_angle_deg < 140  # permutation cyclique ≈ 120°
    assert cal.accel_sign == -1.0 and any("inverted sign" in n for n in cal.notes)
    # applied: gyroscope AND gravity right in the camera frame
    ic = ImuCalib(cal.R_cam_imu, cal.gyro_bias, 0.0, 1.0, "test", cal.R_cam_acc, cal.accel_sign)
    from visionaid.bus.messages import ImuBatch

    b = ImuBuffer(ic)
    b.add(ImuBatch(t_imu[:50], gyro_page[:50], acc_ios[:50]))
    a_mean, _ = b.accel_mean(t_imu[0], t_imu[49])
    up_cam = R_prior @ np.array([0.0, 1.0, 0.0])  # device +y in the camera frame
    assert np.degrees(np.arccos(a_mean @ up_cam / np.linalg.norm(a_mean))) < 3
