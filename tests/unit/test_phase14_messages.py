"""Phase 14: new messages (IMU, clock) and `Frame` compatibility."""

import numpy as np
import pytest

from visionaid.bus import messages as M
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit


def _roundtrip(msg):
    return M.decode(M.encode(msg))


def test_imu_sample_roundtrip():
    s = M.ImuSample(1.25, np.array([0.1, -0.2, 0.3]), np.array([0.0, 9.81, 0.1]))
    r = _roundtrip(s)
    assert r.t == 1.25
    np.testing.assert_allclose(r.gyro, s.gyro)
    np.testing.assert_allclose(r.accel, s.accel)
    assert r.accel_lin is None


def test_imu_batch_roundtrip_and_samples():
    samples = [M.ImuSample(0.01 * i, np.ones(3) * i, np.array([0, 9.81, 0])) for i in range(5)]
    b = M.ImuBatch.from_samples(samples, t_recv=123.0)
    assert len(b) == 5
    r = _roundtrip(b)
    assert r.t_recv == 123.0
    np.testing.assert_allclose(r.t, b.t)
    np.testing.assert_allclose(r.gyro, b.gyro)
    back = r.samples()
    assert [s.t for s in back] == pytest.approx([s.t for s in samples])
    assert len(M.ImuBatch.from_samples([])) == 0


def test_clock_sync_roundtrip():
    c = M.ClockSync(10.0, -0.0123, 0.004, 0.008)
    assert _roundtrip(c) == c


def test_frame_with_phone_times_roundtrip():
    img = np.zeros((4, 6, 3), np.uint8)
    f = M.Frame(5.0, 7, img, np.eye(3), t_phone=5.0, t_recv=1700000000.5)
    r = _roundtrip(f)
    assert r.t_phone == 5.0 and r.t_recv == 1700000000.5


def test_frame_without_phone_times_still_valid():
    img = np.zeros((4, 6, 3), np.uint8)
    f = M.Frame(1.0, 0, img, np.eye(3))
    r = _roundtrip(f)
    assert r.t_phone is None and r.t_recv is None
    np.testing.assert_array_equal(r.image, img)


def test_phase2_config_sections():
    cfg = load_config()
    assert cfg.phone.protocol_version == 1
    assert np.asarray(cfg.body.T_cam_body).shape == (4, 4)
    R = np.asarray(cfg.body.R_cam_imu, float)
    np.testing.assert_allclose(R @ R.T, np.eye(3))
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert cfg.body.camera_height_m > 0.5
