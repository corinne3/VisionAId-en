"""Phase 16: reading and replaying phone sessions."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest

from visionaid.apps.session_info import summarize, write_video
from visionaid.io.session import (
    SessionImu,
    SessionSource,
    is_session_dir,
    load_session,
    media_hash,
    open_media,
    write_synthetic_session,
)
from visionaid.io.sources import ReplaySource
from visionaid.phone.fake_phone import synthetic_frames
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit


def _imgs(n, w=320, h=240):
    it = synthetic_frames(w, h)
    return [next(it) for _ in range(n)]


def _imu(t):
    return np.array([0.0, 0.1 * t, 0.0]), np.array([0.0, 9.81, 0.0])


@pytest.fixture
def session_dir(tmp_path):
    return write_synthetic_session(
        tmp_path / "s1", _imgs(30), fps=15, imu_fn=_imu, imu_hz=60, calib_images=_imgs(2)
    )


def test_load_and_replay_identical(session_dir):
    s = load_session(session_dir)
    assert is_session_dir(session_dir) and len(s.frames) == 30
    assert s.fps == pytest.approx(15, rel=0.01) and len(s.calib_files) == 2
    src = SessionSource(s, width=None)
    frames = list(src)
    assert len(frames) == 30
    assert [f.frame_idx for f in frames] == [r["seq"] for r in s.frames]
    assert [f.t for f in frames] == [r["t_phone"] for r in s.frames]
    # images identical to the recorded JPEG (decoding without additional loss)
    raw = cv2.imread(str(session_dir / s.frames[5]["file"]))
    np.testing.assert_array_equal(frames[5].image, raw)
    assert frames[0].t_phone == frames[0].t and frames[0].t_recv is not None
    # two reads give exactly the same thing (determinism)
    again = list(SessionSource(s, width=None))
    assert all(np.array_equal(a.image, b.image) for a, b in zip(frames, again, strict=True))


def test_width_reduction_and_intrinsics(session_dir):
    cfg = load_config()
    src = SessionSource(session_dir, width=160, cfg_intrinsics=cfg.intrinsics)
    f = src.read()
    assert f.image.shape[:2] == (120, 160)
    assert f.K[0, 2] == pytest.approx(80) and f.K[1, 2] == pytest.approx(60)
    assert SessionSource(session_dir, width=1000).read().image.shape[1] == 320  # never enlarged


def test_imu_window_covers_inter_frame_interval(session_dir):
    s = load_session(session_dir)
    imu = SessionImu(s)
    assert imu.rate() == pytest.approx(60, rel=0.02)
    prev = None
    total = 0
    for f in SessionSource(s, width=None):
        b = imu.window(prev, f.t)
        if prev is not None:
            assert len(b) >= 3  # ~4 samples at 60 Hz between two images at 15 Hz
            assert b.t[0] > prev and b.t[-1] <= f.t
            assert f.t - b.t[-1] < 1 / 60 + 1e-9  # no gap at the end of the window
        total += len(b)
        prev = f.t
    # successive windows partition the IMU up to the last image
    assert total == int(np.sum(s.imu_t <= s.frames[-1]["t_phone"]))


def test_replay_real_time_pacing(session_dir):
    src = ReplaySource(SessionSource(session_dir, width=None), 1.0)
    t0 = time.perf_counter()
    n = sum(1 for _ in iter(src.read, None))
    el = time.perf_counter() - t0
    assert n == 30 and el == pytest.approx(29 / 15, rel=0.1)


def test_open_media_and_hash(session_dir, tmp_path):
    cfg = load_config()
    assert isinstance(open_media(cfg, session_dir), SessionSource)
    h1 = media_hash(session_dir)
    assert h1 == media_hash(session_dir) and h1 != media_hash(session_dir, "autre")


def test_truncated_last_line_tolerated(session_dir):
    with open(session_dir / "imu.jsonl", "a") as fh:
        fh.write('{"t": 1.0, "gyro": [0,')  # abrupt stop during writing
    s = load_session(session_dir)
    assert len(s.imu_t) > 100


def test_summary_and_video(session_dir):
    info = summarize(load_session(session_dir))
    assert info["n_frames"] == 30 and info["gaps_over_3_periods"] == 0
    assert info["imu_hz"] == pytest.approx(60, rel=0.02)
    assert info["imu_coverage"]["imu_starts_before_first_frame_s"] > 0
    assert info["frame_age_ms_p50"] == pytest.approx(30, abs=1)
    json.dumps(info)
    v = write_video(load_session(session_dir))
    cap = cv2.VideoCapture(str(v))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
