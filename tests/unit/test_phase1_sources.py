import shutil
import subprocess
import time

import numpy as np
import pytest

from visionaid.bus.messages import Frame
from visionaid.io.intrinsics import from_hfov, load_calibration
from visionaid.io.sources import (
    FrameSource,
    ListSource,
    ReplaySource,
    VideoFileSource,
    WebRTCSource,
)

pytestmark = pytest.mark.unit


def test_frame_count_and_timestamps(synthetic_video):
    path, n = synthetic_video
    src = VideoFileSource(path, width=None)
    assert isinstance(src, FrameSource)
    frames = list(src)
    assert len(frames) == n
    ts = np.array([f.t for f in frames])
    assert np.all(np.diff(ts) > 0)
    assert ts[-1] == pytest.approx((n - 1) / 30.0, abs=0.05)


def test_resize_keeps_aspect(synthetic_video):
    path, _ = synthetic_video
    src = VideoFileSource(path, width=160)
    f = src.read()
    assert f.image.shape == (120, 160, 3)
    K = src.intrinsics.K
    assert K[0, 2] == pytest.approx(80) and K[1, 2] == pytest.approx(60)


def test_rotation_override(synthetic_video):
    path, _ = synthetic_video
    f = VideoFileSource(path, width=None, rotation_override=90).read()
    assert f.image.shape == (320, 240, 3)
    # red band on the left -> at the top after clockwise rotation
    assert f.image[:10, :, 2].mean() > 200 and f.image[-10:, :, 2].mean() < 50


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_rotation_from_metadata(synthetic_video, tmp_path):
    path, _ = synthetic_video
    out = tmp_path / "rot.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-display_rotation",
            "-90",
            "-i",
            str(path),
            "-c",
            "copy",
            str(out),
        ],
        check=True,
        timeout=60,
    )
    src = VideoFileSource(out, width=None)
    assert src.rotation == 90
    f = src.read()
    assert f.image.shape == (320, 240, 3)
    assert f.image[:10, :, 2].mean() > 200


def test_replay_realtime_duration(synthetic_video):
    path, n = synthetic_video
    src = ReplaySource(VideoFileSource(path, width=None), speed=1.0)
    t0 = time.perf_counter()
    got = 0
    while src.read() is not None:
        got += 1
    dt = time.perf_counter() - t0
    expected = (n - 1) / 30.0
    assert abs(dt - expected) / expected < 0.10
    assert got == n and src.skipped == 0


def test_replay_skips_when_consumer_slow(synthetic_video):
    path, n = synthetic_video
    src = ReplaySource(VideoFileSource(path, width=None), speed=1.0)
    t0 = time.perf_counter()
    lags = []
    while (f := src.read()) is not None:
        lags.append((time.perf_counter() - t0) - f.t)
        time.sleep(0.1)  # consumer at 10 Hz for a source at 30 Hz
    assert src.skipped > n / 2
    assert max(lags) < 0.15  # no delay build-up


def test_list_source():
    from visionaid.io.intrinsics import Intrinsics

    intr = Intrinsics(100, 100, 50, 50, 100, 100)
    frames = [
        Frame(t=i / 10, frame_idx=i, image=np.zeros((100, 100, 3), np.uint8)) for i in range(5)
    ]
    s = ListSource(frames, intr, 10)
    assert sum(1 for _ in iter(s.read, None)) == 5


def test_webrtc_stub():
    with pytest.raises(NotImplementedError):
        WebRTCSource()


def test_intrinsics_from_hfov():
    intr = from_hfov(360, 640, 69.0)
    K = intr.K
    assert K[0, 2] == 180 and K[1, 2] == 320
    assert K[0, 0] > 0 and K[0, 0] == K[1, 1]
    assert K[0, 0] == pytest.approx(320 / np.tan(np.radians(34.5)))


def test_intrinsics_calibration(tmp_path):
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    np.savez(tmp_path / "c.npz", K=K, width=640, height=480)
    intr = load_calibration(tmp_path / "c.npz").scaled(320, 240)
    assert intr.fx == 250 and intr.cx == 160


def test_intrinsics_rotation():
    intr = from_hfov(640, 480, 60).rotated90()
    assert (intr.width, intr.height) == (480, 640)
