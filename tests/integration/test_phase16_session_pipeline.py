"""Phase 16 (integration): offline pipeline and real-time source on a phone session."""

import time

import cv2
import numpy as np
import pytest

from visionaid.io.session import write_synthetic_session
from visionaid.perception.depth import DepthEstimator, StubDepthBackend
from visionaid.perception.detector import StubDetector
from visionaid.perception.pipeline import PerceptionBlock
from visionaid.perception.reid import StubEmbedder, TargetReID, masked_crop
from visionaid.utils.config import load_config, load_raw

pytestmark = [pytest.mark.integration, pytest.mark.unit]

W, H = 320, 480
BOX = np.array([150, 300, 175, 360], float)


def scene_image(rng):
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    x0, y0, x1, y1 = BOX.astype(int)
    img[y0 : y0 + 35, x0:x1] = (150, 150, 150)
    img[y0 + 35 : y1, x0:x1] = (0, 128, 255)
    return img


def scene_depth(h, w, K, pitch=np.radians(20), cam_h=1.4, wall=5.0):
    v, u = np.mgrid[0:h, 0:w]
    y = (v - K[1, 2]) / K[1, 1]
    den = np.cos(pitch) * y + np.sin(pitch)
    floor = np.where(den > 1e-3, cam_h / np.maximum(den, 1e-3), np.inf)
    return np.minimum(floor, wall).astype(np.float32)


def make_session(path, n=45):
    rng = np.random.default_rng(0)
    base = scene_image(rng)
    imgs = [
        np.clip(base.astype(int) + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
        for _ in range(n)
    ]
    write_synthetic_session(
        path,
        imgs,
        fps=15,
        # camera tilted 20° downwards, like `scene_depth` (gravity consistent with the scene)
        imu_fn=lambda t: (
            np.zeros(3),
            9.81 * np.array([0.0, np.cos(np.radians(20)), np.sin(np.radians(20))]),
        ),
        meta={"target": "où est ma gourde"},
    )
    return base


def test_offline_pipeline_on_session(tmp_path):
    from visionaid.apps.run_offline import run
    from visionaid.io.intrinsics import from_hfov

    sess = tmp_path / "sess"
    base = make_session(sess)
    cfg = load_config(
        overrides={
            "video.width": W,
            "paths.cache": str(tmp_path / "cache"),
            "depth.scale_factor": 1.0,
            "phone.calib_dir": str(tmp_path / "calib"),  # without the real iPhone calibration
        }
    )
    K = from_hfov(W, H, cfg.intrinsics.hfov_deg).K
    reid = TargetReID(cfg, embedder=StubEmbedder(), backend="stub")
    crop, m = masked_crop(base, BOX, None)
    reid.build_gallery([(crop, m)], n_aug=2)
    reid.threshold = 0.3
    det = StubDetector(fn=lambda img, k: [(BOX, 0.9, "bottle")])
    det.backend, det.weights, det.device = "stub", "stub", "cpu"
    perception = PerceptionBlock(cfg, detector=det, reid=reid)
    depth = DepthEstimator(cfg, backend=StubDepthBackend(lambda h, w: scene_depth(h, w, K)))
    out = tmp_path / "out"
    summ = run(
        cfg, sess, out, perception=perception, depth_est=depth, rerun=False, report_section=None
    )
    assert summ["n_frames"] == 45
    assert summ["confirmed_fraction"] > 0.5
    assert summ["n_plans"] > 0
    assert (out / "annotated.mp4").stat().st_size > 0


def test_session_replay_block_emits_frames_and_imu(tmp_path):
    from visionaid.apps.realtime_blocks import SessionReplayBlock

    sess = tmp_path / "sess"
    make_session(sess, n=12)
    raw = load_raw()
    raw["_session"] = str(sess)
    raw["video"]["width"] = W
    raw["_t0_wall"] = time.time()
    blk = SessionReplayBlock(raw)
    outs = [blk.process({}) for _ in range(14)]  # 12 images then loop
    frames = [o["frame"] for o in outs if o]
    assert frames[0].t == pytest.approx(0.0) and frames[0].t_phone > 1e9
    assert all(b.frame_idx > a.frame_idx for a, b in zip(frames, frames[1:], strict=False))
    assert all(b.t > a.t for a, b in zip(frames, frames[1:], strict=False))
    assert blk.loops == 1
    imus = [o["imu"] for o in outs if o and "imu" in o]
    assert len(imus) >= 10
    for o in outs[1:11]:  # IMU brought into the same time frame as the image
        assert o["imu"].t[-1] <= o["frame"].t + 1e-9
        assert o["frame"].t - o["imu"].t[-1] < 1 / 60 + 1e-6
