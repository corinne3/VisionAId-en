"""Full pipeline on a synthetic video with stubs (without models): checks the wiring."""

import time

import cv2
import numpy as np
import pytest

from visionaid.perception.depth import DepthEstimator, StubDepthBackend
from visionaid.perception.detector import StubDetector
from visionaid.perception.pipeline import PerceptionBlock
from visionaid.perception.reid import StubEmbedder, TargetReID, masked_crop
from visionaid.utils.config import load_config

pytestmark = [pytest.mark.integration, pytest.mark.unit]

W, H = 320, 480
BOX = np.array([150, 300, 175, 360], float)  # bottle on the floor, ~3 m ahead


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


def write_video(path, n=45):
    rng = np.random.default_rng(0)
    base = scene_image(rng)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    for _ in range(n):
        vw.write(
            np.clip(base.astype(int) + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
        )
    vw.release()
    return base


def test_full_pipeline_with_stubs(tmp_path):
    from visionaid.apps.run_offline import run
    from visionaid.io.intrinsics import from_hfov

    t0 = time.perf_counter()
    video = tmp_path / "synth.mp4"
    base = write_video(video)
    cfg = load_config(
        overrides={
            "video.width": W,
            "paths.cache": str(tmp_path / "cache"),
            "depth.scale_factor": 1.0,
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
        cfg, video, out, perception=perception, depth_est=depth, rerun=True, report_section=None
    )
    assert time.perf_counter() - t0 < 30
    for name in ("annotated.mp4", "session.rrd", "guidance.wav"):
        p = out / name
        assert p.exists() and p.stat().st_size > 0, name
    assert summ["n_frames"] == 45
    assert summ["confirmed_fraction"] > 0.5
    assert summ["final_state"] in ("NAVIGATING", "APPROACHING", "ARRIVED", "OBSTACLE_ALERT")
    assert summ["n_plans"] > 0
