import numpy as np
import pytest

from visionaid.io.sources import VideoFileSource
from visionaid.perception.depth import object_height_m, video_depth_estimator
from visionaid.perception.pipeline import unpack_masks
from visionaid.utils.report import update_report

pytestmark = [pytest.mark.slow, pytest.mark.data]


@pytest.fixture(scope="module")
def depth_samples(video_path, cfg):
    """50 frames: 25 consecutive pairs spread over the video."""
    est = video_depth_estimator(cfg)
    src = VideoFileSource.from_config(cfg)
    starts = set(np.linspace(0, src.frame_count - 2, 25).astype(int).tolist())
    want = starts | {s + 1 for s in starts}
    out = {}
    while (f := src.read()) is not None:
        if f.frame_idx in want:
            out[f.frame_idx] = (f, est(f))
    return est, starts, out


def test_median_depth_indoor(depth_samples):
    est, _, out = depth_samples
    meds = np.array([np.median(d.depth) for _, d in out.values()])
    t = np.array(est.times) if est.times else np.array([np.nan])
    update_report(
        "depth",
        {
            "backend": est.backend.name,
            "median_depth_m": float(np.median(meds)),
            "ms_per_frame": float(np.nanmedian(t) * 1e3),
            "fps": float(1 / np.nanmedian(t)) if est.times else None,
        },
    )
    assert len(out) == 50
    assert ((meds >= 0.3) & (meds <= 8.0)).all(), meds


def test_temporal_consistency(depth_samples):
    _, starts, out = depth_samples
    rel = []
    for s in sorted(starts):
        a, b = out[s][1].depth, out[s + 1][1].depth
        rel.append(np.median(np.abs(a - b) / np.maximum(a, 1e-3)))
    med = float(np.median(rel))
    update_report("depth", {"consecutive_rel_change_median": med})
    assert med < 0.15


def test_scale_with_known_object(video_perception, cfg):
    """3D height of the bottle (mask + depth + K) within ±40% of target.height_m."""
    dets, _ = video_perception
    est = video_depth_estimator(cfg)
    src = VideoFileSource.from_config(cfg)
    by_idx = {d.frame_idx: d for d in dets if d.target_idx >= 0}
    heights = []
    while (f := src.read()) is not None:
        d = by_idx.get(f.frame_idx)
        if d is None or f.frame_idx % 3:
            continue
        x0, y0, x1, y1 = d.boxes[d.target_idx]
        h, w = f.image.shape[:2]
        if x0 < 3 or y0 < 3 or x1 > w - 3 or y1 > h - 3:
            continue  # object cut off by the edge
        masks = unpack_masks(d, w)
        hm = object_height_m(masks[d.target_idx], est(f).depth, f.K)
        if hm is not None:
            heights.append(hm)
    med = float(np.median(heights))
    target = cfg.target.height_m
    update_report(
        "depth",
        {"target_height_est_m": med, "target_height_true_m": target, "n_height_frames": len(heights), "scale_factor": cfg.depth.scale_factor},
    )
    assert len(heights) >= 5
    assert abs(med - target) / target <= 0.40, f"estimated height {med:.3f} m"
